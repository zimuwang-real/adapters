"""Exercise the real MCP app with a deterministic, credential-free runtime."""

import ast
import asyncio
import io
import json
import runpy
import sys
import urllib.error
import urllib.request
from pathlib import Path
from threading import Thread
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("fastmcp")
import httpx
from fastmcp import Client

SERVER = (
    Path(__file__).parents[1]
    / "src/tau3_bench/task-template/environment/runtime-server/server.py"
)


@pytest.fixture()
def http_module():
    fixture_runtime = SimpleNamespace(
        environment=SimpleNamespace(get_tools=lambda: []),
        evaluation_calls=0,
        start_conversation=lambda: "offline customer observation",
    )

    def evaluate():
        fixture_runtime.evaluation_calls += 1
        return 200, {
            "status": "passed",
            "reward": 1.0,
            "reward_info": {"hidden": "criteria"},
        }

    fixture_runtime.evaluate = evaluate
    tree = ast.parse(SERVER.read_text(), filename=str(SERVER))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "runtime"
            for target in node.targets
        ):
            node.value = ast.Name(id="fixture_runtime", ctx=ast.Load())
    ast.fix_missing_locations(tree)
    module = ModuleType("tau3_http_under_test")
    module.fixture_runtime = fixture_runtime
    exec(compile(tree, str(SERVER), "exec"), module.__dict__)
    return module


def test_candidate_cannot_trigger_or_read_authoritative_evaluation(http_module):
    async def candidate_request():
        app = http_module.mcp.http_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://candidate-runtime"
        ) as client:
            return await client.post("/evaluate", json={})

    response = asyncio.run(candidate_request())
    assert response.status_code == 404
    assert http_module.runtime.evaluation_calls == 0
    assert "criteria" not in response.text


def test_candidate_can_still_call_conversation_tools(http_module):
    async def conversation():
        async with Client(http_module.mcp) as client:
            return await client.call_tool("start_conversation")

    result = asyncio.run(conversation())
    assert result.content[0].text == "offline customer observation"


@pytest.fixture()
def configurable_runtime(http_module, monkeypatch):
    # Isolate model settings from the developer's environment and credentials.
    monkeypatch.setattr(http_module.os, "environ", {})
    runtime = http_module.Tau3Runtime.__new__(http_module.Tau3Runtime)
    runtime.termination_reason = None
    runtime.bootstrap_complete = False
    runtime.seed = None
    runtime.max_steps = 200
    runtime.max_errors = 10
    runtime.step_count = 0
    runtime.num_errors = 0
    runtime.user = SimpleNamespace(llm_args={"temperature": 0.0})
    runtime._write_state = lambda: None
    http_module.runtime = runtime
    return runtime


def test_public_configuration_schema_excludes_user_model_overrides(http_module):
    async def list_tools():
        async with Client(http_module.mcp) as client:
            return await client.list_tools()

    tools = asyncio.run(list_tools())
    configure = next(tool for tool in tools if tool.name == "configure_run")
    assert set(configure.input_schema["properties"]) == {
        "seed",
        "max_steps",
        "max_errors",
    }


def test_candidate_cannot_inject_mock_user_responses(http_module, configurable_runtime):
    async def attack():
        async with Client(http_module.mcp) as client:
            return await client.call_tool(
                "configure_run",
                {
                    "seed": 99,
                    "user_llm_args_json": json.dumps(
                        {
                            "mock_response": "forged customer approval",
                            "mock_tool_calls": [],
                        }
                    ),
                },
                raise_on_error=False,
            )

    result = asyncio.run(attack())
    assert result.is_error
    assert configurable_runtime.user.llm_args == {"temperature": 0.0}
    assert configurable_runtime.seed is None


def test_runtime_configuration_rejects_candidate_model_kwargs(configurable_runtime):
    with pytest.raises(TypeError, match="user_llm_args_json"):
        configurable_runtime.configure_run(
            user_llm_args_json='{"mock_response":"forged customer approval"}'
        )
    assert configurable_runtime.user.llm_args == {"temperature": 0.0}


@pytest.mark.parametrize("host_override", [None, {"temperature": 0.25}])
def test_parity_configuration_preserves_host_owned_model_settings(
    http_module, configurable_runtime, host_override
):
    if host_override is not None:
        http_module.os.environ["TAU2_USER_LLM_ARGS_JSON"] = json.dumps(host_override)

    async def configure():
        async with Client(http_module.mcp) as client:
            return await client.call_tool(
                "configure_run", {"seed": 17, "max_steps": 40, "max_errors": 3}
            )

    result = asyncio.run(configure())
    status = json.loads(result.content[0].text)
    assert status == {
        "termination_reason": None,
        "step_count": 0,
        "num_errors": 0,
        "max_steps": 40,
        "max_errors": 3,
        "seed": 17,
    }
    expected_settings = host_override or {"reasoning_effort": "low"}
    assert configurable_runtime.user.llm_args == {**expected_settings, "seed": 17}


@pytest.mark.parametrize("status", [200, 409])
def test_export_collects_loopback_result_without_initializing_task(
    monkeypatch, tmp_path, status
):
    output = tmp_path / "private/evaluation.json"
    payload = {
        "status": "passed" if status == 200 else "not_terminated",
        "reward": 1.0 if status == 200 else 0.0,
        "reward_basis": [],
    }
    requested_urls = []

    def fetch(request, timeout):
        requested_urls.append(request.full_url)
        body = io.BytesIO(json.dumps(payload).encode())
        if status == 409:
            raise urllib.error.HTTPError(
                request.full_url, status, "unfinished", {}, body
            )
        return body

    monkeypatch.setattr(urllib.request, "urlopen", fetch)
    monkeypatch.setattr(sys, "argv", [str(SERVER), "--export-evaluation", str(output)])
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(SERVER), run_name="__main__")
    assert exited.value.code == 0
    assert requested_urls == ["http://127.0.0.1:8001/evaluate"]
    assert json.loads(output.read_text()) == payload


def test_export_does_not_turn_evaluator_failure_into_reward(monkeypatch, tmp_path):
    output = tmp_path / "evaluation.json"

    def fail(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            500,
            "failed",
            {},
            io.BytesIO(b'{"status":"evaluation_error"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    monkeypatch.setattr(sys, "argv", [str(SERVER), "--export-evaluation", str(output)])
    with pytest.raises(urllib.error.HTTPError):
        runpy.run_path(str(SERVER), run_name="__main__")
    assert not output.exists()


@pytest.mark.parametrize("outcome", ["success", "not_terminated", "error"])
def test_private_listener_binds_loopback_and_preserves_evaluation_outcome(
    http_module, outcome
):
    def evaluate():
        if outcome == "error":
            raise RuntimeError("sensitive internal detail")
        if outcome == "not_terminated":
            return 409, {"status": "not_terminated", "reward": 0.0, "reward_basis": []}
        return 200, {"status": "passed", "reward": 1.0}

    http_module.runtime.evaluate = evaluate
    server = http_module.create_evaluation_server(http_module.runtime, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert server.server_address[0] == "127.0.0.1"
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/evaluate", data=b"{}", method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                status, body = response.status, response.read()
        except urllib.error.HTTPError as response:
            status, body = response.code, response.read()
        payload = json.loads(body)
        if outcome == "success":
            assert status == 200
            assert payload == {"status": "passed", "reward": 1.0}
        elif outcome == "not_terminated":
            assert status == 409
            assert payload == {
                "status": "not_terminated",
                "reward": 0.0,
                "reward_basis": [],
            }
        else:
            assert status == 500
            assert payload == {"status": "evaluation_error", "error": "RuntimeError"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
