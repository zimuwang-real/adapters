"""Offline regressions for ordinary MCP client error handling."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.client import streamable_http

RUNNER = Path(__file__).parents[1] / "run_tau3_llm_agent.py"


@pytest.fixture
def runner(monkeypatch):
    # No transport is used here; newer host MCP versions removed the legacy
    # import that is still provided by the adapter's pinned Docker image.
    monkeypatch.setattr(streamable_http, "streamablehttp_client", None, raising=False)
    spec = importlib.util.spec_from_file_location("tau3_llm_runner_under_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class LocalSession:
    """Supply MCP results without starting a server or making model requests."""

    def __init__(self, responses):
        self.responses = responses

    async def call_tool(self, name, arguments, *, read_timeout_seconds):
        return self.responses[name]


def text_result(text, *, is_error=False):
    # Match the pinned runtime client's MCP result fields without depending on
    # the host MCP version's Python aliases for the wire-format field names.
    return SimpleNamespace(isError=is_error, content=[{"type": "text", "text": text}])


def test_bootstrap_mcp_error_stops_before_model_completion(runner):
    session = LocalSession(
        {
            "configure_run": text_result("{}"),
            "start_conversation": text_result("User simulator failed", is_error=True),
            "get_runtime_status": text_result("{}"),
        }
    )

    def unexpected_completion(**kwargs):
        pytest.fail("Model completion must not run after a bootstrap MCP error")

    with pytest.raises(RuntimeError, match="start_conversation.*User simulator failed"):
        asyncio.run(
            runner.run_agent_loop(
                session=session,
                model="offline-model",
                system_prompt="Local test policy",
                tool_schemas=[],
                max_steps=1,
                llm_args={},
                completion_fn=unexpected_completion,
                tool_timeout_sec=1,
            )
        )


@pytest.mark.parametrize("helper", ["_call_text_tool", "_call_json_tool"])
def test_mcp_error_reports_tool_name_and_error_text(runner, helper):
    session = LocalSession(
        {"submit_assistant_message": text_result("Conversation failed", is_error=True)}
    )
    with pytest.raises(
        RuntimeError, match="submit_assistant_message.*Conversation failed"
    ):
        asyncio.run(
            getattr(runner, helper)(session, "submit_assistant_message", timeout_sec=1)
        )


def test_successful_text_observation_is_preserved(runner):
    session = LocalSession({"start_conversation": text_result("Please help me.")})
    assert (
        asyncio.run(
            runner._call_text_tool(session, "start_conversation", timeout_sec=1)
        )
        == "Please help me."
    )


@pytest.mark.parametrize("structured", [False, True])
def test_successful_json_result_preserves_domain_tool_error(runner, structured):
    # A normal domain error is task feedback, not an MCP execution failure.
    payload = {"tool_results": [{"error": True, "content": "Customer not found"}]}
    result = (
        SimpleNamespace(isError=False, content=[], structuredContent=payload)
        if structured
        else text_result(json.dumps(payload))
    )
    session = LocalSession({"submit_assistant_tool_calls": result})
    assert (
        asyncio.run(
            runner._call_json_tool(
                session, "submit_assistant_tool_calls", timeout_sec=1
            )
        )
        == payload
    )
