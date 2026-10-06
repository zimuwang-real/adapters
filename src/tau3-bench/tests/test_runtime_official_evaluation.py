"""Offline integration with the pinned, unmodified tau2 evaluator.

Set TAU2_BENCH_TEST_ROOT to the pinned tau2 checkout when it is not installed
editable. Only the NL judge's model call is replaced; scoring stays upstream.
"""

import ast
import importlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SERVER = (
    Path(__file__).parents[1]
    / "src/tau3_bench/task-template/environment/runtime-server/server.py"
)
TAU2_COMMIT = "1d244f5dca42944b67a379b44bfeb9f5748f189d"


@pytest.fixture()
def official_runtime(monkeypatch):
    """Import real dependencies only after removing credentials and blocking IO."""
    configured_root = os.environ.get("TAU2_BENCH_TEST_ROOT")
    if importlib.util.find_spec("fastmcp") is None:
        pytest.skip("Official evaluator integration requires fastmcp")
    if configured_root:
        root = Path(configured_root).resolve()
    else:
        spec = importlib.util.find_spec("tau2")
        if spec is None or spec.origin is None:
            pytest.skip("Official evaluator integration requires pinned tau2 sources")
        root = Path(spec.origin).resolve().parents[2]
    package = root / "src/tau2"
    if not (package / "__init__.py").is_file():
        pytest.skip(
            "No tau2 source checkout; set TAU2_BENCH_TEST_ROOT or install editable"
        )
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        clean = (
            subprocess.run(
                ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", "src/tau2"],
                check=False,
            ).returncode
            == 0
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("Official evaluator integration requires a tau2 Git checkout")
    if revision != TAU2_COMMIT or not clean:
        pytest.skip(f"Official evaluator integration requires clean tau2 {TAU2_COMMIT}")
    for name, module in tuple(sys.modules.items()):
        if name == "tau2" or name.startswith("tau2."):
            source = getattr(module, "__file__", None)
            if source is None or not Path(source).resolve().is_relative_to(package):
                pytest.skip("Previously imported tau2 is not from the pinned checkout")

    # Do not inherit provider credentials, tracing endpoints, or proxy settings.
    monkeypatch.setattr(
        os,
        "environ",
        {
            "PYTHON_DOTENV_DISABLED": "1",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "HF_HUB_OFFLINE": "1",
            "TAU2_DATA_DIR": str(root / "data"),
        },
    )
    monkeypatch.syspath_prepend(str(root / "src"))

    import dotenv
    import dotenv.main

    for module in (dotenv, dotenv.main):
        monkeypatch.setattr(module, "load_dotenv", lambda *args, **kwargs: False)
        monkeypatch.setattr(module, "dotenv_values", lambda *args, **kwargs: {})
        monkeypatch.setattr(module, "find_dotenv", lambda *args, **kwargs: "")

    unexpected_calls = []

    def forbidden_io(*args, **kwargs):
        unexpected_calls.append(True)
        pytest.fail("Unexpected network or unmocked model call in offline integration")

    monkeypatch.setattr(socket, "create_connection", forbidden_io)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden_io)
    monkeypatch.setattr(socket.socket, "connect", forbidden_io)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden_io)

    tau2 = importlib.import_module("tau2")
    assert Path(tau2.__file__).resolve() == package / "__init__.py"
    nl = importlib.import_module("tau2.evaluator.evaluator_nl_assertions")
    evaluator = importlib.import_module("tau2.evaluator.evaluator")
    llm_utils = importlib.import_module("tau2.utils.llm_utils")
    config = importlib.import_module("tau2.config")
    # Guard alternate model paths, while leaving all evaluator logic untouched.
    monkeypatch.setattr(llm_utils, "generate", forbidden_io)
    monkeypatch.setattr(llm_utils, "completion", forbidden_io)
    monkeypatch.setattr(llm_utils.litellm, "completion", forbidden_io)
    monkeypatch.setattr(llm_utils.litellm, "acompletion", forbidden_io)
    # Also restore these import-time constants after each parameterized case.
    monkeypatch.setattr(
        config, "DEFAULT_LLM_NL_ASSERTIONS", config.DEFAULT_LLM_NL_ASSERTIONS
    )
    monkeypatch.setattr(nl, "DEFAULT_LLM_NL_ASSERTIONS", nl.DEFAULT_LLM_NL_ASSERTIONS)

    tree = ast.parse(SERVER.read_text(), filename=str(SERVER))
    startup = next(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "runtime"
            for target in node.targets
        )
    )
    tree.body = tree.body[:startup]
    module = ModuleType("tau3_runtime_official_evaluation_under_test")
    # Execute trusted local definitions without starting the runtime service.
    exec(compile(tree, str(SERVER), "exec"), module.__dict__)  # noqa: S102
    yield SimpleNamespace(
        runtime=module,
        evaluator=evaluator,
        nl=nl,
        tasks=importlib.import_module("tau2.data_model.tasks"),
        simulation=importlib.import_module("tau2.data_model.simulation"),
        message=importlib.import_module("tau2.data_model.message"),
        modes=importlib.import_module("tau2.orchestrator.modes"),
    )
    assert not unexpected_calls


@pytest.mark.parametrize("configured_model", [None, "test-provider/judge-model"])
@pytest.mark.parametrize("judgment_passes", [True, False], ids=["pass", "fail"])
def test_runtime_preserves_official_nl_judgment(
    official_runtime, monkeypatch, configured_model, judgment_passes
):
    official = official_runtime
    if configured_model is not None:
        monkeypatch.setenv("TAU2_NL_ASSERTIONS_MODEL", configured_model)
    assertions = [
        "The assistant states that the parcel is at the north collection desk.",
        "The assistant explains that collection closes at 18:00.",
    ]
    checks = [
        {"nl_assertion": assertions[0], "met": True, "justification": "Desk stated."},
        {
            "nl_assertion": assertions[1],
            "met": judgment_passes,
            "justification": "Deterministic closing-time judgment.",
        },
    ]
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return official.message.AssistantMessage(
            role="assistant",
            content=json.dumps(
                {
                    "results": [
                        {
                            "expectedOutcome": check["nl_assertion"],
                            "metExpectation": check["met"],
                            "reasoning": check["justification"],
                        }
                        for check in checks
                    ]
                }
            ),
        )

    monkeypatch.setattr(official.nl, "generate", generate)
    assert official.evaluator.NLAssertionsEvaluator is official.nl.NLAssertionsEvaluator
    task = official.tasks.Task(
        id="offline-nl-task",
        user_scenario={"instructions": "Ask where and when to collect the parcel."},
        evaluation_criteria={
            "nl_assertions": assertions,
            "reward_basis": [official.tasks.RewardType.NL_ASSERTION],
        },
    )
    messages = [
        official.message.UserMessage(
            role="user", content="Where and when can I collect my parcel?"
        ),
        official.message.AssistantMessage(
            role="assistant",
            content="Your parcel is at the north collection desk. It closes at 18:00.",
        ),
    ]
    runtime = official.runtime.Tau3Runtime.__new__(official.runtime.Tau3Runtime)
    runtime.task = task
    runtime.messages = messages
    runtime.TerminationReason = official.simulation.TerminationReason
    runtime.termination_reason = official.simulation.TerminationReason.AGENT_STOP.value
    runtime.seed = 7
    runtime.domain = "mock"
    runtime.env_kwargs = {}
    runtime._evaluation_result = None
    runtime._evaluation_in_progress = False

    status_code, result = runtime.evaluate()

    # Independently construct the real upstream SimulationRun and compare the
    # runtime's complete serialized reward to the official ALL evaluation.
    simulation = official.simulation.SimulationRun(
        id="reference-simulation",
        task_id=task.id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:00:00",
        duration=0.0,
        termination_reason=official.simulation.TerminationReason.AGENT_STOP,
        messages=messages,
        seed=7,
        mode=official.modes.CommunicationMode.HALF_DUPLEX.value,
    )
    reference = official.evaluator.evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=official.evaluator.EvaluationType.ALL,
        solo_mode=False,
        domain="mock",
        mode=official.modes.CommunicationMode.HALF_DUPLEX,
        env_kwargs={},
        strict_replay=True,
    )

    assert status_code == 200
    assert result["status"] == ("passed" if judgment_passes else "mismatch")
    assert result["reward"] == float(judgment_passes)
    assert result["reward_basis"] == ["NL_ASSERTION"]
    assert result["reward_info"] == reference.model_dump(mode="json")
    assert result["reward_info"]["reward_breakdown"] == {
        "NL_ASSERTION": float(judgment_passes)
    }
    assert result["reward_info"]["nl_assertions"] == checks
    assert len(calls) == 2
    for call in calls:
        assert call["model"] == (configured_model or "gpt-5.2")
        assert call["call_name"] == "nl_assertions_eval"
        assert call["temperature"] == 0.0
        assert [message.role for message in call["messages"]] == ["system", "user"]
        prompt = call["messages"][1].content
        for message in messages:
            assert f"{message.role}: {message.content}" in prompt
        for assertion in assertions:
            assert assertion in prompt
