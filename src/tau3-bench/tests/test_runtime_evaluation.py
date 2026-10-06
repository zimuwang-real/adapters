"""Offline regressions for the runtime's authoritative evaluator boundary."""

import ast
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

TEMPLATE_DIR = Path(__file__).parents[1] / "src/tau3_bench/task-template"
SERVER = TEMPLATE_DIR / "environment/runtime-server/server.py"


@pytest.fixture()
def runtime_module(monkeypatch):
    """Load runtime definitions without starting a task or any model clients."""
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
    fastmcp = ModuleType("fastmcp")
    fastmcp.FastMCP = lambda _name: None
    monkeypatch.setitem(sys.modules, "fastmcp", fastmcp)
    module = ModuleType("tau3_runtime_under_test")
    exec(compile(tree, str(SERVER), "exec"), module.__dict__)
    return module


class TerminationReason(str, Enum):
    AGENT_STOP = "agent_stop"
    USER_STOP = "user_stop"


@pytest.mark.parametrize("configured_model", [None, "test-provider/judge-model"])
def test_official_evaluation_uses_configured_nl_assertions_model(
    runtime_module, monkeypatch, configured_model
):
    # tau2 captures this constant at import time, which can precede evaluation.
    tau2_config = ModuleType("tau2.config")
    tau2_config.DEFAULT_LLM_NL_ASSERTIONS = "upstream-default"
    nl_evaluator = ModuleType("tau2.evaluator.evaluator_nl_assertions")
    nl_evaluator.DEFAULT_LLM_NL_ASSERTIONS = "upstream-default"
    monkeypatch.setitem(sys.modules, "tau2.config", tau2_config)
    monkeypatch.setitem(sys.modules, nl_evaluator.__name__, nl_evaluator)
    if configured_model is None:
        monkeypatch.delenv("TAU2_NL_ASSERTIONS_MODEL", raising=False)
    else:
        monkeypatch.setenv("TAU2_NL_ASSERTIONS_MODEL", configured_model)

    runtime = runtime_module.Tau3Runtime.__new__(runtime_module.Tau3Runtime)
    runtime.task = SimpleNamespace(id="offline-task")
    runtime.messages = [{"role": "assistant", "content": "finished"}]
    runtime.termination_reason = "agent_stop"
    runtime.TerminationReason = TerminationReason
    runtime.seed = 7
    runtime.domain = "airline"
    runtime.env_kwargs = {}
    evaluation_calls = []

    def evaluate_simulation(**kwargs):
        evaluation_calls.append(kwargs)
        # Observe the value consumed by the pinned tau2 NL evaluator, not just
        # the config module from which it originally imported that value.
        return {"judge_model": nl_evaluator.DEFAULT_LLM_NL_ASSERTIONS}

    class CommunicationMode(str, Enum):
        HALF_DUPLEX = "half_duplex"

    attributes = {
        ("tau2.data_model.simulation", "SimulationRun"): SimpleNamespace,
        ("tau2.evaluator.evaluator", "evaluate_simulation"): evaluate_simulation,
        ("tau2.evaluator.evaluator", "EvaluationType"): SimpleNamespace(ALL="all"),
        ("tau2.orchestrator.modes", "CommunicationMode"): CommunicationMode,
        ("tau2.utils.utils", "get_now"): lambda: "offline-now",
    }
    monkeypatch.setattr(
        runtime_module, "_require_attr", lambda mod, attr: attributes[mod, attr]
    )

    result = runtime._evaluate_official()

    expected_model = configured_model or "gpt-5.2"
    assert result == {"judge_model": expected_model}
    assert tau2_config.DEFAULT_LLM_NL_ASSERTIONS == expected_model
    assert evaluation_calls[0]["evaluation_type"] == "all"
    assert evaluation_calls[0]["strict_replay"] is True
    assert evaluation_calls[0]["simulation"].messages == runtime.messages
    assert evaluation_calls[0]["simulation"].messages is not runtime.messages


def test_compose_forwards_nl_assertions_model_to_runtime():
    compose = (TEMPLATE_DIR / "environment/docker-compose.yaml").read_text()
    runtime_service = compose.split("\n  tau3-runtime:", 1)[1]
    assert (
        "TAU2_NL_ASSERTIONS_MODEL=${TAU2_NL_ASSERTIONS_MODEL:-gpt-5.2}"
        in runtime_service
    )
