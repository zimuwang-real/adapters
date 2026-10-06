"""Regression tests for the verifier's isolated, runtime-collected input."""

import importlib.util
import json
import shlex
import sys
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).parents[1] / "src/tau3_bench/task-template"
EVALUATE = TEMPLATE / "tests/evaluate.py"
spec = importlib.util.spec_from_file_location("tau3_evaluate", EVALUATE)
assert spec is not None
assert spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def shell_arguments() -> list[str]:
    source = (TEMPLATE / "tests/test.sh").read_text()
    command = source[source.index("python3 /tests/evaluate.py") :]
    return shlex.split(command.replace("\\\n", ""))[2:]


def test_generated_shell_command_matches_verifier_cli(monkeypatch, tmp_path):
    """Execute the actual shell command's argument shape, not a copied CLI list."""
    evaluation = tmp_path / "runtime-result.json"
    result = {"status": "passed", "reward": 1.0, "reward_basis": []}
    evaluation.write_text(json.dumps(result))
    args = [
        value.replace("${LOG_DIR}", str(tmp_path)).replace(
            "/tmp/tau3-evaluation.json", str(evaluation)
        )
        for value in shell_arguments()
    ]
    monkeypatch.setattr(sys, "argv", [str(EVALUATE), *args])
    module.main()
    assert (tmp_path / "reward.txt").read_text() == "1.0"
    assert json.loads((tmp_path / "result.json").read_text()) == result


@pytest.mark.parametrize(
    "status,reward", [("passed", 1.0), ("mismatch", 0.0), ("not_terminated", 0.0)]
)
def test_runtime_outcomes_are_preserved(status, reward):
    result = {"status": status, "reward": reward, "reward_basis": []}
    assert module.validate_result(result) == result


@pytest.mark.parametrize(
    "result",
    [
        None,
        [],
        {"status": "evaluation_error", "reward": 0},
        {"status": "passed", "reward": True},
        {"status": "passed", "reward": "1"},
        {"status": "passed", "reward": float("nan")},
        {"status": "passed", "reward": float("inf")},
        {"status": "passed", "reward": -1},
        {"status": "passed", "reward": 2},
        {"status": "passed", "reward": 0},
        {"status": "mismatch", "reward": 1},
        {"status": "not_terminated", "reward": 1},
    ],
)
def test_invalid_or_inconsistent_result_is_rejected(result):
    with pytest.raises(ValueError):
        module.validate_result(result)


@pytest.mark.parametrize(
    "contents", [None, "bad json", '{"status":"evaluation_error","reward":0}']
)
def test_missing_or_corrupt_artifact_fails_without_reward(
    monkeypatch, tmp_path, contents
):
    evaluation = tmp_path / "runtime-result.json"
    if contents is not None:
        evaluation.write_text(contents)
    reward = tmp_path / "reward.txt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(EVALUATE),
            "--evaluation",
            str(evaluation),
            "--reward",
            str(reward),
            "--result",
            str(tmp_path / "result.json"),
        ],
    )
    with pytest.raises((OSError, ValueError)):
        module.main()
    assert not reward.exists()
