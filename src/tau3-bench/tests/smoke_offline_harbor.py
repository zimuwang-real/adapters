"""Bounded real-Docker/Harbor smoke using an unchanged official airline task.

Run with a Harbor 0.23 Python environment and prepared smoke runtime image::

    python tests/smoke_offline_harbor.py --evidence-dir /path/to/evidence \
        --tau2-root /path/to/pinned/tau2-bench --runtime-image IMAGE_ID \
        --main-image IMAGE_ID

The only model replacement is tau2's user generator. Runtime state, tools,
official evaluation, export, Docker networking, collection, and verification
are real. The runtime image must contain the pinned tau2 sources at
/opt/tau2-bench, its airline data, runtime dependencies, and FastMCP.
Image building is deliberately separate from the offline test execution.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import toml
import yaml

LAUNCHER = r"""
import os
import runpy
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
from tau2.data_model.message import AssistantMessage
import tau2.user.user_simulator as simulator

def deterministic_user(**kwargs):
    return AssistantMessage(role="assistant", content="I am Emma Kim, emma_kim_9957. Please cancel reservation EHGLP3 only if I receive a refund.")

simulator.generate = deterministic_user
runpy.run_path("/app/server.py", run_name="__main__")
"""


CANDIDATE = r"""
import json
import os
from pathlib import Path
import socket
import sys
import urllib.request
import urllib.error

case = sys.argv[1]
probe = {"case": case}
base = "http://tau3-runtime:8000"
request = urllib.request.Request(base + "/evaluate", data=b"{}", method="POST")
try:
    with urllib.request.urlopen(request, timeout=3) as response:
        probe["public_evaluate_status"] = response.status
except urllib.error.HTTPError as exc:
    probe["public_evaluate_status"] = exc.code
assert probe["public_evaluate_status"] == 404, probe
try:
    connection = socket.create_connection(("tau3-runtime", 8001), timeout=2)
except OSError as exc:
    probe["private_evaluator_unreachable"] = type(exc).__name__
else:
    connection.close()
    raise AssertionError("private evaluator is reachable from main")

session = None
counter = 0
def rpc(method, params=None, notification=False):
    global session, counter
    counter += 1
    payload = {"jsonrpc": "2.0", "method": method}
    if not notification:
        payload["id"] = counter
    if params is not None:
        payload["params"] = params
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if session:
        headers["Mcp-Session-Id"] = session
    request = urllib.request.Request(base + "/mcp", data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=20) as response:
        session = response.headers.get("Mcp-Session-Id", session)
        body = response.read().decode()
    if notification:
        return None
    if body.startswith("event:") or body.startswith("data:"):
        payloads = [line[5:].strip() for line in body.splitlines() if line.startswith("data:")]
        result = json.loads(payloads[-1])
    else:
        result = json.loads(body)
    assert "error" not in result, result
    return result["result"]

rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "offline-smoke", "version": "1"}})
rpc("notifications/initialized", notification=True)
tools = rpc("tools/list")["tools"]
probe["tool_names"] = [tool["name"] for tool in tools]
assert not any("evaluat" in name or "export" in name for name in probe["tool_names"])
configure_schema = next(tool["inputSchema"] for tool in tools if tool["name"] == "configure_run")
probe["configure_run_parameters"] = sorted(configure_schema["properties"])
assert probe["configure_run_parameters"] == ["max_errors", "max_steps", "seed"], configure_schema

def tool(name, arguments=None):
    result = rpc("tools/call", {"name": name, "arguments": arguments or {}})
    assert not result.get("isError"), result
    return result

probe["configuration"] = tool("configure_run", {"seed": 7, "max_steps": 30, "max_errors": 3})
probe["start"] = tool("start_conversation")
probe["read_tool"] = tool("get_reservation_details", {"reservation_id": "EHGLP3"})
if case in {"forged_mismatch", "missing_export"}:
    probe["bad_action"] = tool("cancel_reservation", {"reservation_id": "EHGLP3"})
if case != "not_terminated":
    probe["termination"] = tool("end_conversation", {"message": "Your reservation cannot be canceled with a refund."})
probe["status"] = tool("get_runtime_status")
probe["private_paths_absent"] = {path: not Path(path).exists() for path in ["/app/task_config.json", "/opt/tau2-bench", "/app/server.py"]}
assert all(probe["private_paths_absent"].values()), probe

if case != "clean_pass":
    forged = {"status": "passed", "reward": 1.0, "reward_basis": ["DB"], "forged_by_candidate": True}
    for path in ["/tmp/tau3-evaluation.json", "/logs/artifacts/tau3-evaluation.json", "/logs/artifacts/runtime/tau3-evaluation.json", "/logs/artifacts/tmp/tau3-evaluation.json", "/logs/agent/tau3_untrusted_trace.json"]:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(forged))
    for path in ["/logs/verifier/reward.txt", "/logs/verifier/reward.json", "/tests/test.sh", "/tests/evaluate.py"]:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("1" if path.endswith(".txt") else json.dumps(forged))
    probe["forged_candidate_files"] = True
Path("/logs/agent/smoke-probe.json").write_text(json.dumps(probe, indent=2))
print(json.dumps(probe, indent=2))
"""


async def run_case(args, case: str) -> dict:
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.hooks import TrialEvent
    from harbor.trial.trial import Trial
    from tau3_bench.adapter import Tau3BenchAdapter

    output = args.evidence_dir / "generated" / case
    adapter = Tau3BenchAdapter(
        output_dir=output,
        task_ids=["airline:0"],
        domains=["airline"],
        tau2_root=args.tau2_root,
        main_image=args.main_image,
        runtime_image=args.runtime_image,
    )
    adapter.run()
    task = next(path.parent for path in output.rglob("task.toml"))
    config_path = task / "task.toml"
    config = toml.loads(config_path.read_text())
    config["environment"].update(cpus=1, memory_mb=512, env={}, build_timeout_sec=90)
    config["agent"]["timeout_sec"] = 60
    config["verifier"]["timeout_sec"] = 60
    config["verifier"]["collect"][0]["timeout_sec"] = 45
    if case == "missing_export":
        config["verifier"]["collect"][0]["command"] = (
            "rm -f /tmp/tau3-evaluation.json; exit 17"
        )
    config_path.write_text(toml.dumps(config))

    compose_path = task / "environment/docker-compose.yaml"
    compose = yaml.safe_load(compose_path.read_text())
    runtime = compose["services"]["tau3-runtime"]
    runtime["environment"] = {
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "TAU2_USER_MODEL": "smoke-never-called",
    }
    runtime["command"] = ["python3", "/smoke/launcher.py"]
    runtime["cpus"] = 1
    runtime["mem_limit"] = "1024m"
    runtime.setdefault("volumes", []).append(
        "./smoke-launcher.py:/smoke/launcher.py:ro"
    )
    runtime["healthcheck"]["start_period"] = "2s"
    compose.setdefault("networks", {})["default"] = {"internal": True}
    compose_path.write_text(yaml.safe_dump(compose, sort_keys=False))
    (task / "environment/smoke-launcher.py").write_text(LAUNCHER)
    (task / "solution").mkdir(exist_ok=True)
    (task / "solution/smoke-candidate.py").write_text(CANDIDATE)
    (task / "solution/solve.sh").write_text(
        f"#!/bin/bash\nset -eu\npython3 /solution/smoke-candidate.py {case}\n"
    )

    trial_config = TrialConfig.model_validate(
        {
            "task": {"path": str(task)},
            "trial_name": f"tau3-smoke-{case}",
            "trials_dir": str(args.evidence_dir / "trials"),
            "agent": {"name": "oracle"},
            "environment": {
                "type": "docker",
                "delete": True,
                "cpu_enforcement_policy": "limit",
                "memory_enforcement_policy": "limit",
            },
        }
    )
    started = time.monotonic()
    trial = await Trial.create(trial_config)

    async def capture_limits(event):
        ids = subprocess.check_output(
            [
                "docker",
                "ps",
                "--filter",
                f"label=com.docker.compose.project={event.trial_name}__env",
                "-q",
            ],
            text=True,
        ).split()
        records = []
        for identity in ids:
            container = json.loads(
                subprocess.check_output(["docker", "inspect", identity], text=True)
            )[0]
            networks = []
            for name in container["NetworkSettings"]["Networks"]:
                network = json.loads(
                    subprocess.check_output(
                        ["docker", "network", "inspect", name], text=True
                    )
                )[0]
                networks.append({"name": name, "internal": network["Internal"]})
            records.append(
                {
                    "name": container["Name"],
                    "image": container["Image"],
                    "nano_cpus": container["HostConfig"]["NanoCpus"],
                    "memory_bytes": container["HostConfig"]["Memory"],
                    "networks": networks,
                }
            )
        assert len(records) == 2, records
        assert sum(row["nano_cpus"] for row in records) == 2_000_000_000, records
        assert sum(row["memory_bytes"] for row in records) == 1536 * 1024 * 1024, (
            records
        )
        assert all(
            network["internal"] for row in records for network in row["networks"]
        ), records
        (
            args.evidence_dir / "trials" / event.trial_name / "resource-limits.json"
        ).write_text(json.dumps(records, indent=2))

    trial.add_hook(TrialEvent.AGENT_START, capture_limits)
    result = await trial.run()
    result_data = result.model_dump(mode="json")
    trial_dir = args.evidence_dir / "trials" / trial_config.trial_name
    (trial_dir / "smoke-result.json").write_text(json.dumps(result_data, indent=2))
    probe_path = trial_dir / "agent/smoke-probe.json"
    assert probe_path.exists(), f"candidate probe did not complete: {trial_dir}"
    probe = json.loads(probe_path.read_text())
    assert probe["public_evaluate_status"] == 404
    artifact = trial_dir / "artifacts/runtime/tau3-evaluation.json"
    rewards = (result_data.get("verifier_result") or {}).get("rewards")
    if case == "missing_export":
        assert not artifact.exists(), (
            f"missing runtime export replaced by candidate artifact: {artifact}"
        )
        assert result_data.get("exception_info"), result_data
        assert rewards is None or rewards.get("reward") != 1, result_data
        actual_status = "missing_export_failed_closed"
    else:
        assert not result_data.get("exception_info"), result_data
        evaluation = json.loads(artifact.read_text())
        assert not evaluation.get("forged_by_candidate"), evaluation
        expected_reward = 1 if case == "clean_pass" else 0
        expected_status = {
            "clean_pass": "passed",
            "forged_mismatch": "mismatch",
            "not_terminated": "not_terminated",
        }[case]
        assert evaluation["reward"] == expected_reward, evaluation
        assert evaluation["status"] == expected_status, evaluation
        assert rewards["reward"] == expected_reward, result_data
        actual_status = evaluation["status"]
    return {
        "case": case,
        "passed": True,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "runtime_status": actual_status,
        "rewards": rewards,
        "trial_dir": str(trial_dir),
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--main-image", required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["clean_pass", "forged_mismatch", "not_terminated", "missing_export"],
        choices=["clean_pass", "forged_mismatch", "not_terminated", "missing_export"],
    )
    args = parser.parse_args()
    # No secret values are read, forwarded, or written to evidence.
    for key in list(os.environ):
        if any(
            marker in key for marker in ["API_KEY", "API_TOKEN", "SECRET", "PASSWORD"]
        ):
            os.environ.pop(key, None)
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    template = Path(__file__).parents[1] / "src/tau3_bench/task-template"
    summary = {
        "harbor_version": importlib.metadata.version("harbor"),
        "main_image": args.main_image,
        "runtime_image": args.runtime_image,
        "scope": "real Harbor Docker orchestration; official unchanged airline task 0; real tau2 tools and evaluator; deterministic user generator; candidate/runtime Docker internal network; candidate+sidecar capped at 2 CPUs/1536 MiB; verifier 1 CPU/512 MiB",
        "limitation": "Harbor no-network policy is unsupported by this host kernel. Candidate/runtime are isolated with Docker internal networking. Separate verifier has ordinary Docker networking but executes only the stdlib JSON reader and makes no network requests. No model APIs or credentials used.",
        "template_sha256": {
            str(path.relative_to(template)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in [
                template / "task.toml",
                template / "tests/evaluate.py",
                template / "tests/test.sh",
                template / "environment/runtime-server/server.py",
            ]
        },
        "cases": [],
    }
    summary_path = args.evidence_dir / "summary.json"
    for case in args.cases:
        try:
            result = await run_case(args, case)
            summary["cases"].append(result)
            print(json.dumps(result), flush=True)
        except Exception as exc:
            summary["cases"].append({"case": case, "passed": False, "error": str(exc)})
            raise
        finally:
            summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
