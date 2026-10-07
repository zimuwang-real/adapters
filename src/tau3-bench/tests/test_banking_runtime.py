"""Real offline BM25 routing with a scripted user, not model/parity evidence."""

import importlib
import json
import os
from pathlib import Path

from test_runtime_official_evaluation import official_runtime as official_runtime


def test_banking_runtime_routes_search_to_real_bm25(
    official_runtime, monkeypatch, tmp_path
):
    official = official_runtime
    data_dir = Path(os.environ["TAU2_DATA_DIR"])
    task_path = data_dir / "tau2/domains/banking_knowledge/tasks/task_012.json"
    canonical_task = json.loads(task_path.read_text())
    config_path = tmp_path / "task_config.json"
    config_path.write_text(
        json.dumps({"domain": "banking_knowledge", "task": canonical_task})
    )
    monkeypatch.setattr(
        official.runtime, "UNTRUSTED_TRACE_PATH", tmp_path / "runtime_trace.json"
    )
    monkeypatch.setattr(official.runtime, "TAU2_RUNTIME_ROOT", data_dir.parent)
    user_module = importlib.import_module("tau2.user.user_simulator")
    opening = (
        "Hi, I'm traveling to Japan next month and I want to make sure my "
        "Platinum Rewards Card doesn't get blocked. "
        "How do I set up a travel notification?"
    )

    def scripted_opening(**_kwargs):
        return official.message.AssistantMessage(role="assistant", content=opening)

    # The shared fixture removes credentials and rejects all network/model IO.
    # Replace only the simulator's generation; domain construction and retrieval
    # remain the pinned upstream implementation with its original documents.
    monkeypatch.setattr(user_module, "generate", scripted_opening)
    runtime = official.runtime.Tau3Runtime(config_path)
    original_task = runtime.task.model_dump(mode="json")
    assert runtime.env_kwargs["retrieval_variant"] == "bm25"
    assert runtime.start_conversation() == opening
    before = runtime.step_count

    result = json.loads(
        runtime.submit_assistant_tool_calls(
            json.dumps(
                [
                    {
                        "id": "ordinary-kb-search",
                        "name": "KB_search",
                        "arguments": {
                            "query": (
                                "Platinum Rewards Card annual fee "
                                "foreign transaction fee"
                            )
                        },
                    }
                ]
            )
        )
    )

    assert result["termination_reason"] is None
    assert result["num_errors"] == 0
    assert result["step_count"] == before + 2
    assert len(result["tool_results"]) == 1
    tool_result = result["tool_results"][0]
    assert tool_result["id"] == "ordinary-kb-search"
    assert tool_result["error"] is False
    assert "doc_credit_cards_platinum_rewards_card_007" in tool_result["content"]
    assert "0% foreign transaction fees" in tool_result["content"]
    assert "Earn a $150.00 rebate on your annual fee" in tool_result["content"]
    assert result["observation"] == tool_result["content"]
    assert runtime.messages[-1].content == tool_result["content"]
    assert runtime.task.model_dump(mode="json") == original_task
