"""Offline regressions for pinned input and task-selection validation."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from tau3_bench.adapter import Tau3BenchAdapter, TauTask

MAIN_IMAGE = "sha256:" + "1" * 64
RUNTIME_IMAGE = "sha256:" + "2" * 64


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(root),
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
        },
    ).stdout.strip()


def commit(root: Path) -> str:
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "fixture",
    )
    return git(root, "rev-parse", "HEAD")


@pytest.fixture()
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    git(root, "init")
    files = {
        "data/tau2/domains/airline/tasks.json": json.dumps([{"id": "0"}, {"id": "1"}]),
        "data/tau2/domains/airline/split_tasks.json": json.dumps({"base": ["0", "1"]}),
        "data/tau2/domains/airline/policy.md": "Original policy",
        "src/tau2/config.py": "SETTING = 1\n",
        "pyproject.toml": "[project]\nname = 'fixture'\n",
        "uv.lock": "version = 1\n",
        "README.md": "Fixture documentation\n",
        ".gitignore": "__pycache__/\n*.pyc\n.cache/\n",
    }
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    monkeypatch.setattr(Tau3BenchAdapter, "_EXPECTED_TAU2_COMMIT", commit(root))
    return root


def make_adapter(source: Path, output: Path, **kwargs: object) -> Tau3BenchAdapter:
    return Tau3BenchAdapter(
        output_dir=output,
        tau2_root=source,
        domains=["airline"],
        main_image=MAIN_IMAGE,
        runtime_image=RUNTIME_IMAGE,
        **kwargs,
    )


def test_fallback_fetches_pinned_commit_instead_of_remote_head(
    source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pinned = Tau3BenchAdapter._EXPECTED_TAU2_COMMIT
    (source / "data/tau2/domains/airline/policy.md").write_text("New upstream policy")
    assert commit(source) != pinned
    monkeypatch.setattr(Tau3BenchAdapter, "_TAU2_BENCH_REPO_URL", str(source))

    checkout = tmp_path / "checkout"
    adapter = make_adapter(checkout, tmp_path / "output")

    assert adapter._resolve_tau2_commit() == pinned
    assert (
        checkout / "data/tau2/domains/airline/policy.md"
    ).read_text() == "Original policy"
    assert git(checkout, "status", "--porcelain") == ""


def test_failed_fallback_fetch_can_be_retried(
    source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pinned = Tau3BenchAdapter._EXPECTED_TAU2_COMMIT
    monkeypatch.setattr(Tau3BenchAdapter, "_TAU2_BENCH_REPO_URL", str(source))
    monkeypatch.setattr(Tau3BenchAdapter, "_EXPECTED_TAU2_COMMIT", "f" * 40)
    checkout = tmp_path / "checkout"

    with pytest.raises(RuntimeError, match="failed to fetch pinned"):
        make_adapter(checkout, tmp_path / "output")

    assert not checkout.exists()
    monkeypatch.setattr(Tau3BenchAdapter, "_EXPECTED_TAU2_COMMIT", pinned)
    assert make_adapter(checkout, tmp_path / "output")._resolve_tau2_commit() == pinned


def test_fallback_does_not_remove_preexisting_path(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    sentinel = checkout / "notes.txt"
    sentinel.write_text("keep this")

    with pytest.raises(FileNotFoundError, match="not a valid tau2-bench"):
        make_adapter(checkout, tmp_path / "output")

    assert sentinel.read_text() == "keep this"


@pytest.mark.parametrize(
    "relative",
    [
        "data/tau2/domains/airline/tasks.json",
        "data/tau2/domains/airline/split_tasks.json",
        "data/tau2/domains/airline/policy.md",
        "src/tau2/config.py",
        "pyproject.toml",
        "uv.lock",
    ],
)
@pytest.mark.parametrize("staged", [False, True])
def test_rejects_modified_benchmark_inputs(
    source: Path,
    tmp_path: Path,
    relative: str,
    staged: bool,
) -> None:
    (source / relative).write_text("modified")
    if staged:
        git(source, "add", relative)

    with pytest.raises(ValueError, match="uncommitted benchmark"):
        make_adapter(source, tmp_path / "output")._resolve_tau2_commit()


@pytest.mark.parametrize(
    "relative",
    [
        "data/tau2/domains/banking_knowledge/tasks/task_extra.json",
        "data/tau2/domains/banking_knowledge/prompts/components/extra.md",
        "src/tau2/extra.py",
    ],
)
def test_rejects_untracked_benchmark_inputs(
    source: Path,
    tmp_path: Path,
    relative: str,
) -> None:
    path = source / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("extra input")

    with pytest.raises(ValueError, match="uncommitted benchmark"):
        make_adapter(source, tmp_path / "output")._resolve_tau2_commit()


def test_rejects_deleted_benchmark_input(source: Path, tmp_path: Path) -> None:
    (source / "data/tau2/domains/airline/policy.md").unlink()

    with pytest.raises(ValueError, match="uncommitted benchmark"):
        make_adapter(source, tmp_path / "output")._resolve_tau2_commit()


def test_allows_unrelated_results_credentials_and_python_cache(
    source: Path,
    tmp_path: Path,
) -> None:
    for relative in [
        "README.md",
        ".env",
        "results/result.json",
        "logs/run.log",
        "data/simulations/trial.json",
        "src/tau2/__pycache__/config.cpython-312.pyc",
    ]:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("local artifact")

    assert make_adapter(source, tmp_path / "output")._resolve_tau2_commit() == (
        Tau3BenchAdapter._EXPECTED_TAU2_COMMIT
    )


@pytest.mark.parametrize(
    "task_ids",
    [
        ["unknown"],
        ["airline:0", "unknown"],
        [" "],
    ],
)
def test_invalid_task_selection_preserves_existing_dataset(
    source: Path,
    tmp_path: Path,
    task_ids: list[str],
) -> None:
    output = tmp_path / "dataset"
    make_adapter(source, output).run()
    before = {
        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }

    with pytest.raises(ValueError, match="task IDs"):
        make_adapter(source, output, overwrite=True, task_ids=task_ids).run()

    after = {
        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }
    assert after == before


def test_task_aliases_match_without_duplicates_and_preserve_source_order(
    source: Path,
    tmp_path: Path,
) -> None:
    adapter = make_adapter(
        source,
        tmp_path / "output",
        task_ids=[
            " 0 ",
            "airline:0",
            "airline/0",
            "tau3-airline-0",
            "0",
        ],
    )
    tasks = [
        TauTask(domain, "0", f"tau3-{domain}-0", {}, "policy")
        for domain in ["airline", "retail", "telecom"]
    ]

    assert adapter._filter_tasks(tasks) == tasks


def test_domain_qualified_alias_selects_only_that_domain(
    source: Path, tmp_path: Path
) -> None:
    adapter = make_adapter(source, tmp_path / "output", task_ids=["airline:0"])
    tasks = [
        TauTask(domain, "0", f"tau3-{domain}-0", {}, "policy")
        for domain in ["airline", "retail", "telecom"]
    ]

    assert adapter._filter_tasks(tasks) == tasks[:1]


@pytest.mark.parametrize("unowned", ["tau3-notes", "other-files"])
def test_overwrite_rejects_unowned_directories_before_deleting_tasks(
    source: Path,
    tmp_path: Path,
    unowned: str,
) -> None:
    output = tmp_path / "dataset"
    make_adapter(source, output).run()
    sentinel = output / unowned / "notes.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("unrelated files")
    before = {
        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }

    with pytest.raises(ValueError, match="unrecognized output"):
        make_adapter(source, output, overwrite=True, task_ids=["airline:0"]).run()

    after = {
        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    "invalid_task", ["foreign_name", "missing_asset", "invalid_toml"]
)
def test_overwrite_requires_generated_task_identity_and_assets(
    source: Path,
    tmp_path: Path,
    invalid_task: str,
) -> None:
    output = tmp_path / "dataset"
    make_adapter(source, output).run()
    task = output / "tau3-airline-1"
    if invalid_task == "missing_asset":
        (task / "tests/evaluate.py").unlink()
    elif invalid_task == "invalid_toml":
        (task / "task.toml").write_text("not TOML")
    else:
        (task / "task.toml").write_text('[task]\nname="different-benchmark/task"\n')
    before = {
        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }

    with pytest.raises(ValueError, match="unrecognized output"):
        make_adapter(source, output, overwrite=True).run()

    after = {
        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }
    assert after == before


def test_overwrite_accepts_legacy_generated_tasks_without_manifest(
    source: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "dataset"
    make_adapter(source, output).run()
    (output / "tau3-adapter-manifest.json").unlink()
    legacy_solution = output / "tau3-airline-1/solution/solve.sh"
    legacy_solution.parent.mkdir()
    legacy_solution.write_text("legacy oracle")

    make_adapter(source, output, overwrite=True, task_ids=["airline:0"]).run()

    assert not (output / "tau3-airline-1").exists()
    assert (output / "tau3-airline-0/tests/test.sh").is_file()
