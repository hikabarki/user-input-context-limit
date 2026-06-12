"""Tests for storage + flow engine: checkpointing, kill-and-resume."""

from __future__ import annotations

import pytest

from token_flow_extender.core.flow_engine import FlowEngine
from token_flow_extender.core.window_planner import WindowPlanner
from token_flow_extender.exceptions import LLMCallError
from token_flow_extender.storage.checkpoint_store import (
    MemoryCheckpointStore,
    SQLiteCheckpointStore,
)
from token_flow_extender.storage.job_queue import JobQueue, deterministic_job_id
from token_flow_extender.storage.state_backup import export_job, import_job

from conftest import CharTokenCounter, MockLLM, make_budget


def _make_engine(llm, config, store, max_input=400):
    budget = make_budget(max_input, config)
    return FlowEngine(llm, CharTokenCounter(), budget, config, store), budget


def _text(paragraphs=30):
    return "\n\n".join(f"Para {i}. " + "content " * 30 + f"NEEDLE-{i}."
                       for i in range(paragraphs))


async def test_full_run_completes_and_covers(config, store):
    llm = MockLLM()
    engine, budget = _make_engine(llm, config, store)
    text = _text()
    plan = WindowPlanner(CharTokenCounter(), budget).plan(text)
    result = await engine.run(text, plan, task="find needles", model="mock")
    assert result.coverage is not None and result.coverage.complete
    assert result.coverage.as_dict()["tokens_lost"] == 0


async def test_crash_then_resume_skips_completed(config, store):
    text = _text()
    budget = make_budget(400, config)
    plan = WindowPlanner(CharTokenCounter(), budget).plan(text)
    assert plan.total_windows >= 4

    # First run: kill on window 3 (every window costs exactly 1 call here,
    # and retries are also injected to fail).
    killer = MockLLM(fail_on_calls=[3, 4, 5])  # call 3 + its 2 retries
    engine, _ = _make_engine(killer, config, store)
    with pytest.raises(LLMCallError):
        await engine.run(text, plan, task="find needles", model="mock")

    job_id = deterministic_job_id(text, "mock", "find needles", config)
    completed_before = set(store.get_window_results(job_id))
    assert completed_before == {0, 1}  # windows 0 and 1 checkpointed

    # Second run (fresh engine = restarted process): resumes, completes.
    resumer = MockLLM()
    engine2, _ = _make_engine(resumer, config, store)
    result = await engine2.run(text, plan, task="find needles", model="mock")
    assert result.coverage is not None and result.coverage.complete

    # Windows 0 and 1 must NOT be reprocessed: the resumed run's prompts
    # must not contain window 0's first paragraph as a primary body.
    assert all("INPUT PART 1/" not in p for p in resumer.prompts)


async def test_completed_job_returns_stored_result(config, store):
    text = _text(10)
    budget = make_budget(10000, config)
    plan = WindowPlanner(CharTokenCounter(), budget).plan(text)
    llm = MockLLM()
    engine, _ = _make_engine(llm, config, store, max_input=10000)
    first = await engine.run(text, plan, task="t", model="mock")
    calls_after_first = llm.calls
    second = await engine.run(text, plan, task="t", model="mock")
    assert second.output == first.output
    assert llm.calls == calls_after_first  # zero new LLM calls


def test_sqlite_store_roundtrip(tmp_path):
    store = SQLiteCheckpointStore(str(tmp_path / "ckpt.db"))
    from token_flow_extender.storage.checkpoint_store import JobRecord
    store.create_job(JobRecord(job_id="j1", status="pending", model="m", task="t",
                               strategy="sequential_refine", input_sha256="x",
                               total_windows=3, config_json="{}"))
    store.save_window_result("j1", 0, "r0", 10, 0, 40)
    store.save_window_result("j1", 1, "r1", 10, 40, 80)
    assert store.get_window_results("j1") == {0: "r0", 1: "r1"}
    assert store.completed_indices("j1") == [0, 1]
    store.save_final("j1", "final")
    job = store.get_job("j1")
    assert job is not None and job.status == "completed" and job.result == "final"


def test_job_queue_resume_point(store):
    from token_flow_extender.storage.checkpoint_store import JobRecord
    store.create_job(JobRecord(job_id="j2", status="pending", model="m", task="t",
                               strategy="s", input_sha256="x", total_windows=5,
                               config_json="{}"))
    queue = JobQueue(store)
    store.save_window_result("j2", 0, "r", 1, 0, 1)
    store.save_window_result("j2", 1, "r", 1, 1, 2)
    store.save_window_result("j2", 3, "r", 1, 3, 4)  # gap at 2
    assert queue.resume_point("j2") == 2


def test_deterministic_job_id_stability(config):
    a = deterministic_job_id("input", "model", "task", config)
    b = deterministic_job_id("input", "model", "task", config)
    c = deterministic_job_id("input2", "model", "task", config)
    assert a == b
    assert a != c


def test_state_backup_roundtrip(tmp_path, config):
    src = MemoryCheckpointStore()
    from token_flow_extender.storage.checkpoint_store import JobRecord
    src.create_job(JobRecord(job_id="j3", status="running", model="m", task="t",
                             strategy="s", input_sha256="x", total_windows=2,
                             config_json="{}"))
    src.save_window_result("j3", 0, "partial", 5, 0, 20)
    path = str(tmp_path / "backup.json")
    export_job(src, "j3", path)

    dst = MemoryCheckpointStore()
    restored = import_job(dst, path)
    assert restored == "j3"
    assert dst.get_window_results("j3") == {0: "partial"}
