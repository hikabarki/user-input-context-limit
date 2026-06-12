"""Tests for core/coverage_ledger.py — the 100 percent proof."""

from __future__ import annotations

import pytest

from token_flow_extender.core.coverage_ledger import CoverageLedger
from token_flow_extender.core.window_planner import WindowPlanner
from token_flow_extender.exceptions import CoverageError

from conftest import CharTokenCounter, make_budget


def _plan(config, paragraphs=40):
    text = "\n\n".join(f"Para {i}. " + "content " * 30 for i in range(paragraphs))
    planner = WindowPlanner(CharTokenCounter(), make_budget(400, config))
    return planner.plan(text)


def test_fresh_ledger_is_incomplete(config):
    ledger = CoverageLedger.from_plan(_plan(config))
    report = ledger.report()
    assert not report.complete
    assert report.processed_windows == 0


def test_verify_raises_on_partial_coverage(config):
    plan = _plan(config)
    ledger = CoverageLedger.from_plan(plan)
    for window in plan.windows[:-1]:  # leave one window unprocessed
        ledger.mark_processed(window.index)
    with pytest.raises(CoverageError):
        ledger.verify()


def test_full_coverage_verifies(config):
    plan = _plan(config)
    ledger = CoverageLedger.from_plan(plan)
    for window in plan.windows:
        ledger.mark_processed(window.index)
    report = ledger.verify()
    assert report.complete
    assert report.coverage_ratio == 1.0
    assert report.processed_tokens == report.input_tokens
    assert report.as_dict()["tokens_lost"] == 0


def test_unknown_window_rejected(config):
    ledger = CoverageLedger.from_plan(_plan(config))
    with pytest.raises(CoverageError):
        ledger.mark_processed(999999)


def test_out_of_order_processing_still_completes(config):
    plan = _plan(config)
    ledger = CoverageLedger.from_plan(plan)
    for window in reversed(plan.windows):
        ledger.mark_processed(window.index)
    assert ledger.verify().complete


def test_report_percentages_progress(config):
    plan = _plan(config)
    assert plan.total_windows >= 2
    ledger = CoverageLedger.from_plan(plan)
    ledger.mark_processed(plan.windows[0].index)
    report = ledger.report()
    assert 0 < report.coverage_ratio < 1
    assert report.processed_windows == 1
