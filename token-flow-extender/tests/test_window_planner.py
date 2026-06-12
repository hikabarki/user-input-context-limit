"""Tests for core/window_planner.py — lossless structure-aware splitting."""

from __future__ import annotations

import pytest

from token_flow_extender.core.window_planner import WindowPlanner
from token_flow_extender.exceptions import WindowPlanError

from conftest import CharTokenCounter, make_budget


def _planner(config, max_input=400):
    return WindowPlanner(CharTokenCounter(), make_budget(max_input, config))


def test_empty_input_rejected(config):
    with pytest.raises(WindowPlanError):
        _planner(config).plan("")


def test_small_input_single_window(config):
    plan = _planner(config, max_input=10000).plan("Short text.")
    assert plan.total_windows == 1
    assert plan.windows[0].char_start == 0
    assert plan.windows[0].char_end == len("Short text.")


def test_spans_tile_input_exactly(config):
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(60))
    plan = _planner(config).plan(text)
    assert plan.total_windows > 1
    spans = sorted((w.char_start, w.char_end) for w in plan.windows)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        assert next_start == prev_end  # no gaps, no primary overlaps


def test_window_text_matches_spans(config):
    text = "\n\n".join(f"Para {i}. " + "data " * 30 for i in range(40))
    plan = _planner(config).plan(text)
    for window in plan.windows:
        assert window.text == text[window.char_start:window.char_end]


def test_reassembly_is_lossless(config):
    text = "\n\n".join(f"Section {i}: " + "alpha beta gamma " * 25 for i in range(50))
    plan = _planner(config).plan(text)
    reassembled = "".join(w.text for w in sorted(plan.windows, key=lambda w: w.char_start))
    assert reassembled == text  # byte-identical reconstruction


def test_code_block_never_split_when_it_fits(config):
    code = "```python\n" + "x = 1\n" * 40 + "```"
    text = ("padding sentence. " * 80) + "\n\n" + code + "\n\n" + ("more padding. " * 80)
    plan = _planner(config, max_input=600).plan(text)
    code_start = text.index("```python")
    code_end = text.index("```", code_start + 3) + 3
    holders = [w for w in plan.windows
               if w.char_start <= code_start and w.char_end >= code_end]
    assert holders, "code block was split across windows although it fits one window"


def test_oversized_unit_gets_continuation_parts(config):
    huge_code = "```\n" + ("data line\n" * 2000) + "```"  # far over any window
    plan = _planner(config, max_input=400).plan(huge_code)
    continuations = [w for w in plan.windows if w.is_continuation]
    assert len(continuations) >= 2
    assert all(w.parts == len(continuations) for w in continuations)
    assert [w.part for w in continuations] == list(range(1, len(continuations) + 1))


def test_window_token_budget_respected(config):
    budget = make_budget(400, config)
    text = "word " * 5000
    plan = WindowPlanner(CharTokenCounter(), budget).plan(text)
    for window in plan.windows:
        assert window.token_count <= budget.usable_window_tokens


def test_token_ranges_are_contiguous(config):
    text = "\n\n".join("paragraph " * 30 for _ in range(30))
    plan = _planner(config).plan(text)
    cursor = 0
    for window in plan.windows:
        assert window.token_start == cursor
        assert window.token_end == cursor + window.token_count
        cursor = window.token_end
    assert cursor == plan.input_tokens


def test_deterministic_plan_hash(config):
    text = "\n\n".join(f"P{i} " + "content " * 40 for i in range(30))
    plan_a = _planner(config).plan(text)
    plan_b = _planner(config).plan(text)
    assert plan_a.plan_id == plan_b.plan_id
    assert [(w.char_start, w.char_end) for w in plan_a.windows] == \
           [(w.char_start, w.char_end) for w in plan_b.windows]


def test_overlap_is_duplicate_context_only(config):
    text = "\n\n".join(f"Para {i}. " + "filler " * 40 for i in range(40))
    plan = _planner(config).plan(text)
    assert plan.windows[0].overlap_text == ""
    for prev, window in zip(plan.windows, plan.windows[1:]):
        if window.overlap_text:
            assert prev.text.endswith(window.overlap_text)


def test_korean_text_plans_losslessly(config):
    text = "\n\n".join("한국어 문단입니다. 토큰 한도를 시험합니다. " * 10 for _ in range(30))
    plan = _planner(config).plan(text)
    reassembled = "".join(w.text for w in sorted(plan.windows, key=lambda w: w.char_start))
    assert reassembled == text
