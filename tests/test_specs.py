"""Tests for the distribution-based task specifications."""

from __future__ import annotations

import string

from kitchenbench.instances import K_INSTANCES
from kitchenbench.specs import SPEC_BY_KEY, SPECS

_EXPECTED_KEYS = {
    "place_cutlery",
    "stack",
    "place_in_rack",
    "pour_pasta",
    "open_container",
    "fold_cloth",
    "seal_container",
    "handoff",
    "sort_cutlery",
    "scoop_pasta",
}


def _placeholders(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


def test_exactly_ten_tasks() -> None:
    assert {s.key for s in SPECS} == _EXPECTED_KEYS
    assert set(SPEC_BY_KEY) == _EXPECTED_KEYS


def test_each_task_has_k_instances() -> None:
    for spec in SPECS:
        assert len(spec.instances) == K_INSTANCES, spec.key


def test_max_seconds_positive() -> None:
    for spec in SPECS:
        assert spec.max_seconds > 0.0, spec.key


def test_language_vars_and_placeholders_consistent() -> None:
    for spec in SPECS:
        for inst in spec.instances:
            assert set(inst.language_vars) <= set(inst.setup), inst.instance_id
            # every {placeholder} in the goal must be a declared language var...
            assert _placeholders(inst.goal) <= set(inst.language_vars), inst.instance_id
            # ...and (transitively) backed by a setup distribution.
            assert _placeholders(inst.goal) <= set(inst.setup), inst.instance_id


def test_target_kinds_non_empty() -> None:
    for spec in SPECS:
        for inst in spec.instances:
            assert inst.target_kind, inst.instance_id


def test_instance_ids_globally_unique() -> None:
    ids = [inst.instance_id for spec in SPECS for inst in spec.instances]
    assert len(ids) == len(set(ids)) == 50


def test_instances_are_unvalidated_drafts() -> None:
    # Honest provenance: AI-authored, not yet human-validated.
    for spec in SPECS:
        for inst in spec.instances:
            assert inst.validation.source == "opus-draft"
            assert inst.validation.validated is False
