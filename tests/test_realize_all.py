"""Realize every authored instance — catches goal/placeholder/format bugs across
all 50 instances and exercises every distribution's sample + describe."""

from __future__ import annotations

import json

from kitchenbench.specs import SPECS


def test_realize_every_instance() -> None:
    for spec in SPECS:
        for inst in spec.instances:
            r = inst.realize(0)
            # Goal formatted (no leftover braces), values cover the setup, JSON-native.
            assert "{" not in r.instruction and "}" not in r.instruction, inst.instance_id
            assert set(r.values) == set(inst.setup), inst.instance_id
            json.dumps(r.values)
            assert len(r.setup_lines) == len(inst.setup)


def test_describe_every_distribution() -> None:
    for spec in SPECS:
        for inst in spec.instances:
            spec_strings = inst.setup_spec()
            assert set(spec_strings) == set(inst.setup)
            for text in spec_strings.values():
                assert text  # non-empty description


def test_realizations_reproduce_per_seed() -> None:
    for spec in SPECS:
        for inst in spec.instances:
            assert inst.realize(7) == inst.realize(7), inst.instance_id
