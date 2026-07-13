"""The README task table must stay in sync with the specs.

The table's "Varies over" column shows, for every goal placeholder, the
concrete values the specs sample (the union over the task's 5 instances).
This guard keeps the table honest when instances are revised: every
categorical value a language var can take must appear in that task's row.
"""

from pathlib import Path

import pytest

from kitchenbench import SPEC_BY_KEY
from kitchenbench.distributions import Categorical

README = Path(__file__).parent.parent / "README.md"


def _table_row(key: str) -> str:
    rows = [
        line
        for line in README.read_text().splitlines()
        if line.startswith(f"| `kitchenbench/{key}`")
    ]
    assert len(rows) == 1, f"expected exactly one task-table row for {key}"
    return rows[0]


@pytest.mark.parametrize("key", SPEC_BY_KEY)
def test_readme_row_names_every_language_var_value(key: str) -> None:
    row = _table_row(key)
    for inst in SPEC_BY_KEY[key].instances:
        for var in inst.language_vars:
            dist = inst.setup[var]
            if not isinstance(dist, Categorical):
                continue
            for value in dist.values:
                assert str(value) in row, (
                    f"README row for {key} does not mention {var}={value!r} "
                    f"(sampled by {inst.instance_id})"
                )
