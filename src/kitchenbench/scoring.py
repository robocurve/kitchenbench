"""KitchenBench scoring.

``task_success`` works both in the dependency-free mock (which reports privileged
success via the termination reason) and on real hardware: there is no success
oracle on a real kitchen, so the real embodiment turns the operator's
confirmation into either a ``"success"`` termination reason or a recorded
operator verdict — ``task_success`` accepts either.
"""

from __future__ import annotations

from dataclasses import dataclass

from robolens import Score, Scorer, Target, TrialRecord

# Affirmative operator verdicts (case-insensitive), for real-world runs.
_AFFIRMATIVE = frozenset({"success", "pass", "yes", "y", "1", "true"})


@dataclass(frozen=True)
class _TaskSuccess:
    name: str = "task_success"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        if record.termination_reason == "success":
            return Score(value=True, explanation="terminated with success")
        verdict = record.operator_judgement
        if verdict is not None and verdict.strip().lower() in _AFFIRMATIVE:
            return Score(value=True, explanation=f"operator verdict: {verdict!r}")
        return Score(value=False, explanation="no success signal")


def task_success() -> Scorer:
    """Success iff the trial terminated successfully or an operator confirmed it."""
    return _TaskSuccess()
