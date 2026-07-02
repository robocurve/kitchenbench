"""End-to-end: eval() runs 5 realizations per instance and the per-scene reduced
task_success is the instance success probability P̂[Yᵢ=1]."""

from __future__ import annotations

from inspect_robots import eval as rl_eval
from inspect_robots.logging.sink import NullSink
from inspect_robots.rollout import TrialRecord

from kitchenbench.tasks import realize_scene


class _RecordingSink(NullSink):
    """Capture the realized instruction seen at each trial's reset observation."""

    def __init__(self) -> None:
        self.instructions: list[str | None] = []

    def on_trial_end(self, record: TrialRecord) -> None:
        self.instructions.append(record.steps[0].observation.instruction)


def test_eval_runs_five_realizations_and_scores_probability() -> None:
    sink = _RecordingSink()
    (log,) = rl_eval("kitchenbench/pour_pasta", "kitchen_scripted", "kitchen", sinks=[sink])

    # 5 instances (scenes) x 5 realizations (epochs) = 25 trials.
    assert len(log.samples) == 5
    assert len(sink.instructions) == 25

    # The scripted oracle solves every realization -> per-instance P̂ = 1.0.
    for sample in log.samples:
        assert sample.reduced["task_success"] == 1.0
    assert log.status == "success"


def test_eval_realized_instructions_match_realize_scene() -> None:
    sink = _RecordingSink()
    rl_eval("kitchenbench/pour_pasta", "kitchen_scripted", "kitchen", sinks=[sink])
    # Every captured instruction is a concrete (brace-free) realized goal.
    assert all(i and "{" not in i for i in sink.instructions)
    # And the pour vessel vocabulary shows up across realizations.
    assert any("bowl" in i or "cup" in i or "pot" in i for i in sink.instructions if i)


def test_failing_policy_scores_zero_probability() -> None:
    (log,) = rl_eval("kitchenbench/pour_pasta", "kitchen_noop", "kitchen", sinks=[])
    for sample in log.samples:
        assert sample.reduced["task_success"] == 0.0


def test_realize_scene_is_the_run_seam() -> None:
    # Sanity: the helper a real embodiment uses produces brace-free instructions.
    from inspect_robots.registry import resolve

    task = resolve("task", "kitchenbench/pour_pasta")
    for scene in task.scenes:
        assert "{" not in realize_scene(scene, 0).instruction
