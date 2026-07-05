"""End-to-end: eval() runs 5 realizations per instance and the per-scene reduced
task_success is the instance success probability P̂[Yᵢ=1]."""

from __future__ import annotations

# inspect_robots.eval is the framework's evaluation runner (Task rollouts), not
# the Python builtin eval() — no string execution involved.
from inspect_robots import eval as rl_eval
from inspect_robots.logging.sink import NullSink
from inspect_robots.rollout import TrialRecord, derive_seed

from kitchenbench.instances import K_REALIZATIONS
from kitchenbench.specs import SPEC_BY_KEY, SPECS
from kitchenbench.tasks import build_scenes, realize_scene


class _RecordingSink(NullSink):
    """Capture (scene_id, epoch, instruction) seen at each trial's reset observation."""

    def __init__(self) -> None:
        self.trials: list[tuple[str, int, str | None]] = []

    def on_trial_end(self, record: TrialRecord) -> None:
        self.trials.append((record.scene_id, record.epoch, record.steps[0].observation.instruction))


def test_eval_runs_five_realizations_and_scores_probability() -> None:
    sink = _RecordingSink()
    (log,) = rl_eval("kitchenbench/pour_pasta", "kitchen_scripted", "kitchen", sinks=[sink])

    # 5 instances (scenes) x 5 realizations (epochs) = 25 trials.
    assert len(log.samples) == 5
    assert len(sink.trials) == 25

    # The scripted oracle solves every realization -> per-instance P̂ = 1.0.
    for sample in log.samples:
        assert sample.reduced["task_success"] == 1.0
    assert log.status == "success"


def test_eval_realized_instructions_match_realize_scene() -> None:
    # The instruction observed in each trial must be exactly the realization for
    # that (scene, epoch) — i.e. eval() and the realize_scene seam agree on the
    # seed mapping derive_seed(eval_seed=0, scene.init_seed, epoch). Keyed on the
    # captured (scene_id, epoch), not on trial order.
    sink = _RecordingSink()
    rl_eval("kitchenbench/pour_pasta", "kitchen_scripted", "kitchen", sinks=[sink])
    scenes = {scene.id: scene for scene in build_scenes(SPEC_BY_KEY["pour_pasta"])}

    seen = {(scene_id, epoch) for scene_id, epoch, _ in sink.trials}
    assert seen == {(sid, e) for sid in scenes for e in range(K_REALIZATIONS)}
    for scene_id, epoch, instruction in sink.trials:
        scene = scenes[scene_id]
        expected = realize_scene(scene, derive_seed(0, scene.init_seed, epoch)).instruction
        assert instruction == expected
        assert instruction and "{" not in instruction  # concrete, brace-free goal


def test_realizations_vary_across_epochs() -> None:
    # Frozen-seed regression guard: if the per-epoch seed derivation ever
    # collapsed (e.g. realize_scene ignoring the seed), every epoch would sample
    # identical values. Checked on realized .values directly — instructions alone
    # are not distinguishing (many instances have no language vars, so their
    # instruction is identical across epochs by design).
    for spec in SPECS:
        for scene in build_scenes(spec):
            per_epoch = [
                realize_scene(scene, derive_seed(0, scene.init_seed, epoch)).values
                for epoch in range(K_REALIZATIONS)
            ]
            distinct = {tuple(sorted(values.items())) for values in per_epoch}
            assert len(distinct) >= 2, f"frozen realizations for scene {scene.id!r}"


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
