"""Task generation, scene expansion, and registry reachability."""

from __future__ import annotations

from importlib.metadata import entry_points

import pytest
from robolens import Task
from robolens.registry import registered, resolve

from kitchenbench.specs import SPECS, TaskSpec
from kitchenbench.tasks import TASK_FACTORIES, build_scenes, make_task


def _expected_scene_count(spec: TaskSpec) -> int:
    n = 1
    for values in spec.axes.values():
        n *= len(values)
    return n


def test_all_specs_have_a_factory() -> None:
    assert set(TASK_FACTORIES) == {spec.key for spec in SPECS}


def test_total_scene_count() -> None:
    total = sum(len(factory().scenes) for factory in TASK_FACTORIES.values())
    assert total == sum(_expected_scene_count(spec) for spec in SPECS) == 37


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_task_shape(spec: TaskSpec) -> None:
    task = TASK_FACTORIES[spec.key]()
    assert isinstance(task, Task)
    assert task.name == f"kitchenbench/{spec.key}"
    assert len(task.scenes) == _expected_scene_count(spec)
    ids = [s.id for s in task.scenes]
    assert len(ids) == len(set(ids))  # unique scene ids
    for scene in task.scenes:
        assert "{" not in scene.instruction and "}" not in scene.instruction
        assert scene.target is not None
        assert scene.target.kind == spec.target_kind
        for axis, value in scene.metadata["axes"].items():
            assert scene.target.spec[axis] == value
        for k, v in spec.extra.items():
            assert scene.target.spec[k] == v
        assert scene.metadata["bimanual"] == spec.bimanual


def test_make_task_has_two_scorers() -> None:
    task = make_task(SPECS[0])
    assert len(task.scorers) == 2
    assert {s.name for s in task.scorers} == {"task_success", "episode_length"}


def test_build_scenes_slugifies_ids() -> None:
    spec = next(s for s in SPECS if s.key == "scoop_pasta")
    scenes = build_scenes(spec)
    # "measuring cup" -> "measuring-cup" in the id, instruction keeps the space.
    assert any("measuring-cup" in s.id for s in scenes)
    assert any("measuring cup" in s.instruction for s in scenes)


def test_tasks_registered_in_robolens() -> None:
    reg = registered("task")
    for spec in SPECS:
        name = f"kitchenbench/{spec.key}"
        assert name in reg
        assert resolve("task", name).name == name


def test_entry_points_match_specs() -> None:
    eps = {
        ep.name
        for ep in entry_points(group="robolens.tasks")
        if ep.name.startswith("kitchenbench/")
    }
    assert eps == {f"kitchenbench/{spec.key}" for spec in SPECS}
    # one round-trips through load() to the same Task name
    ep = next(e for e in entry_points(group="robolens.tasks") if e.name == "kitchenbench/handoff")
    assert ep.load()().name == "kitchenbench/handoff"


def test_embodiment_and_policies_registered() -> None:
    assert "kitchen" in registered("embodiment")
    assert {"kitchen_scripted", "kitchen_random", "kitchen_noop"} <= set(registered("policy"))
