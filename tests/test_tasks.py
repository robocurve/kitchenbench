"""Scene/task construction from distribution-based instances + registry reachability."""

from __future__ import annotations

import json
from dataclasses import replace
from importlib.metadata import entry_points

import pytest
from robolens import Epochs, Task
from robolens.registry import registered, resolve
from robolens.rollout import derive_seed

from kitchenbench.instances import K_INSTANCES, K_REALIZATIONS
from kitchenbench.specs import SPEC_BY_KEY, SPECS, TaskSpec
from kitchenbench.tasks import TASK_FACTORIES, build_scenes, make_task, realize_scene


def test_all_specs_have_a_factory() -> None:
    assert set(TASK_FACTORIES) == {spec.key for spec in SPECS}


def test_total_scene_count() -> None:
    total = sum(len(factory().scenes) for factory in TASK_FACTORIES.values())
    assert total == len(SPECS) * K_INSTANCES == 50


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_task_shape(spec: TaskSpec) -> None:
    task = TASK_FACTORIES[spec.key]()
    assert isinstance(task, Task)
    assert task.name == f"kitchenbench/{spec.key}"
    assert len(task.scenes) == K_INSTANCES == 5
    ids = [s.id for s in task.scenes]
    assert len(ids) == len(set(ids))  # unique scene ids
    # 5 realizations per instance, mean-reduced -> per-scene success probability.
    assert task.epochs == Epochs(count=K_REALIZATIONS, reducer="mean")
    for index, scene in enumerate(task.scenes):
        assert "{" not in scene.instruction and "}" not in scene.instruction
        assert scene.target is not None
        assert scene.target.kind == spec.instances[index].target_kind
        assert scene.metadata["bimanual"] == spec.bimanual
        assert scene.metadata["instance_index"] == index


def test_scene_metadata_is_json_native() -> None:
    scene = build_scenes(SPEC_BY_KEY["pour_pasta"])[0]
    json.dumps(dict(scene.metadata))  # must not raise
    assert scene.metadata["benchmark"] == "kitchenbench"
    assert scene.metadata["task"] == "pour_pasta"
    assert scene.metadata["setup"]["vessel"].startswith("Categorical(")
    assert scene.metadata["validation"]["source"] == "opus-draft"
    assert isinstance(scene.metadata["language_vars"], list)
    assert isinstance(scene.metadata["validation"]["representativeness"], list)


def test_canonical_instruction_matches_epoch_zero() -> None:
    spec = SPEC_BY_KEY["pour_pasta"]
    for index, scene in enumerate(build_scenes(spec)):
        expected = spec.instances[index].realize(derive_seed(0, index, 0)).instruction
        assert scene.instruction == expected


def test_make_task_has_two_scorers() -> None:
    task = make_task(SPECS[0])
    assert {s.name for s in task.scorers} == {"task_success", "episode_length"}
    assert task.metadata["k_instances"] == 5
    assert task.metadata["k_realizations"] == 5


def test_realize_scene_recovers_and_guards_none() -> None:
    scene = build_scenes(SPEC_BY_KEY["pour_pasta"])[0]
    r = realize_scene(scene, 123)
    assert r.values["vessel"] in ("bowl", "cup", "pot")
    assert realize_scene(scene, None) == realize_scene(scene, 0)  # None -> 0


def test_realize_scene_fails_loudly_on_unknown_instance_id() -> None:
    # A scene logged under a removed/renamed instance must not silently realize a
    # different one — recovery is by instance_id, not position.
    scene = build_scenes(SPEC_BY_KEY["pour_pasta"])[0]
    stale = replace(scene, metadata={**scene.metadata, "instance_id": "pour_pasta/retired"})
    with pytest.raises(LookupError, match="pour_pasta/retired"):
        realize_scene(stale, 0)


def test_build_scenes_slugifies_ids() -> None:
    scenes = build_scenes(SPEC_BY_KEY["scoop_pasta"])
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
    ep = next(e for e in entry_points(group="robolens.tasks") if e.name == "kitchenbench/handoff")
    assert ep.load()().name == "kitchenbench/handoff"


def test_embodiment_and_policies_registered() -> None:
    assert "kitchen" in registered("embodiment")
    assert {"kitchen_scripted", "kitchen_random", "kitchen_noop"} <= set(registered("policy"))
