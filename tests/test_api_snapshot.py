"""Pin the public API — additions and removals must edit this list consciously."""

from __future__ import annotations

import kitchenbench

EXPECTED_API = [
    "ASSETS",
    "SIM_CONTRACT_VERSION",
    "SPECS",
    "SPEC_BY_KEY",
    "TASK_FACTORIES",
    "AssetSpec",
    "Categorical",
    "Constant",
    "Distribution",
    "K_EXPERTS",
    "K_INSTANCES",
    "K_REALIZATIONS",
    "LAYOUT_VERSION",
    "KitchenEmbodiment",
    "NoopKitchenPolicy",
    "Normal",
    "RandomKitchenPolicy",
    "Realization",
    "SceneBlueprint",
    "SceneObject",
    "ScriptedKitchenPolicy",
    "SimObject",
    "SimSpec",
    "TaskInstance",
    "TaskSpec",
    "Uniform",
    "Validation",
    "Var",
    "Verdict",
    "WorldState",
    "__version__",
    "build_blueprint",
    "build_scenes",
    "export_scene_layout",
    "fold_cloth",
    "handoff",
    "load_rig",
    "load_scene_layout",
    "make_success_checker",
    "make_task",
    "open_container",
    "place_cutlery",
    "place_in_rack",
    "pour_pasta",
    "realize_scene",
    "scoop_pasta",
    "seal_container",
    "sort_cutlery",
    "stack",
    "task_success",
]


def test_public_api_snapshot() -> None:
    assert sorted(kitchenbench.__all__) == sorted(EXPECTED_API)


def test_all_names_resolve() -> None:
    for name in kitchenbench.__all__:
        assert hasattr(kitchenbench, name), name
