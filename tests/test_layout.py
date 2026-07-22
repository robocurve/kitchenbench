"""Scene-layout export, strict loading, schemas, rig geometry, and CLI behavior."""

from __future__ import annotations

import json
import zlib
from importlib.resources import files
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from inspect_robots.rollout import derive_seed

from kitchenbench.layout import (
    LAYOUT_VERSION,
    export_scene_layout,
    load_rig,
    load_scene_layout,
    main,
    scene_layout_json,
)
from kitchenbench.sim.blueprint import build_blueprint
from kitchenbench.specs import SPEC_BY_KEY
from kitchenbench.tasks import build_scenes

INSTANCE_ID = "seal_container/twist-lid-jar"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(tmp_path: Path, value: object, name: str = "value.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _layout() -> dict[str, Any]:
    return export_scene_layout(INSTANCE_ID)


def _rig() -> dict[str, Any]:
    return json.loads((REPO_ROOT / "rigs" / "yam-bimanual.json").read_text(encoding="utf-8"))


def test_export_is_deterministic_and_epoch_sensitive() -> None:
    first = export_scene_layout(INSTANCE_ID, eval_seed=0, epoch=0)
    second = export_scene_layout(INSTANCE_ID, eval_seed=0, epoch=0)
    later = export_scene_layout(INSTANCE_ID, eval_seed=0, epoch=1)
    assert scene_layout_json(first) == scene_layout_json(second)
    assert scene_layout_json(first).endswith("\n")
    assert first["seed"] != later["seed"]
    assert first["setup_values"] != later["setup_values"]


def test_export_seed_matches_rollout_derivation() -> None:
    eval_seed = 91
    epoch = 3
    expected = derive_seed(
        eval_seed,
        zlib.crc32(INSTANCE_ID.encode()) & 0xFFFFFFFF,
        epoch,
    )
    assert export_scene_layout(INSTANCE_ID, eval_seed=eval_seed, epoch=epoch)["seed"] == expected


@pytest.mark.parametrize(
    "instance_id",
    ["stack/mixed-sizes", "pour_pasta/measuring-cup-to-bowl"],
)
def test_export_preserves_blueprint_objects(instance_id: str) -> None:
    layout = export_scene_layout(instance_id, eval_seed=7, epoch=2)
    key = instance_id.split("/")[0]
    scene = next(
        scene
        for scene in build_scenes(SPEC_BY_KEY[key])
        if scene.metadata["instance_id"] == instance_id
    )
    blueprint = build_blueprint(scene, layout["seed"])
    assert len(layout["objects"]) == len(blueprint.objects)
    for projected, obj in zip(layout["objects"], blueprint.objects, strict=True):
        assert projected["name"] == obj.name
        assert projected["asset"] == obj.asset
        assert projected["role"] == obj.role
        assert projected["frame"] == ("bench" if obj.parent is None else obj.parent)
        assert projected["xy_cm"] == [obj.x_cm, obj.y_cm]
        assert projected["yaw_deg"] == obj.yaw_deg
        assert ("size_cm" in projected) == (obj.size_cm is not None)
        assert ("amount_g" in projected) == (obj.amount_g is not None)
        if obj.size_cm is not None:
            assert projected["size_cm"] == obj.size_cm
        if obj.amount_g is not None:
            assert projected["amount_g"] == obj.amount_g


def test_every_instance_exports_and_matches_packaged_schema() -> None:
    schema = json.loads(
        files("kitchenbench").joinpath("schemas/scene-layout.schema.json").read_text()
    )
    count = 0
    for spec in SPEC_BY_KEY.values():
        for instance in spec.instances:
            jsonschema.validate(export_scene_layout(instance.instance_id), schema)
            count += 1
    assert count == 50


def test_scene_layout_round_trip(tmp_path: Path) -> None:
    layout = export_scene_layout("pour_pasta/measuring-cup-to-bowl", eval_seed=12, epoch=4)
    path = tmp_path / "layout.json"
    path.write_text(scene_layout_json(layout), encoding="utf-8")
    loaded = load_scene_layout(str(path))
    assert loaded.layout_version == layout["layout_version"]
    assert loaded.kitchenbench_version == layout["kitchenbench_version"]
    assert loaded.sim_contract_version == layout["sim_contract_version"]
    assert loaded.instance_id == layout["instance_id"]
    assert loaded.eval_seed == layout["eval_seed"]
    assert loaded.epoch == layout["epoch"]
    assert loaded.seed == layout["seed"]
    assert loaded.instruction == layout["instruction"]
    assert loaded.target_kind == layout["target_kind"]
    assert loaded.conditions == layout["conditions"]
    assert loaded.setup_values == layout["setup_values"]
    for obj, value in zip(loaded.objects, layout["objects"], strict=True):
        assert obj.name == value["name"]
        assert obj.xy_cm == tuple(value["xy_cm"])
        assert obj.size_cm == value.get("size_cm")
        assert obj.amount_g == value.get("amount_g")


def test_layout_accepts_null_role_and_empty_objects(tmp_path: Path) -> None:
    layout = _layout()
    layout["objects"] = []
    assert load_scene_layout(_write_json(tmp_path, layout)).objects == ()
    layout = _layout()
    layout["objects"][0]["role"] = None
    layout["objects"][0]["size_cm"] = 9
    layout["objects"][0]["amount_g"] = 12
    loaded = load_scene_layout(_write_json(tmp_path, layout)).objects[0]
    assert loaded.role is None
    assert loaded.size_cm == 9.0
    assert loaded.amount_g == 12.0


def test_export_rejects_unknown_task_and_instance() -> None:
    with pytest.raises(KeyError, match=r"unknown task key.*retired"):
        export_scene_layout("retired/example")
    with pytest.raises(KeyError, match=r"unknown instance.*seal_container/retired"):
        export_scene_layout("seal_container/retired")


def test_layout_loader_rejects_non_object_and_top_level_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scene layout must be an object"):
        load_scene_layout(_write_json(tmp_path, []))
    layout = _layout()
    layout["typo"] = True
    with pytest.raises(ValueError, match="unknown field 'typo'"):
        load_scene_layout(_write_json(tmp_path, layout))
    layout = _layout()
    del layout["instruction"]
    with pytest.raises(ValueError, match="missing required field 'instruction'"):
        load_scene_layout(_write_json(tmp_path, layout))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("layout_version", "1", "layout_version must be an integer"),
        ("kitchenbench_version", 1, "kitchenbench_version must be a string"),
        ("sim_contract_version", "1", "sim_contract_version must be an integer"),
        ("instance_id", 1, "instance_id must be a string"),
        ("eval_seed", True, "eval_seed must be an integer"),
        ("epoch", "0", "epoch must be an integer"),
        ("seed", [], "seed must be an integer"),
        ("instruction", None, "instruction must be a string"),
        ("target_kind", False, "target_kind must be a string"),
        ("objects", {}, "objects must be a list"),
        ("conditions", [], "conditions must be an object"),
        ("setup_values", [], "setup_values must be an object"),
    ],
)
def test_layout_loader_rejects_wrong_top_level_types(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    layout = _layout()
    layout[field] = value
    with pytest.raises(ValueError, match=match):
        load_scene_layout(_write_json(tmp_path, layout))


def test_layout_loader_rejects_unknown_version(tmp_path: Path) -> None:
    layout = _layout()
    layout["layout_version"] = LAYOUT_VERSION + 1
    with pytest.raises(ValueError, match="layout_version must be 1, got 2"):
        load_scene_layout(_write_json(tmp_path, layout))


def test_layout_loader_rejects_non_object_and_bad_object_keys(tmp_path: Path) -> None:
    layout = _layout()
    layout["objects"][0] = "jar"
    with pytest.raises(ValueError, match=r"objects\[0\] must be an object"):
        load_scene_layout(_write_json(tmp_path, layout))
    layout = _layout()
    layout["objects"][0]["typo"] = 1
    with pytest.raises(ValueError, match="unknown field 'typo'"):
        load_scene_layout(_write_json(tmp_path, layout))
    layout = _layout()
    del layout["objects"][0]["asset"]
    with pytest.raises(ValueError, match="missing required field 'asset'"):
        load_scene_layout(_write_json(tmp_path, layout))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("name", 1, r"objects\[0\].name must be a string"),
        ("asset", 1, r"objects\[0\].asset must be a string"),
        ("role", 1, r"objects\[0\].role must be a string or null"),
        ("frame", 1, r"objects\[0\].frame must be a string"),
        ("xy_cm", "0, 0", r"objects\[0\].xy_cm must be a list"),
        ("xy_cm", [0, 0, 0], r"objects\[0\].xy_cm must contain exactly 2 numbers"),
        ("xy_cm", [True, 0], r"objects\[0\].xy_cm\[0\] must be a number"),
        ("xy_cm", [0, "zero"], r"objects\[0\].xy_cm\[1\] must be a number"),
        ("yaw_deg", True, r"objects\[0\].yaw_deg must be a number"),
        ("yaw_deg", "0", r"objects\[0\].yaw_deg must be a number"),
        ("size_cm", "1", r"objects\[0\].size_cm must be a number"),
        ("amount_g", False, r"objects\[0\].amount_g must be a number"),
    ],
)
def test_layout_loader_rejects_wrong_object_fields(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    layout = _layout()
    layout["objects"][0][field] = value
    with pytest.raises(ValueError, match=match):
        load_scene_layout(_write_json(tmp_path, layout))


def test_rig_loads_and_matches_packaged_schema() -> None:
    path = REPO_ROOT / "rigs" / "yam-bimanual.json"
    rig = load_rig(str(path))
    assert rig.rig_id == "yam-bimanual"
    assert rig.arms["left"].base_xy_cm == (-30.0, -20.0)
    assert rig.arms["right"].reach_cm == 65.0
    assert rig.frame == _rig()["frame"]
    assert rig.bench == {"size_cm": [120.0, 60.0]}
    schema = json.loads(files("kitchenbench").joinpath("schemas/rig.schema.json").read_text())
    jsonschema.validate(_rig(), schema)


def test_minimal_rig_omits_optional_fields(tmp_path: Path) -> None:
    rig = _rig()
    del rig["frame"]
    del rig["bench"]
    del rig["arms"]["left"]["reach_cm"]
    del rig["arms"]["right"]["reach_cm"]
    loaded = load_rig(_write_json(tmp_path, rig))
    assert loaded.frame is None
    assert loaded.bench is None
    assert loaded.arms["left"].reach_cm is None


def test_rig_loader_rejects_non_object_and_top_level_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rig must be an object"):
        load_rig(_write_json(tmp_path, []))
    rig = _rig()
    rig["typo"] = True
    with pytest.raises(ValueError, match="unknown field 'typo'"):
        load_rig(_write_json(tmp_path, rig))
    rig = _rig()
    del rig["rig_id"]
    with pytest.raises(ValueError, match="missing required field 'rig_id'"):
        load_rig(_write_json(tmp_path, rig))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("layout_version", "1", "layout_version must be an integer"),
        ("rig_id", 1, "rig_id must be a string"),
        ("rig_id", "", "rig_id must be a non-empty string"),
        ("arms", [], "arms must be an object"),
        ("frame", [], "frame must be an object"),
        ("bench", [], "bench must be an object"),
    ],
)
def test_rig_loader_rejects_wrong_top_level_types(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    rig = _rig()
    rig[field] = value
    with pytest.raises(ValueError, match=match):
        load_rig(_write_json(tmp_path, rig))


def test_rig_loader_rejects_unknown_version(tmp_path: Path) -> None:
    rig = _rig()
    rig["layout_version"] = 2
    with pytest.raises(ValueError, match="layout_version must be 1, got 2"):
        load_rig(_write_json(tmp_path, rig))


def test_rig_loader_rejects_missing_or_unknown_arm(tmp_path: Path) -> None:
    rig = _rig()
    del rig["arms"]["left"]
    with pytest.raises(ValueError, match="arms is missing required field 'left'"):
        load_rig(_write_json(tmp_path, rig))
    rig = _rig()
    rig["arms"]["center"] = rig["arms"]["left"]
    with pytest.raises(ValueError, match="arms has unknown field 'center'"):
        load_rig(_write_json(tmp_path, rig))


def test_rig_loader_rejects_non_object_and_bad_arm_keys(tmp_path: Path) -> None:
    rig = _rig()
    rig["arms"]["left"] = []
    with pytest.raises(ValueError, match=r"arms\.left must be an object"):
        load_rig(_write_json(tmp_path, rig))
    rig = _rig()
    rig["arms"]["left"]["typo"] = 1
    with pytest.raises(ValueError, match=r"arms\.left has unknown field 'typo'"):
        load_rig(_write_json(tmp_path, rig))
    rig = _rig()
    del rig["arms"]["left"]["faces"]
    with pytest.raises(ValueError, match=r"arms\.left is missing required field 'faces'"):
        load_rig(_write_json(tmp_path, rig))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("faces", 1, "arms.left.faces must be a string"),
        ("faces", "up", "arms.left.faces must be one of"),
        ("base_xy_cm", "0, 0", "arms.left.base_xy_cm must be a list"),
        ("base_xy_cm", [-30, -20, 0], "must contain exactly 2 numbers"),
        ("reach_cm", "65", "arms.left.reach_cm must be a number"),
        ("reach_cm", 0, "arms.left.reach_cm must be greater than 0"),
    ],
)
def test_rig_loader_rejects_wrong_arm_fields(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    rig = _rig()
    rig["arms"]["left"][field] = value
    with pytest.raises(ValueError, match=match):
        load_rig(_write_json(tmp_path, rig))


@pytest.mark.parametrize(
    ("side", "base"),
    [("left", [30, -20]), ("right", [-30, -20])],
)
def test_rig_loader_enforces_arm_x_convention(tmp_path: Path, side: str, base: list[int]) -> None:
    rig = _rig()
    rig["arms"][side]["base_xy_cm"] = base
    with pytest.raises(ValueError, match="left base x < 0 < right base x"):
        load_rig(_write_json(tmp_path, rig))


def test_rig_loader_rejects_bad_bench_keys_and_size(tmp_path: Path) -> None:
    rig = _rig()
    rig["bench"]["typo"] = 1
    with pytest.raises(ValueError, match="bench has unknown field 'typo'"):
        load_rig(_write_json(tmp_path, rig))
    rig = _rig()
    del rig["bench"]["size_cm"]
    with pytest.raises(ValueError, match="bench is missing required field 'size_cm'"):
        load_rig(_write_json(tmp_path, rig))
    rig = _rig()
    rig["bench"]["size_cm"] = [120, 60, 1]
    with pytest.raises(ValueError, match=r"bench\.size_cm must contain exactly 2 numbers"):
        load_rig(_write_json(tmp_path, rig))


def test_cli_writes_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([INSTANCE_ID, "--eval-seed", "5", "--epoch", "2"]) == 0
    captured = capsys.readouterr()
    assert (
        json.loads(captured.out)["seed"]
        == export_scene_layout(INSTANCE_ID, eval_seed=5, epoch=2)["seed"]
    )
    assert captured.err == ""


def test_cli_writes_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "layout.json"
    assert main([INSTANCE_ID, "-o", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == scene_layout_json(export_scene_layout(INSTANCE_ID))
    assert capsys.readouterr().out == ""


def test_cli_reports_unknown_instance(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["seal_container/retired"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unknown instance 'seal_container/retired'" in captured.err
