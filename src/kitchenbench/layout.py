"""Export and validate deterministic scene layouts and hand-authored rig geometry.

Scene layouts are regenerable projections of a benchmark instance and trial
seed. Rig files bind those layouts to the same bench frame used by the sim
contract: +x is robot right, with the left arm based at -x.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

from inspect_robots.rollout import derive_seed

from kitchenbench.sim.blueprint import build_blueprint
from kitchenbench.specs import SPEC_BY_KEY
from kitchenbench.tasks import build_scenes

#: Version of the scene-layout and rig file contracts.
LAYOUT_VERSION = 1

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class LayoutObject:
    """One object's frame-local pose and optional realized physical properties."""

    name: str
    asset: str
    role: str | None
    frame: str
    xy_cm: tuple[float, float]
    yaw_deg: float
    size_cm: float | None = None
    amount_g: float | None = None


@dataclass(frozen=True)
class SceneLayout:
    """A validated, deterministic projection of one KitchenBench trial setup."""

    layout_version: int
    kitchenbench_version: str
    sim_contract_version: int
    instance_id: str
    eval_seed: int
    epoch: int
    seed: int
    instruction: str
    target_kind: str
    objects: tuple[LayoutObject, ...]
    conditions: dict[str, JsonValue]
    setup_values: dict[str, JsonValue]


@dataclass(frozen=True)
class ArmSpec:
    """A validated arm base pose, facing direction, and optional reach."""

    base_xy_cm: tuple[float, float]
    faces: str
    reach_cm: float | None = None


@dataclass(frozen=True)
class Rig:
    """A validated bimanual rig expressed in the KitchenBench bench frame."""

    layout_version: int
    rig_id: str
    arms: dict[str, ArmSpec]
    frame: dict[str, JsonValue] | None = None
    bench: dict[str, JsonValue] | None = None


def export_scene_layout(instance_id: str, *, eval_seed: int = 0, epoch: int = 0) -> dict[str, Any]:
    """Project one instance and trial coordinates into the layout JSON contract.

    Raises ``KeyError`` when the task key or full instance identifier is not in
    the shipped benchmark specification.
    """
    key = instance_id.split("/")[0]
    try:
        spec = SPEC_BY_KEY[key]
    except KeyError:
        raise KeyError(f"unknown task key {key!r} from instance {instance_id!r}") from None

    scene = next(
        (
            candidate
            for candidate in build_scenes(spec)
            if candidate.metadata["instance_id"] == instance_id
        ),
        None,
    )
    if scene is None:
        raise KeyError(f"unknown instance {instance_id!r} in task {key!r}")

    seed = derive_seed(eval_seed, scene.init_seed, epoch)
    blueprint = build_blueprint(scene, seed)
    objects: list[dict[str, Any]] = []
    for obj in blueprint.objects:
        projected: dict[str, Any] = {
            "name": obj.name,
            "asset": obj.asset,
            "role": obj.role,
            "frame": "bench" if obj.parent is None else obj.parent,
            "xy_cm": [obj.x_cm, obj.y_cm],
            "yaw_deg": obj.yaw_deg,
        }
        if obj.size_cm is not None:
            projected["size_cm"] = obj.size_cm
        if obj.amount_g is not None:
            projected["amount_g"] = obj.amount_g
        objects.append(projected)

    from kitchenbench import __version__

    return {
        "layout_version": LAYOUT_VERSION,
        "kitchenbench_version": __version__,
        "sim_contract_version": blueprint.contract_version,
        "instance_id": instance_id,
        "eval_seed": eval_seed,
        "epoch": epoch,
        "seed": seed,
        "instruction": blueprint.instruction,
        "target_kind": blueprint.target_kind,
        "objects": objects,
        "conditions": dict(blueprint.conditions),
        "setup_values": dict(blueprint.values),
    }


def scene_layout_json(layout: dict[str, Any]) -> str:
    """Serialize a scene layout in the canonical stable text representation."""
    return json.dumps(layout, sort_keys=True, indent=2) + "\n"


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return cast("dict[str, object]", value)


def _list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return cast("list[object]", value)


def _keys(
    value: Mapping[str, object],
    field: str,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    unknown = set(value) - required - optional
    if unknown:
        raise ValueError(f"{field} has unknown field {min(unknown)!r}")
    missing = required - set(value)
    if missing:
        raise ValueError(f"{field} is missing required field {min(missing)!r}")


def _string(value: object, field: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if nonempty and not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    return float(value)


def _pair(value: object, field: str) -> tuple[float, float]:
    values = _list(value, field)
    if len(values) != 2:
        raise ValueError(f"{field} must contain exactly 2 numbers")
    return _number(values[0], f"{field}[0]"), _number(values[1], f"{field}[1]")


def _json_mapping(value: object, field: str) -> dict[str, JsonValue]:
    mapping = _mapping(value, field)
    return cast("dict[str, JsonValue]", mapping)


_LAYOUT_KEYS = frozenset(
    {
        "layout_version",
        "kitchenbench_version",
        "sim_contract_version",
        "instance_id",
        "eval_seed",
        "epoch",
        "seed",
        "instruction",
        "target_kind",
        "objects",
        "conditions",
        "setup_values",
    }
)
_OBJECT_KEYS = frozenset({"name", "asset", "role", "frame", "xy_cm", "yaw_deg"})
_OBJECT_OPTIONAL_KEYS = frozenset({"size_cm", "amount_g"})


def _layout_object(value: object, index: int) -> LayoutObject:
    field = f"objects[{index}]"
    mapping = _mapping(value, field)
    _keys(
        mapping,
        field,
        required=_OBJECT_KEYS,
        optional=_OBJECT_OPTIONAL_KEYS,
    )
    role_value = mapping["role"]
    if role_value is not None and not isinstance(role_value, str):
        raise ValueError(f"{field}.role must be a string or null")
    return LayoutObject(
        name=_string(mapping["name"], f"{field}.name"),
        asset=_string(mapping["asset"], f"{field}.asset"),
        role=role_value,
        frame=_string(mapping["frame"], f"{field}.frame"),
        xy_cm=_pair(mapping["xy_cm"], f"{field}.xy_cm"),
        yaw_deg=_number(mapping["yaw_deg"], f"{field}.yaw_deg"),
        size_cm=(_number(mapping["size_cm"], f"{field}.size_cm") if "size_cm" in mapping else None),
        amount_g=(
            _number(mapping["amount_g"], f"{field}.amount_g") if "amount_g" in mapping else None
        ),
    )


def load_scene_layout(path: str | Path) -> SceneLayout:
    """Read and strictly validate a scene-layout JSON file."""
    raw = cast(object, json.loads(Path(path).read_text(encoding="utf-8")))
    layout = _mapping(raw, "scene layout")
    _keys(layout, "scene layout", required=_LAYOUT_KEYS)
    layout_version = _integer(layout["layout_version"], "layout_version")
    if layout_version != LAYOUT_VERSION:
        raise ValueError(f"layout_version must be {LAYOUT_VERSION}, got {layout_version}")
    objects = _list(layout["objects"], "objects")
    return SceneLayout(
        layout_version=layout_version,
        kitchenbench_version=_string(layout["kitchenbench_version"], "kitchenbench_version"),
        sim_contract_version=_integer(layout["sim_contract_version"], "sim_contract_version"),
        instance_id=_string(layout["instance_id"], "instance_id"),
        eval_seed=_integer(layout["eval_seed"], "eval_seed"),
        epoch=_integer(layout["epoch"], "epoch"),
        seed=_integer(layout["seed"], "seed"),
        instruction=_string(layout["instruction"], "instruction"),
        target_kind=_string(layout["target_kind"], "target_kind"),
        objects=tuple(_layout_object(obj, index) for index, obj in enumerate(objects)),
        conditions=_json_mapping(layout["conditions"], "conditions"),
        setup_values=_json_mapping(layout["setup_values"], "setup_values"),
    )


_RIG_KEYS = frozenset({"layout_version", "rig_id", "arms"})
_RIG_OPTIONAL_KEYS = frozenset({"frame", "bench"})
_ARM_KEYS = frozenset({"base_xy_cm", "faces"})
_ARM_OPTIONAL_KEYS = frozenset({"reach_cm"})
_FACES = frozenset({"+x", "-x", "+y", "-y"})


def _arm(value: object, side: str) -> ArmSpec:
    field = f"arms.{side}"
    mapping = _mapping(value, field)
    _keys(mapping, field, required=_ARM_KEYS, optional=_ARM_OPTIONAL_KEYS)
    faces = _string(mapping["faces"], f"{field}.faces")
    if faces not in _FACES:
        raise ValueError(f"{field}.faces must be one of +x, -x, +y, -y")
    reach = _number(mapping["reach_cm"], f"{field}.reach_cm") if "reach_cm" in mapping else None
    if reach is not None and reach <= 0:
        raise ValueError(f"{field}.reach_cm must be greater than 0")
    return ArmSpec(
        base_xy_cm=_pair(mapping["base_xy_cm"], f"{field}.base_xy_cm"),
        faces=faces,
        reach_cm=reach,
    )


def load_rig(path: str | Path) -> Rig:
    """Read a rig JSON file and enforce its schema and bench-frame handedness."""
    raw = cast(object, json.loads(Path(path).read_text(encoding="utf-8")))
    rig = _mapping(raw, "rig")
    _keys(rig, "rig", required=_RIG_KEYS, optional=_RIG_OPTIONAL_KEYS)
    layout_version = _integer(rig["layout_version"], "layout_version")
    if layout_version != LAYOUT_VERSION:
        raise ValueError(f"layout_version must be {LAYOUT_VERSION}, got {layout_version}")

    arms_value = _mapping(rig["arms"], "arms")
    _keys(arms_value, "arms", required=frozenset({"left", "right"}))
    arms = {side: _arm(arms_value[side], side) for side in ("left", "right")}
    if not (arms["left"].base_xy_cm[0] < 0 < arms["right"].base_xy_cm[0]):
        raise ValueError("arms must follow the bench convention: left base x < 0 < right base x")

    frame = _json_mapping(rig["frame"], "frame") if "frame" in rig else None
    bench: dict[str, JsonValue] | None = None
    if "bench" in rig:
        bench_value = _mapping(rig["bench"], "bench")
        _keys(bench_value, "bench", required=frozenset({"size_cm"}))
        size_cm = _pair(bench_value["size_cm"], "bench.size_cm")
        bench = {"size_cm": [size_cm[0], size_cm[1]]}

    return Rig(
        layout_version=layout_version,
        rig_id=_string(rig["rig_id"], "rig_id", nonempty=True),
        arms=arms,
        frame=frame,
        bench=bench,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scene-layout exporter command and report lookup failures as usage errors."""
    parser = argparse.ArgumentParser(prog="kitchenbench-layout")
    parser.add_argument("instance_id")
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(argv)
    try:
        rendered = scene_layout_json(
            export_scene_layout(args.instance_id, eval_seed=args.eval_seed, epoch=args.epoch)
        )
    except KeyError as exc:
        print(f"kitchenbench-layout: error: {exc.args[0]}", file=sys.stderr)
        return 2
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0
