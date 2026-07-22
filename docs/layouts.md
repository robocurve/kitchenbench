# Layouts

KitchenBench scene layouts provide machine-readable object poses for one trial.
Rig files describe how a physical bimanual system relates to the same bench
frame.

## Frame convention

The bench origin is between the arm bases at table height. The +x axis points
to the robot's right, and +y points away from the arms. The left arm is based
at negative x and the right arm is based at positive x. Object coordinates are
frame-local. An object whose `frame` is `bench` uses bench coordinates. Any
other value names the parent object's frame.

Lengths are centimeters and angles are degrees.

## Scene layout format

A scene layout is generated from an instance and trial coordinates. Optional
`size_cm` and `amount_g` fields appear only when the realized object defines
them.

```json
{
  "layout_version": 1,
  "kitchenbench_version": "<kitchenbench.__version__>",
  "sim_contract_version": 1,
  "instance_id": "seal_container/twist-lid-jar",
  "eval_seed": 0,
  "epoch": 0,
  "seed": 812041,
  "instruction": "seal the jar with its lid",
  "target_kind": "seal",
  "objects": [
    {"name": "jar", "asset": "jar", "role": "container", "frame": "bench", "xy_cm": [-1.4, 0.0], "yaw_deg": 0.0},
    {"name": "lid", "asset": "lid", "role": "lid", "frame": "jar", "xy_cm": [8.0, 0.0], "yaw_deg": 0.0}
  ],
  "conditions": {"thread_turns": 2, "seat_torque_nm": 1.3},
  "setup_values": {"jar_x_cm": -1.4, "thread_turns": 2, "seat_torque_nm": 1.3}
}
```

The JSON is a regenerable projection of `(instance_id, eval_seed, epoch)` at a
pinned KitchenBench version. The single source of truth remains in the task and
simulation annotations in code. Do not hand-edit scene layouts. Generate them
again from the same coordinates when needed.

## Rig format

Rigs are hand-authored. They describe arm bases and optional reach, frame
documentation, and bench dimensions. The loader enforces the left-at-negative-x
and right-at-positive-x convention.

```json
{
  "layout_version": 1,
  "rig_id": "yam-bimanual",
  "frame": {
    "origin": "midpoint between the two arm bases, projected to table height",
    "axes": {"+x": "robot right", "+y": "away from the arms"},
    "units": {"length": "cm", "angle": "deg"}
  },
  "arms": {
    "left": {"base_xy_cm": [-30.0, -20.0], "faces": "+y", "reach_cm": 65},
    "right": {"base_xy_cm": [30.0, -20.0], "faces": "+y", "reach_cm": 65}
  },
  "bench": {"size_cm": [120, 60]}
}
```

The packaged schemas are `schemas/scene-layout.schema.json` and
`schemas/rig.schema.json` within the `kitchenbench` installation.

## Command line usage

Write canonical JSON to standard output:

```console
kitchenbench-layout seal_container/twist-lid-jar --eval-seed 0 --epoch 0
```

Write the same representation to a file:

```console
kitchenbench-layout seal_container/twist-lid-jar --eval-seed 0 --epoch 0 -o layout.json
```
