<div align="center">

# 🍳 KitchenBench

**A bimanual kitchen-manipulation benchmark for VLA models.**

Built on [RoboLens](https://github.com/robocurve/robolens) · part of
[WorldEvals](https://github.com/robocurve/worldevals), the "Inspect Evals for robotics".

[![CI](https://github.com/robocurve/kitchenbench/actions/workflows/ci.yml/badge.svg)](https://github.com/robocurve/kitchenbench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/robocurve/kitchenbench/actions/workflows/ci.yml)
[![Built on RoboLens](https://img.shields.io/badge/built%20on-RoboLens-indigo)](https://github.com/robocurve/robolens)

</div>

KitchenBench is **10 kitchen-manipulation tasks** expressed as RoboLens `Task`s —
embodiment-agnostic, so you run them against *any* compatible policy/embodiment.
The set emphasizes **bimanual coordination**: pouring, lid removal, folding,
part-mating, a pure two-arm handover, and tool-mediated scooping, alongside
classic pick-place / stacking / slotted insertion and a multi-instance sort.

It ships a **dependency-free mock kitchen** so the whole suite runs in CI, and is
designed to point straight at real hardware — e.g. **YAM bimanual arms** driven by
**MolmoAct2**.

## The tasks

| Task (`--task`) | Instruction | Variations | Bimanual | Category |
|---|---|---|:--:|---|
| `kitchenbench/place_cutlery` | place the {cutlery} on the {dishware} | spoon/fork/knife × plate/bowl/napkin | | pick-place |
| `kitchenbench/stack` | stack the {items} | cups/bowls/plates | | stacking |
| `kitchenbench/place_in_rack` | place the {dishware} into the dish rack | plate/bowl/cup | | insertion |
| `kitchenbench/pour_pasta` | pour the dry pasta into the {vessel} | bowl/cup/pot | ✅ | granular |
| `kitchenbench/open_container` | open the {container} | jar/bottle/food container | ✅ | articulated |
| `kitchenbench/fold_cloth` | fold the {cloth} | dish towel/napkin/cloth | ✅ | deformable |
| `kitchenbench/seal_container` | seal the {container} with its lid | food container/pot/jar | ✅ | mating |
| `kitchenbench/handoff` | hand off the {item} from one arm to the other | utensil/cup/produce item | ✅ | coordination |
| `kitchenbench/sort_cutlery` | sort the cutlery into the correct tray compartments | 3 pile layouts | | classification |
| `kitchenbench/scoop_pasta` | scoop the {pasta} with the {tool} and transfer it to the container | penne/rigatoni × spoon/measuring cup | ✅ | granular+tool |

Each task expands its variation axes into one `Scene` per combination (37 scenes
total), each with a filled-in language instruction and a success `Target`.

## Install

```bash
# RoboLens isn't on PyPI yet, so install both from GitHub (uv recommended):
uv pip install "robolens @ git+https://github.com/robocurve/robolens@v0.1.0"
uv pip install "kitchenbench @ git+https://github.com/robocurve/kitchenbench"
```

## Run it (mock kitchen, no hardware)

KitchenBench registers a dependency-free mock embodiment (`kitchen`) and policies
(`kitchen_scripted` / `kitchen_random` / `kitchen_noop`) via entry points:

```bash
robolens list tasks                       # see all kitchenbench/* tasks
robolens run --task kitchenbench/pour_pasta --policy kitchen_scripted --embodiment kitchen
```

Or in Python:

```python
from robolens import eval

(log,) = eval("kitchenbench/open_container", "kitchen_scripted", "kitchen")
print(log.status, log.results.metrics)    # success {'task_success': 1.0, 'episode_length': ...}
```

The mock is abstract (it models *progress toward the scene goal*, like RoboLens's
`CubePick`) — its job is to exercise the pipeline and give you a template. The
**value is the task definitions**, which run unchanged on a real robot.

## Run it on real hardware (YAM arms + MolmoAct2)

KitchenBench tasks are embodiment-agnostic. To evaluate on real **YAM bimanual
arms** with **MolmoAct2**, provide two RoboLens components (e.g. in your own
adapter package such as `robocurve/embodiments`):

- a **`Policy`** wrapping MolmoAct2: `act(observation) -> ActionChunk` (the
  scene's `instruction` is fed to the VLA verbatim);
- an **`Embodiment`** for the YAM arms: `reset`/`step`/`close`, declaring its
  action space (e.g. two 7-DoF arms + grippers) and cameras. Because there is no
  privileged success oracle, the embodiment should turn the **operator's
  confirmation** at episode end into `StepResult(terminated=True,
  termination_reason="success")` (or set `record.operator_judgement`) —
  KitchenBench's `task_success` scorer reads either. Declare the `"self_paced"`
  capability and pace the control loop inside `step()`.

```bash
robolens run --task kitchenbench/pour_pasta --policy molmoact2 --embodiment yam_arms
```

RoboLens checks `(policy, embodiment)` compatibility (action dims, semantics,
camera/state keys) before any motion and writes an immutable `EvalLog`.

## Development

```bash
uv venv && uv pip install -e ".[dev]"     # robolens resolved from the v0.1.0 tag
uv run pre-commit install
uv run pytest --cov                        # 100% coverage required
uv run ruff check . && uv run mypy
```

## License

[MIT](LICENSE)
