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

| Task (`--task`) | Goal | Bimanual | Category |
|---|---|:--:|---|
| `kitchenbench/place_cutlery` | place the {cutlery} on the {dishware} | | pick-place |
| `kitchenbench/stack` | stack the {items} | | stacking |
| `kitchenbench/place_in_rack` | place the {dishware} into the dish rack | | insertion |
| `kitchenbench/pour_pasta` | pour the dry pasta into the {vessel} | ✅ | granular |
| `kitchenbench/open_container` | open the {container} | ✅ | articulated |
| `kitchenbench/fold_cloth` | fold the {cloth} | ✅ | deformable |
| `kitchenbench/seal_container` | seal the {container} with its lid | ✅ | mating |
| `kitchenbench/handoff` | hand off the {item} from one arm to the other | ✅ | coordination |
| `kitchenbench/sort_cutlery` | sort the cutlery into the correct tray compartments | | classification |
| `kitchenbench/scoop_pasta` | scoop the {pasta} with the {tool} and transfer it to the container | ✅ | granular+tool |

## Task instances & realizations

KitchenBench follows the
[physical-automation methodology](https://github.com/jeqcho/physical-automation-methodology-docs):
each task is a set of **task instances**, and each instance is a *stochastic
environment specification* — named random variables with **distributions** — plus
a goal. Defaults match the methodology's recommendations:

- **5 instances per task** (`K_INSTANCES`) → one RoboLens `Scene` each.
- **5 realizations per instance** (`K_REALIZATIONS`) → run via
  `Epochs(count=5, reducer="mean")`. Each realization samples the instance's
  random variables (seeded per epoch) into one concrete environment.

```python
from kitchenbench import SPEC_BY_KEY, realize_scene
inst = SPEC_BY_KEY["pour_pasta"].instances[0]
inst.setup_spec()          # {'fill_g': 'Uniform[80, 200]', 'vessel': 'Categorical({bowl, cup, pot})', ...}
inst.realize(seed=0).instruction   # a concrete goal, e.g. "pour the dry pasta into the cup"
```

The mean reducer over 5 realizations makes the **per-scene reduced `task_success`
the instance success probability P̂[Yᵢ=1]** — exactly the methodology's estimator.
On real hardware, an embodiment/operator calls `realize_scene(scene, seed)` to get
the concrete setup to arrange (`Realization.setup_lines`).

Distribution types: `Uniform[a,b]`, `Categorical({…})`, `N(μ,σ²)`, `Constant`.

> **Validation status.** The shipped instances are AI-authored drafts
> (`Validation(source="opus-draft")`, `validated=False`). The methodology's
> `K_i = 5` is *after* human validation (3 experts, representativeness & quality
> ≥ 4); run that commissioning pipeline before trusting the instances. We do **not**
> fabricate ratings.

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
# Per-instance success probability P̂[Yᵢ=1] lives in each sample's reduced score:
for s in log.samples:
    print(s.scene_id, s.reduced["task_success"])
# log.results.metrics["task_success"] is the mean of P̂ over instances — a
# convenience aggregate, NOT a methodology quantity (the methodology sorts P̂ into
# quantiles and fits pTQ / automation-halvings; out of scope here).
```

The mock is abstract (it models *progress toward the scene goal*, like RoboLens's
`CubePick`) — its job is to exercise the pipeline and give you a template. **In the
mock, success depends only on the seeded goal direction, so the sampled setup
distributions have no causal effect** (P̂ is degenerately 1.0 for the scripted
oracle); the distribution *content* only bites on a real embodiment. The **value is
the task definitions**, which run unchanged on a real robot.

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
