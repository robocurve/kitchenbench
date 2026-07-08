<div align="center">

# 🍳 KitchenBench

**A bimanual kitchen-manipulation benchmark for VLA models.**

Built on [Inspect Robots](https://github.com/robocurve/inspect-robots) · part of
[WorldEvals](https://github.com/robocurve/worldevals), the "Inspect Evals for robotics".

![Status: alpha](https://img.shields.io/badge/status-alpha-blue)
[![CI](https://github.com/robocurve/kitchenbench/actions/workflows/ci.yml/badge.svg)](https://github.com/robocurve/kitchenbench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/robocurve/kitchenbench/actions/workflows/ci.yml)
[![Built on Inspect Robots](https://img.shields.io/badge/built%20on-Inspect%20Robots-indigo)](https://github.com/robocurve/inspect-robots)

</div>

> [!NOTE]
> This project is in early development. The API may change between releases, so pin a version before depending on it.

KitchenBench is **10 kitchen-manipulation tasks** expressed as Inspect Robots `Task`s —
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
| `kitchenbench/stack` | stack the cups / bowls / plates | | stacking |
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
[physical-automation methodology](https://github.com/jeqcho/physical-automation-methodology-docs).
The key ideas, top-down:

- A **task** (e.g. `pour_pasta`) is a set of **task instances**.
- A **task instance** is one concrete *scenario written as a distribution*: a
  **stochastic setup** (named random variables, each with a distribution) plus a
  **goal** (a natural-language success criterion that may reference the sampled
  variables). It is *not* a single fixed scene — it is a recipe for generating many.
- A **realization** is one sample of that recipe: draw every random variable from
  its distribution to get one concrete environment (and a concrete goal sentence).
- Running a `(policy, embodiment)` pair on `K_realizations` realizations and
  averaging the binary successes estimates the **instance success probability**
  P̂[Yᵢ = 1].

```
task  pour_pasta
 ├─ instance 1  (a distribution)  ──realize──▶  5 concrete environments  ──▶  P̂₁
 ├─ instance 2  (a distribution)  ──realize──▶  5 concrete environments  ──▶  P̂₂
 │  … 5 instances total …
 └─ instance 5  (a distribution)  ──realize──▶  5 concrete environments  ──▶  P̂₅
```

KitchenBench uses the methodology's recommended defaults: **5 instances per task**
(`K_INSTANCES`) and **5 realizations per instance** (`K_REALIZATIONS`).

### A worked example

This is one of `pour_pasta`'s five instances (from
[`specs.py`](src/kitchenbench/specs.py), lightly reformatted):

```python
TaskInstance(
    instance_id="pour_pasta/measuring-cup-to-bowl",
    goal="pour the dry pasta into the {vessel}",       # {vessel} is sampled
    setup={
        "vessel":         Categorical(("bowl", "cup", "pot")),
        "fill_g":         Uniform(80, 200),            # grams of pasta
        "pour_height_cm": Uniform(8, 15),
        "vessel_x_cm":    Normal(0.0, 3.0),            # placement jitter (cm)
        "vessel_y_cm":    Normal(0.0, 3.0),
    },
    language_vars=("vessel",),
    target_kind="pour_into",
    static={"substance": "dry_pasta"},
)
```

**Realizing it** with different seeds samples those distributions into concrete
environments an operator can physically arrange (and a goal to give the VLA)
(numbers rounded here for readability):

```
realize(seed=0)                          realize(seed=2)
  Goal: pour the dry pasta into the bowl   Goal: pour the dry pasta into the cup
  Setup:                                    Setup:
    vessel        = bowl                      vessel        = cup
    fill_g        = 156                        fill_g        = 111
    pour_height_cm = 9.9                       pour_height_cm = 10.1
    vessel_x_cm   = +0.3                        vessel_x_cm   = -7.3
    vessel_y_cm   = -1.6                        vessel_y_cm   = +5.4
```

Inspect and realize instances from Python:

```python
from kitchenbench import SPEC_BY_KEY

inst = SPEC_BY_KEY["pour_pasta"].instances[0]
inst.setup_spec()
# {'fill_g': 'Uniform[80, 200]', 'pour_height_cm': 'Uniform[8, 15]',
#  'vessel': 'Categorical({bowl, cup, pot})', 'vessel_x_cm': 'N(0, 3²)', ...}

r = inst.realize(seed=0)
r.instruction     # 'pour the dry pasta into the bowl'
r.values          # {'vessel': 'bowl', 'fill_g': 156.43…, 'pour_height_cm': 9.88…, …}  (JSON-native)
r.setup_lines     # ('fill_g = 156.43…', 'pour_height_cm = 9.88…', 'vessel = bowl', …)
```

### How it maps to a run

Each instance becomes one Inspect Robots `Scene`; the 5 realizations are the 5 **epochs**
(`Epochs(count=5, reducer="mean")`), each seeded independently. Because the reducer
is the mean, **each scene's reduced `task_success` is the instance success
probability P̂[Yᵢ = 1]** — exactly the methodology's estimator:

```python
from inspect_robots import eval
(log,) = eval("kitchenbench/pour_pasta", "kitchen_scripted", "kitchen")
for s in log.samples:
    print(s.scene_id, s.reduced["task_success"])   # one P̂ per instance
```

On real hardware an embodiment (or operator tool) calls `realize_scene(scene,
seed)` to get the concrete setup to arrange — `Realization.setup_lines` is the
"arrange this" checklist, and `Realization.instruction` is the goal fed to the VLA.

**Distribution types** (in [`distributions.py`](src/kitchenbench/distributions.py)):
`Uniform(a, b)` continuous · `Categorical((…), weights=None)` over a finite set ·
`Normal(μ, σ)` Gaussian · `Constant(v)` fixed. Every sample is a builtin
`float`/`int`/`str` (JSON-native), and `Categorical` preserves value types (an `int`
category samples back as an `int`).

> **Validation status — read before trusting the numbers.** The shipped instances
> are **AI-authored drafts** (`Validation(source="opus-draft")`, `validated=False`).
> The methodology's `K_i = 5` is the count *after* human validation — 3 experts
> rating each instance on representativeness **and** quality, accepted only if both
> are ≥ 4. Run that commissioning pipeline before relying on the instances; we do
> **not** fabricate ratings. Also note `eval()`'s task-level
> `metrics["task_success"]` is the *mean of P̂ over instances* — a convenience
> aggregate, **not** a methodology output (the methodology sorts the per-instance P̂
> into quantiles and fits the pTQ / automation-halvings curves).

## Install

```bash
# Inspect Robots isn't on PyPI yet, so install both from GitHub (uv recommended):
uv pip install "inspect-robots @ git+https://github.com/robocurve/inspect-robots@v0.3.0"
uv pip install "kitchenbench @ git+https://github.com/robocurve/kitchenbench"
```

## Run it (mock kitchen, no hardware)

KitchenBench registers a dependency-free mock embodiment (`kitchen`) and policies
(`kitchen_scripted` / `kitchen_random` / `kitchen_noop`) via entry points:

```bash
inspect-robots list tasks                       # see all kitchenbench/* tasks
inspect-robots run --task kitchenbench/pour_pasta --policy kitchen_scripted --embodiment kitchen
```

Or in Python:

```python
from inspect_robots import eval

(log,) = eval("kitchenbench/open_container", "kitchen_scripted", "kitchen")
# Per-instance success probability P̂[Yᵢ=1] lives in each sample's reduced score:
for s in log.samples:
    print(s.scene_id, s.reduced["task_success"])
# log.results.metrics["task_success"] is the mean of P̂ over instances — a
# convenience aggregate, NOT a methodology quantity (the methodology sorts P̂ into
# quantiles and fits pTQ / automation-halvings; out of scope here).
```

The mock is abstract (it models *progress toward the scene goal*, like Inspect Robots's
`CubePick`) — its job is to exercise the pipeline and give you a template. **In the
mock, success depends only on the seeded goal direction, so the sampled setup
distributions have no causal effect** (P̂ is degenerately 1.0 for the scripted
oracle); the distribution *content* only bites on a real embodiment. The **value is
the task definitions**, which run unchanged on a real robot.

## Run it on real hardware (YAM arms + MolmoAct2)

KitchenBench tasks are embodiment-agnostic. To evaluate on real **YAM bimanual
arms** with **MolmoAct2**, provide two Inspect Robots components (e.g. in your own
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
inspect-robots run --task kitchenbench/pour_pasta --policy molmoact2 --embodiment yam_arms
```

Inspect Robots checks `(policy, embodiment)` compatibility (action dims, semantics,
camera/state keys) before any motion and writes an immutable `EvalLog`.

## Development

> **Dependency changes:** after editing dependencies in `pyproject.toml`, run
> `uv lock` and commit the updated lockfile — CI installs with
> `uv sync --locked` and fails with "the lockfile needs to be updated" if you
> forget. Day-to-day conventions (PR-only `main`, the required `ci-ok` check,
> one-click releases) are documented in [`CLAUDE.md`](CLAUDE.md).

```bash
uv venv && uv pip install -e ".[dev]"     # inspect_robots resolved from the v0.3.0 tag
uv run pre-commit install
uv run pytest --cov                        # 100% coverage required
uv run ruff check . && uv run mypy
```

## Citation

If you use KitchenBench in your research, please cite it:

```bibtex
@software{kitchenbench,
  author  = {Robocurve},
  title   = {KitchenBench: A bimanual kitchen-manipulation benchmark for VLA models},
  year    = {2026},
  url     = {https://github.com/robocurve/kitchenbench},
  version = {0.3.0},
  license = {MIT}
}
```

## License

[MIT](LICENSE)
