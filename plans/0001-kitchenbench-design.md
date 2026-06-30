# KitchenBench — design

> **KitchenBench** is a bimanual kitchen-manipulation benchmark for VLA models,
> built on [RoboInspect](https://github.com/robocurve/roboinspect). It is a *standalone
> plugin repo* (the first member of [WorldEvals](https://github.com/robocurve/worldevals),
> the "Inspect Evals for robotics"). It ships **10 tasks** as RoboInspect `Task`s,
> registered via entry points, plus a dependency-free mock kitchen so the whole
> thing runs in CI — and is ready to point at real **YAM arms + MolmoAct2**.

## Goals / non-goals

- **Goal:** a clean, reproducible set of kitchen `Task`s (scenes + scorers),
  embodiment-agnostic, that a researcher can run with `roboinspect run --task
  kitchenbench/place_cutlery --policy <vla> --embodiment <robot|sim>`.
- **Goal:** runnable *today* with a mock embodiment (CI, 100% coverage) and a
  scripted policy, serving as the template for a real YAM-arms embodiment.
- **Goal:** mirror the things RoboInspect already established — `Task = scenes +
  scorer`, registry/entry-points, immutable `EvalLog`. Mirror Inspect Evals'
  conventions: co-located per-task metadata, a registry that re-exports every
  task (with a test enforcing reachability), and a README task table.
- **Non-goal:** a physics simulator. The mock is abstract (like RoboInspect's
  `CubePick`); the *value* is the task definitions, which are real and usable on
  hardware. The real YAM-arms embodiment + MolmoAct2 policy live elsewhere.

## The 10 tasks

Each task is a RoboInspect `@task` factory whose scenes are the Cartesian product of
its variation axes. `bimanual=True` marks tasks that genuinely need two arms.

| key | instruction template | axes | target kind | bimanual | category |
|-----|----------------------|------|-------------|----------|----------|
| `place_cutlery` | place the {cutlery} on the {dishware} | cutlery∈{spoon,fork,knife} × dishware∈{plate,bowl,napkin} | `place_on` | no | pick_place |
| `stack` | stack the {items} | items∈{cups,bowls,plates} | `stack` | no | stacking |
| `place_in_rack` | place the {dishware} into the dish rack | dishware∈{plate,bowl,cup} | `place_in` | no | insertion |
| `pour_pasta` | pour the dry pasta into the {vessel} | vessel∈{bowl,cup,pot} | `pour_into` | yes | granular |
| `open_container` | open the {container} | container∈{jar,bottle,food container} | `open` | yes | articulated |
| `fold_cloth` | fold the {cloth} | cloth∈{dish towel,napkin,cloth} | `fold` | yes | deformable |
| `seal_container` | seal the {container} with its lid | container∈{food container,pot,jar} | `seal` | yes | mating |
| `handoff` | hand off the {item} from one arm to the other | item∈{utensil,cup,produce item} | `handoff` | yes | coordination |
| `sort_cutlery` | sort the cutlery into the correct tray compartments | layout∈{a,b,c} (seeds) | `sort` | no | classification |
| `scoop_pasta` | scoop the {pasta} with the {tool} and transfer it to the container | pasta∈{penne,rigatoni} × tool∈{spoon,measuring cup} | `scoop_transfer` | yes | granular_tool |

37 scenes total. Each `Scene`: `id` (key + axis values), `instruction`
(template filled), `target` (`Target(kind, spec=axis-values + extras)`),
`init_seed`, `metadata` (axes, bimanual, category, task version).

Rationale for the set (from the user): coverage of pick-place, stacking,
slotted insertion, part-mating, granular pour vs. tool-mediated scoop,
articulated lid removal, deformable folding, a pure two-arm handover (the "must
use both arms" anchor), and a multi-instance classification sort (tests
consistency, not a single lucky success).

## Architecture

```
kitchenbench/
  pyproject.toml            # depends on roboinspect (git dep until PyPI); entry points
  src/kitchenbench/
    __init__.py             # re-exports all 10 task factories (reachability)
    py.typed
    specs.py                # TaskSpec dataclass + the declarative SPECS list (the 10)
    tasks.py                # @task factories generated from SPECS + scene builder
    scoring.py              # task_success() scorer (privileged OR operator verdict)
    embodiment.py           # KitchenEmbodiment — abstract bimanual mock world
    policies.py             # ScriptedKitchenPolicy / Random / Noop (chunk-aware)
    _registry.py            # imports tasks so entry points resolve; reachability list
  tests/                    # 100% coverage, pytest
  README.md                 # task table + usage (mock now, YAM arms tomorrow)
```

### TaskSpec (declarative — single source of truth)

```python
@dataclass(frozen=True)
class TaskSpec:
    key: str
    title: str
    instruction: str           # str.format template over the axes
    axes: dict[str, tuple[str, ...]]
    target_kind: str
    bimanual: bool
    category: str
    max_steps: int
    version: str = "1"
    description: str = ""
```

`tasks.py` turns each spec into a registered `@task` factory named
`kitchenbench/<key>` (slashed namespace so WorldEvals can host many benchmarks
without key collisions). The factory builds the scene dataset via the axis
product. A `make_task(spec)` helper + a loop register all 10; `__init__` and
`_registry` import them so `roboinspect list` / entry points see every task. A test
asserts the entry-point set == the SPECS set (Inspect-style reachability guard).

### Scoring

Real kitchens have no privileged success oracle, so the default scorer must work
both for the mock (privileged) and for hardware (operator/learned):

```python
def task_success() -> Scorer:
    # success iff the trial terminated with reason "success"
    # (mock/sim privileged signal, OR a real embodiment that reports operator-
    # confirmed success) OR an affirmative operator verdict was recorded.
```

Also expose RoboInspect's `operator_scorer` and `episode_length` for convenience.
Each task uses `[task_success(), episode_length()]`.

### Mock embodiment (`KitchenEmbodiment`)

Abstract, deterministic, dependency-free (NumPy only) — the `CubePick` of
kitchens. It does **not** simulate physics; it models *progress toward the scene
goal* so the pipeline (scenes → chunked rollout → score → log) is exercised:

- **Action space:** `Box(shape=(8,))` = two arms × `[dx, dy, dz, gripper]`,
  `ActionSemantics(control_mode="eef_delta_pos")`. Bimanual shape on purpose, so
  a real YAM embodiment is a natural superset.
- **Observation:** `state = {"progress": [p], "left_eef": ..., "right_eef": ...}`,
  a tiny rendered top-down `images["overhead"]`, plus the scene instruction.
- **reset(scene, seed):** progress←0; store the target; seed an RNG.
- **step(action):** progress += clipped contribution of action magnitude;
  `terminated` with reason `"success"` and `info["success"]=True` once progress
  ≥ threshold; exposes `info["progress"]`. Declares `PRIVILEGED_SUCCESS`,
  `SEEDABLE`, `RENDERABLE`; `supported_target_kinds` left empty (runs any task).

### Policies

- `ScriptedKitchenPolicy` — reads `progress`, emits a full-magnitude action
  chunk toward completion → deterministic success (the CI oracle / template).
- `RandomKitchenPolicy` — small random actions; usually fails within `max_steps`.
- `NoopKitchenPolicy` — zeros; never succeeds.

All three emit `ActionChunk`s (H>1) to exercise open-loop execution.

### Real run tomorrow (YAM arms + MolmoAct2)

KitchenBench provides only the embodiment-agnostic `Task`s. The real run pairs
them with a YAM-arms `Embodiment` (16-DoF bimanual: two 7-DoF arms + 2 grippers,
multi-camera, operator-confirmed success) and a MolmoAct2 `Policy` — both live in
their own adapter packages (e.g. `robocurve/embodiments`). Compatibility is
checked at `eval()` time; instructions feed the VLA verbatim. README documents
the exact command and the `Embodiment`/`Policy` contract to implement.

## Quality bar (same as RoboInspect)

- `pyproject` (hatchling + hatch-vcs); MIT; `py.typed`.
- `ruff` + `ruff format` + `mypy --strict` + `pytest`.
- **100% coverage** (`--cov-fail-under=100`), pre-commit hooks (ruff/mypy on
  commit, coverage on push via `uv run`), GitHub Actions matrix (Linux+macOS ×
  py3.11/3.12 blocking; extras allow-failure), branch protection on PRs.
- Reachability test: every entry-point task is importable and registered.
- Determinism test: scripted policy succeeds on all 37 scenes; random/noop don't
  all succeed.

## Milestones (commit/push each)

- **M0** repo skeleton, packaging (roboinspect git dep), CI, pre-commit, README stub.
- **M1** `specs` + `tasks` (10 factories, scene generation) + reachability test.
- **M2** `KitchenEmbodiment` + policies + `scoring` + integration tests (eval all
  10 with scripted → success) → 100% coverage.
- **M3** entry points wired; `roboinspect list`/`run` works; README task table +
  usage + YAM/MolmoAct2 instructions.
- **M4** branch protection, push, register in WorldEvals.
