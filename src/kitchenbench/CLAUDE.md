# `kitchenbench` package — module map

A RoboLens plugin. Importing the package registers all 10 tasks, the mock
`kitchen` embodiment, and the mock policies with the RoboLens registry; entry
points (in `pyproject.toml`) make them discoverable without importing.

## Modules

| Module | Responsibility |
|--------|----------------|
| `distributions.py` | Pure-numpy `Distribution` types — `Uniform`, `Categorical` (samples **by index** to preserve dtype), `Normal`, `Constant`. Each `sample(rng)` returns a **builtin** scalar (JSON-native); `describe()` renders methodology notation. |
| `instances.py` | `TaskInstance` (stochastic setup of named distributions + goal), `Realization` (one sampled environment), `Validation` (per-expert ratings + accept rule), and the `K_REALIZATIONS`/`K_INSTANCES`/`K_EXPERTS` constants. `TaskInstance.realize(seed)` samples; `setup_spec()` describes. |
| `specs.py` | `TaskSpec` + `SPECS` — the **single source of truth**: 10 tasks, each with exactly `K_INSTANCES` (5) distribution-based `TaskInstance`s. |
| `tasks.py` | `build_scenes` (one `Scene` per instance), `make_task` (`Epochs(count=5, reducer="mean")`), `realize_scene(scene, seed)` (the run-time seam that recovers + realizes an instance), and the 10 `@task` factories. `TASK_FACTORIES` maps key → factory. |
| `scoring.py` | `task_success()` — success iff `termination_reason == "success"` (mock/sim privileged signal, or a real embodiment reporting operator-confirmed success) **or** an affirmative recorded operator verdict. |
| `embodiment.py` | `KitchenEmbodiment` — dependency-free abstract bimanual mock. Models *progress toward the scene goal* (not physics), like RoboLens's `CubePick`. Action space is `(8,)` = two arms × `[dx,dy,dz,gripper]`. On reset of a KitchenBench scene it calls `realize_scene` so the observed instruction reflects the per-epoch realization (a real embodiment would also arrange the sampled setup). |
| `policies.py` | `ScriptedKitchenPolicy` (reads privileged `goal_dir`, succeeds), `RandomKitchenPolicy`, `NoopKitchenPolicy`. All emit `ActionChunk`s. |
| `__init__.py` | Re-exports the public surface: factories, mock, policies, specs, distributions, `TaskInstance`/`Realization`/`Validation`, `realize_scene`, and the `K_*` constants (fenced by `__all__`). |

## Key invariants

- The entry-point name, the `@task(name=...)`, and the returned `Task.name` are
  **all identical** (`kitchenbench/<key>`). Keep them in sync.
- The mock's hidden unit `goal_dir` + alignment threshold (0.99) is what makes the
  scripted oracle succeed and random/no-op fail **deterministically** — don't
  loosen the threshold or random policies may start passing and break tests.
- Compatibility is exact action-dim equality: the mock (dim 8) pairs with the
  mock scripted policy (dim 8); a real YAM arm (higher DoF) pairs with a real VLA
  — there is no cross-pairing. Tasks carry no action space, so they run on both.

## Methodology mapping (do not break)

- **task instance → `Scene`**, **realization → epoch**, `Epochs(count=5,
  reducer="mean")` → per-scene reduced `task_success` = **P̂[Yᵢ=1]**.
- `metadata["task_success"]` at the eval level is the *mean of P̂ over instances* —
  a **convenience aggregate, not** a methodology quantity (the methodology sorts P̂
  into quantiles → pTQ / automation-halvings; out of scope here).
- Scene `metadata` is **strictly JSON-native** (distributions stored as
  `setup_spec()` strings; the live `TaskInstance` is recovered via
  `SPEC_BY_KEY[task].instances[index]`, never stored).
- `Categorical.sample` samples **by index** (never `rng.choice(values)`, which
  coerces numeric tuples to strings). `derive_seed` comes from `robolens.rollout`
  (not re-exported at the top level).

## Adding a task

1. Author 5 `TaskInstance`s (distribution-based) and a `TaskSpec(...)`; append to
   `SPECS` in `specs.py`. Every `{placeholder}` in a goal must be in
   `language_vars` (and in `setup`).
2. Add an `@task("kitchenbench/<key>")` factory in `tasks.py`, add it to
   `TASK_FACTORIES` + the `__init__` re-export.
3. Add the entry point to `pyproject.toml` under
   `[project.entry-points."robolens.tasks"]`.
4. Add the task key to KitchenBench's entry in WorldEvals' `catalog.py`.

`tests/test_specs.py` / `test_realize_all.py` parametrize over `SPECS` and realize
all 50 instances, so a new spec is exercised automatically — keep coverage at 100%.
