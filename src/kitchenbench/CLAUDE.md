# `kitchenbench` package — module map

A Inspect Robots plugin. Importing the package registers all 10 tasks, the mock
`kitchen` embodiment, and the mock policies with the Inspect Robots registry; entry
points (in `pyproject.toml`) make them discoverable without importing.

## Modules

| Module | Responsibility |
|--------|----------------|
| `distributions.py` | Pure-numpy `Distribution` types — `Uniform`, `Categorical` (samples **by index** to preserve dtype), `Normal`, `Constant`. Each `sample(rng)` returns a **builtin** scalar (JSON-native); `describe()` renders methodology notation. |
| `instances.py` | `TaskInstance` (stochastic setup of named distributions + goal), `Realization` (one sampled environment), `Validation` (per-expert ratings + accept rule), the `K_*` constants, and the sim annotation model: `Var`/`SimObject`/`SimSpec` + `_validate_sim_spec` (import-time coverage invariant: every setup var referenced, at most once among object bindings; parents declared-before-use). |
| `specs.py` | `TaskSpec` + `SPECS` — the **single source of truth**: 10 tasks, each with exactly `K_INSTANCES` (5) distribution-based `TaskInstance`s. |
| `tasks.py` | `build_scenes` (one `Scene` per instance), `make_task` (`Epochs(count=5, reducer="mean")`), `realize_scene(scene, seed)` (the run-time seam that recovers + realizes an instance), and the 10 `@task` factories. `TASK_FACTORIES` maps key → factory. |
| `scoring.py` | `task_success()` — success iff `termination_reason == "success"` (mock/sim privileged signal, or a real embodiment reporting operator-confirmed success) **or** an affirmative recorded operator verdict. |
| `embodiment.py` | `KitchenEmbodiment` — dependency-free abstract bimanual mock. Models *progress toward the scene goal* (not physics), like Inspect Robots's `CubePick`. Action space is `(8,)` = `[left dx,dy,dz, right dx,dy,dz, left gripper, right gripper]` (see `dim_labels`; arm translations first, grippers last). On reset of a KitchenBench scene (marked `metadata["benchmark"] == "kitchenbench"`) it calls `realize_scene` so the observed instruction reflects the per-epoch realization (a real embodiment would also arrange the sampled setup); the instruction is carried on every step observation since real VLA policies re-condition on it each `act()`. |
| `policies.py` | `ScriptedKitchenPolicy` (reads privileged `goal_dir`, succeeds), `RandomKitchenPolicy`, `NoopKitchenPolicy`. All emit `ActionChunk`s. |
| `sim/` | The sim contract (plan 0003): `blueprint.py` resolves an instance's `SimSpec` annotation into a `SceneBlueprint` (spawn list, roles, success params; `ASSETS` nominal-dims catalog; `SIM_CONTRACT_VERSION`); `success.py` defines the per-target-kind quantitative success checkers over the 4-query `WorldState` protocol (never raise on missing objects; fold/handoff capture state). Annotations live on each `TaskInstance` in `specs.py` and are validated at import (coverage invariant). |
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
  `setup_spec()` strings; the live `TaskInstance` is recovered by looking up
  `metadata["instance_id"]` within `SPEC_BY_KEY[task].instances`, never stored —
  id-based so replays of old logs fail loudly instead of silently realizing a
  reordered instance). `metadata["benchmark"] == "kitchenbench"` is the marker
  the mock embodiment keys on to decide a scene is realize-able.
- `Categorical.sample` samples **by index** (never `rng.choice(values)`, which
  coerces numeric tuples to strings). `derive_seed` comes from `inspect_robots.rollout`
  (not re-exported at the top level).

## Adding a task

1. Author 5 `TaskInstance`s (distribution-based) and a `TaskSpec(...)`; append to
   `SPECS` in `specs.py`. Every `{placeholder}` in a goal must be in
   `language_vars` (and in `setup`).
2. Add an `@task("kitchenbench/<key>")` factory in `tasks.py`, add it to
   `TASK_FACTORIES` + the `__init__` re-export.
3. Add the entry point to `pyproject.toml` under
   `[project.entry-points."inspect_robots.tasks"]`.
4. Add the task key to KitchenBench's entry in WorldEvals' `catalog.py`.

`tests/test_specs.py` / `test_realize_all.py` parametrize over `SPECS` and realize
all 50 instances, so a new spec is exercised automatically — keep coverage at 100%.
