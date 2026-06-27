# `kitchenbench` package — module map

A RoboLens plugin. Importing the package registers all 10 tasks, the mock
`kitchen` embodiment, and the mock policies with the RoboLens registry; entry
points (in `pyproject.toml`) make them discoverable without importing.

## Modules

| Module | Responsibility |
|--------|----------------|
| `specs.py` | `TaskSpec` dataclass + `SPECS` — the **declarative single source of truth** for the 10 tasks (key, instruction template, variation `axes`, target kind, bimanual, category, max_steps). |
| `tasks.py` | `build_scenes`/`make_task` (axis product → `Scene`s) and the 10 `@task`-decorated factories registered under `kitchenbench/<key>`. `TASK_FACTORIES` maps key → factory. |
| `scoring.py` | `task_success()` — success iff `termination_reason == "success"` (mock/sim privileged signal, or a real embodiment reporting operator-confirmed success) **or** an affirmative recorded operator verdict. |
| `embodiment.py` | `KitchenEmbodiment` — dependency-free abstract bimanual mock. Models *progress toward the scene goal* (not physics), like RoboLens's `CubePick`. Action space is `(8,)` = two arms × `[dx,dy,dz,gripper]`. |
| `policies.py` | `ScriptedKitchenPolicy` (reads privileged `goal_dir`, succeeds), `RandomKitchenPolicy`, `NoopKitchenPolicy`. All emit `ActionChunk`s. |
| `__init__.py` | Re-exports the factories, mock, policies, specs (the package's public surface). |

## Key invariants

- The entry-point name, the `@task(name=...)`, and the returned `Task.name` are
  **all identical** (`kitchenbench/<key>`). Keep them in sync.
- The mock's hidden unit `goal_dir` + alignment threshold (0.99) is what makes the
  scripted oracle succeed and random/no-op fail **deterministically** — don't
  loosen the threshold or random policies may start passing and break tests.
- Compatibility is exact action-dim equality: the mock (dim 8) pairs with the
  mock scripted policy (dim 8); a real YAM arm (higher DoF) pairs with a real VLA
  — there is no cross-pairing. Tasks carry no action space, so they run on both.

## Adding a task

1. Append a `TaskSpec(...)` to `SPECS` in `specs.py`.
2. Add an `@task("kitchenbench/<key>")` factory in `tasks.py` (call
   `make_task(SPEC_BY_KEY["<key>"])`) and add it to `TASK_FACTORIES` + the
   `__init__` re-export.
3. Add the entry point to `pyproject.toml` under
   `[project.entry-points."robolens.tasks"]`.
4. Add the task key to KitchenBench's entry in WorldEvals' `catalog.py`.

The tests in `tests/test_tasks.py` parametrize over `SPECS`, so a new spec is
exercised automatically — but keep coverage at 100% (any task-specific branch
needs a test).
