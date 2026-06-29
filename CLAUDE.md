# KitchenBench — agent guide

KitchenBench is a **standalone benchmark repo**: 10 bimanual kitchen-manipulation
tasks for VLA models, expressed as [RoboLens](https://github.com/robocurve/robolens)
`Task`s. It is the first member of
[WorldEvals](https://github.com/robocurve/worldevals) (the "Inspect Evals for
robotics"). It is a **RoboLens plugin** — it depends on RoboLens, defines tasks,
and registers them via entry points.

## The one big idea

A benchmark is **embodiment-agnostic**: it defines *what* to evaluate (scenes +
scorers), never *how* the robot is built. The same 10 tasks run against the
dependency-free mock kitchen (for CI) and against real **YAM bimanual arms +
MolmoAct2** — only the `Policy`/`Embodiment` change.

Each task is a set of **task instances** (the
[physical-automation methodology](https://github.com/jeqcho/physical-automation-methodology-docs),
cloned read-only into `reference/`, gitignored). A task instance is a *stochastic
environment spec* (named random variables + **distributions**) + a goal. Defaults
follow the methodology: **5 instances per task** (one `Scene` each) × **5
realizations per instance** (`Epochs(count=5, reducer="mean")`) = 50 scenes. The
mean over 5 realizations makes each scene's reduced `task_success` the instance
success probability **P̂[Yᵢ=1]**. Instances are AI-authored drafts
(`Validation(source="opus-draft")`, `validated=False`) — not yet human-validated.

## Layout

- `src/kitchenbench/` — the package (see `src/kitchenbench/CLAUDE.md` for the
  module map and how to add a task).
- `tests/` — pytest; the mock `KitchenEmbodiment` + scripted policy exercise the
  whole stack with no hardware.
- `plans/0002-task-instances-distributions.md` — the design doc for the
  task-instance/distribution model (read before changing the task set, the
  distributions, or the 5×5 defaults).
- `reference/` — read-only local copy of the methodology PDFs (gitignored).
- `README.md` — the task table + how to run on the mock and on YAM/MolmoAct2.

## Working here (important gotchas)

- **Dependency on RoboLens is a git tag.** `pyproject.toml` declares
  `robolens>=0.1` with `[tool.uv.sources] robolens = { git = ..., tag = "v0.1.0" }`.
  CI uses `uv` (plain pip ignores `tool.uv.sources`). A sibling checkout exists at
  `../robolens`; for local dev against it, override with
  `uv pip install -e ../robolens`.
- **Conda is active in this shell.** `uv pip install` targets the *active* env, so
  a bare `uv pip install -e .` lands in conda base, not `.venv`. Always
  `source .venv/bin/activate && export VIRTUAL_ENV="$PWD/.venv"` first (or use
  `uv run`).
- Dev loop: `uv venv && uv pip install -e ".[dev]"`, `uv run pre-commit install`,
  `uv run pytest --cov`.
- **Gates (all must pass):** `ruff check .`, `ruff format --check .`, `mypy`
  (strict), `pytest --cov` at **100% coverage**. Pre-commit runs ruff+mypy on
  commit and the coverage gate on push (via `uv run`). CI (Linux+macOS ×
  py3.11/3.12) and the 100% gate are **required, blocking PR checks**.
- **Authoring imports come from the top-level `robolens` package** (its public
  API, stable as of v0.1.0): `from robolens import Task, Scene, Target, task,
  ActionChunk, ...`. Don't import from `robolens.<submodule>` unless a symbol
  isn't re-exported (capability flags like `SEEDABLE` live in
  `robolens.embodiment`).

## Out of scope (lives elsewhere)

The real YAM-arms `Embodiment` and the MolmoAct2 `Policy` are *adapters* and live
in their own package (e.g. `robocurve/embodiments`), not here. KitchenBench ships
only the tasks + a mock world. See the README's "Run it on real hardware" section
for the contract those adapters must implement (operator-confirmed success →
`termination_reason="success"`; declare `"self_paced"` and pace `step()`).
