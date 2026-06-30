# 0001 — KitchenBench tasks as distributions (task instances + realizations)

**Status:** design + implementation plan (rev 2 — addresses critique round 1)
**Goal:** adopt the *task-instance* model from the
[physical-automation methodology](https://github.com/jeqcho/physical-automation-methodology-docs):
write each KitchenBench task as a small set of **task instances** — each a
*stochastic environment specification (named random variables + distributions) +
a goal* — and default the number of rollouts to the methodology's
recommendations.

The methodology PDFs are cloned read-only into `reference/` (gitignored).

## 1. The model we are adopting (from the reference)

- A **task** `t` contains multiple **task instances** `i ∈ t`.
- A **task instance** is a *self-contained scenario*: a **stochastic setup**
  (named random variables with explicit distributions) **+ a goal** (natural-language
  success criterion that may reference sampled variables).
- A **realization** of an instance samples its random variables to produce one
  concrete environment.
- **Estimating instance success probability:** run the (policy, embodiment) pair
  on **K_realizations = 5** independent realizations; each yields a binary
  success; average them → P̂[Yᵢ = 1].
- **K_i = 5** validated instances per task (after rejection-sampling validation by
  **K_experts = 3**, accept iff representativeness *and* quality ≥ 4 on a 1–5 scale).
- Distributions used in the worked examples: `Uniform[a, b]` (continuous),
  `Categorical({…})` (uniform over a finite set — numeric or string),
  `N(μ, σ²)` (Gaussian, incl. 2-D centroid jitter as two independent Gaussians),
  and discrete `Uniform({…})` over a finite integer set (≡ `Categorical` of ints).

## 2. Clean mapping onto RoboInspect (no framework changes needed)

| Methodology | RoboInspect | Notes |
|---|---|---|
| task instance `i ∈ t` | one `Scene` | 5 scenes per task (`K_i = 5`) |
| realization `k` | one **epoch** | `derive_seed(eval_seed, scene.init_seed, epoch)` varies each realization deterministically |
| K_realizations = 5 | `Epochs(count=5, reducer="mean")` | the per-scene **reduced** metric *is* P̂[Yᵢ=1] |
| goal | `Scene.instruction` (+ `Target`) | realized per seed (categorical language vars filled in) |
| stochastic setup | `Scene.metadata["setup"]` (**JSON-native strings only**) | distribution `describe()` strings for the log; the live instance is recovered via the `SPEC_BY_KEY[task].instances[index]` lookup, **never stored in metadata** |
| validation (rep./quality) | `Scene.metadata["validation"]` | carried, **not faked** — see §6 |

This is exactly Inspect's `Task = dataset + scorer + epochs/reducer`: scenes are
instances, epochs are realizations, the mean reducer yields the success
probability the methodology defines.

**Verified against source (critique round 1):** `"mean"` is a registered reducer
(`roboinspect/scorer.py`); `eval()` persists `SceneResult.reduced["task_success"] =
mean over epochs` in `EvalLog.samples` (`eval.py`), which *is* P̂[Yᵢ=1]; the **same
`Scene` is reused across epochs** with only `derive_seed(eval_seed, init_seed,
epoch)` varying (`eval.py`/`rollout.py`), so realization-via-seed is correct.
**Caveat:** `EvalLog` does **not** persist `scene.instruction`/`metadata` — only
`scene_id/status/reduced/epochs/error`. So a test must verify realized
instructions via `realize_scene(...)` directly or a custom recording sink (see §8),
**not** by reading the `EvalLog`.

## 3. New / changed modules

```
src/kitchenbench/
  distributions.py   # NEW — pure-numpy Distribution types + sampling + describe()
  instances.py       # NEW — TaskInstance, Realization, Validation; realize(seed)
  specs.py           # REWRITE — each TaskSpec carries 5 TaskInstances (distributions)
  tasks.py           # CHANGE — build_scenes from instances; Epochs(count=5) default
  scoring.py         # unchanged (task_success already reads termination/operator)
  embodiment.py      # small change — realize the scene's instance per seed (demo)
  __init__.py        # widen API: distributions, TaskInstance, realize, K_* constants
```

### 3.1 `distributions.py` (pure, numpy-only)
A frozen-dataclass `Distribution` family with `sample(rng) -> float | int | str`
and `describe() -> str` (methodology notation). **All samples are cast to builtin
Python scalars** (`float()/int()/str()`) so `Realization.values` is JSON-native
and mypy-strict clean (NumPy returns `np.float64`/`np.int64`, which fail both):

- `Uniform(low, high)` → continuous; `sample` = `float(rng.uniform(low, high))`;
  `describe()` = `"Uniform[8, 15]"`.
- `Categorical(values, weights=None)` → uniform (or weighted) over a finite set of
  str **or** numbers. **Sample by index** to preserve dtype:
  `i = int(rng.choice(len(values), p=weights)); return values[i]` —
  *never* `rng.choice(values)` (which coerces a numeric tuple to a `'<U'` string
  array, turning `12` into `np.str_('12')`). `values: tuple[...]`,
  `weights: tuple[float, ...] | None`; `describe()` = `"Categorical({bowl, cup, pot})"`.
- `Normal(mean, std)` → `float(rng.normal(mean, std))`; `describe()` = `"N(0, 3.0²)"`.
- `Constant(value)` → returns `value` unchanged; `describe()` = `repr(value)`.

`rng` is `np.random.Generator`. Sampling is deterministic given the seed. A
2-D jitter is just two `Normal`s named `jitter_x`, `jitter_y`.

### 3.2 `instances.py`
```python
K_REALIZATIONS = 5   # rollouts per instance (methodology)
K_INSTANCES = 5      # validated instances per task
K_EXPERTS = 3        # validators per instance

@dataclass(frozen=True)
class Validation:               # carried metadata; honest about provenance
    # Per-expert 1–5 ratings (empty until the commissioning pipeline is run).
    representativeness: tuple[int, ...] = ()
    quality: tuple[int, ...] = ()
    difficulty: tuple[int, ...] = ()            # optional, "nice to have" in the methodology
    source: str = "opus-draft"                  # who authored it (vs "human")

    @property
    def validated(self) -> bool:
        # Methodology accept rule: K_EXPERTS ratings, all >= 4 on BOTH axes.
        return (
            len(self.representativeness) >= K_EXPERTS
            and len(self.quality) >= K_EXPERTS
            and all(r >= 4 for r in self.representativeness)
            and all(q >= 4 for q in self.quality)
        )

@dataclass(frozen=True)
class Realization:
    seed: int
    values: dict[str, float | int | str]   # concrete JSON-native sampled values
    instruction: str            # goal with language vars filled in
    setup_lines: tuple[str, ...]  # human-readable "arrange this" lines for an operator

@dataclass(frozen=True)
class TaskInstance:
    instance_id: str
    goal: str                   # template; {var} placeholders for language-relevant vars
    setup: dict[str, Distribution]
    target_kind: str
    language_vars: tuple[str, ...] = ()   # subset of setup keys used in `goal`
    static: dict[str, Any] = field(default_factory=dict)   # NOT `{}` (frozen dc crash)
    validation: Validation = field(default_factory=Validation)

    def realize(self, seed: int) -> Realization: ...
    def setup_spec(self) -> dict[str, str]:   # {var: distribution.describe()} for logging
```
`realize(seed)` builds `np.random.default_rng(seed)`, samples `setup` in **sorted
key order** (determinism), formats `goal` with **only** the `language_vars`, and
renders `setup_lines` from `f"{k} = {v}"`. Pure; no RoboInspect import. Every
`{placeholder}` in `goal` must be in `language_vars` (and in `setup`) — enforced by
a test that realizes all 50 instances (§8).

### 3.3 `specs.py` (rewrite)
`TaskSpec` keeps `key/title/category/bimanual/max_steps/version/description` and
gains `instances: tuple[TaskInstance, ...]` (exactly **5**), dropping `axes`/
`instruction`/`target_kind`/`extra` (those move into each instance). Each task's
five instances are distribution-based scenarios in the methodology's spirit —
the former axis values become `Categorical` setups, plus added physical
stochasticity (positions, jitter, counts, fill levels) and a concrete goal.
Example (`pour_pasta`, one of five instances):
```python
TaskInstance(
    instance_id="pour_pasta/measuring-cup-to-bowl",
    goal="pour the dry pasta from the measuring cup into the {vessel}",
    setup={
        "vessel": Categorical(("bowl", "cup", "pot")),
        "fill_g": Uniform(80, 200),
        "vessel_x_cm": Normal(0.0, 3.0), "vessel_y_cm": Normal(0.0, 3.0),
        "pour_height_cm": Uniform(8, 15),
    },
    language_vars=("vessel",),
    target_kind="pour_into",
    static={"substance": "dry_pasta"},
    validation=Validation(source="opus-draft"),
),
```

### 3.4 `tasks.py` (change)
- `build_scenes(spec)` → **one Scene per instance**:
  - `id = slug(instance_id)`, `init_seed = inst_base_seed` (stable per instance,
    e.g. `index`),
  - `instruction = inst.realize(derive_seed(0, inst_base_seed, 0)).instruction` — an
    **example realization that matches epoch 0 under the default `eval(seed=0)`**, so
    the displayed sentence equals the first actual rollout (not an unrelated seed),
  - `target = Target(kind=inst.target_kind, spec=dict(inst.static))` — no
    `k_realizations` (the real count lives in `task.epoch_spec.count`; a target field
    would mislead),
  - `metadata = {task, category, bimanual, version, instance_id, instance_index,
    setup: inst.setup_spec(), language_vars: list(...), validation:
    asdict(inst.validation)}` — **strictly JSON-native** (strings/numbers/bools/lists);
    no live `Distribution`/`TaskInstance` objects.
- `make_task(spec)` sets `epochs=Epochs(count=K_REALIZATIONS, reducer="mean")` so a
  default run does **5 realizations per instance** and the per-scene reduced metric
  is the instance success probability P̂[Yᵢ=1]. `K_*` constants live in
  `instances.py`, re-exported by `tasks.py`/`__init__`.
- `realize_scene(scene, seed) -> Realization` helper: recovers the instance via
  `SPEC_BY_KEY[scene.metadata["task"]].instances[scene.metadata["instance_index"]]`
  and realizes it, **guarding `seed if seed is not None else 0`** (direct
  `reset(scene)` calls pass `seed=None`; only `eval()` guarantees an int). This is
  the seam a real embodiment/operator tool uses to get the concrete setup.

### 3.5 `embodiment.py` (small, optional change)
The mock `KitchenEmbodiment.reset` already takes `scene.instruction`. To
*demonstrate* realizations end-to-end, when the scene carries an instance
(`scene.metadata.get("task")` is truthy **and** is a known kitchenbench key) it
calls `realize_scene(scene, seed)` and uses the realized instruction on the
`Observation` (so the per-epoch realization is observable on
`record.steps[0].observation.instruction`). The realization rng
(`default_rng`) is **separate from** the existing `goal_dir` rng
(`RandomState`) — they may share the seed value as independent streams, and the
abstract success model is unchanged, so the scripted policy still succeeds
deterministically and the suite stays hardware-free.

**Both branches must be covered:** kitchenbench scenes (instance path) via the
new tests, and bare scenes (fallback, `metadata.get("task")` falsy) via the
existing `tests/test_world.py` which resets a plain `_SCENE`. Because `eval()`
realizes only transiently, the realized instruction does **not** land in the
`EvalLog`; a test that wants to assert it uses a tiny recording `LogSink` reading
`record.steps[0].observation.instruction` in `on_trial_end`, or calls
`realize_scene` directly.

## 4. Defaults from the reference

- **Rollouts per instance:** `Epochs(count=5)` (K_realizations = 5).
- **Instances per task:** 5 (K_i = 5).
- **Reducer:** `"mean"` → per-instance success probability P̂[Yᵢ=1], matching the
  methodology's estimator exactly.

## 5. Scoring (unchanged)

`task_success` already returns success via `termination_reason=="success"` or an
operator verdict; averaged over 5 epochs by the mean reducer it yields P̂[Yᵢ=1].
`episode_length` stays. No scorer changes.

## 6. Honesty about validation (important)

The methodology's `K_i = 5` is *after* human validation (K_experts = 3,
representativeness & quality ≥ 4). We cannot run Prolific here, so every shipped
instance keeps `Validation()` defaults: empty rating tuples and
`source="opus-draft"`, so `Validation.validated` is `False`. We adopt the **count**
(5), the **accept rule** (`validated` requires 3 ratings all ≥ 4 on both axes), and
the **metadata structure**, and document that running the real
commissioning/validation pipeline is required before the instances are trustworthy.
We will **not** fabricate ratings.

## 7. Quality gates (unchanged from kitchenbench)

`ruff`, `ruff format --check`, `mypy --strict`, `pytest --cov` at **100%**.
New deps: none (numpy already a dep). Update README + both CLAUDE.md files.

## 8. Tests

Determinism is tested with **golden values** (assert exact draws for fixed seeds);
variation is tested by `len({draw for seed in range(20)}) > 1` over a batch —
**never** pairwise `realize(a) != realize(b)` on a small Categorical (collides
~1/3 of the time → flaky).

- `test_distributions.py`: each distribution samples within support; `describe()`
  format; golden determinism (same seed → exact draw); weighted Categorical hits
  the weighted branch; **`type(Categorical((12,14,16)).sample(rng)) is int`** (dtype
  preserved); `sample()` returns builtin `float/int/str` (JSON-native) and
  `json.dumps`-able.
- `test_instances.py`: `realize(seed)` golden determinism + reproducibility;
  variation over a 20-seed batch; `goal` formatting via `language_vars`;
  `setup_spec()` strings; `Validation.validated` True only with 3 ratings all ≥4,
  False otherwise (defaults → False); `seed=None` path through `realize_scene`
  guards to 0.
- `test_specs.py`: exactly 10 tasks, each with **5** instances; every
  `language_var ∈ setup`; **every `{placeholder}` in `goal` ⊆ `language_vars`**;
  `target_kind` non-empty; instance_ids globally unique.
- `test_realize_all.py`: **realize every one of the 50 instances** (and call
  `describe()` on every distribution) — catches placeholder/format bugs across all
  authored content and drives `realize`/`describe`/weighted-branch coverage.
- `test_tasks.py` (rewrite): `build_scenes` → 5 scenes/task; `make_task` epochs =
  `Epochs(5, "mean")`; scene metadata is JSON-native (`json.dumps(dict(meta))`
  succeeds) and carries `setup` spec; `realize_scene` recovers + realizes an
  instance; canonical `scene.instruction == realize(derive_seed(0, init_seed, 0))`;
  all 10 `@task` factories register.
- `test_eval_instances.py`: `eval("kitchenbench/pour_pasta", scripted, kitchen)`
  runs 5 scenes × 5 epochs, per-scene reduced `task_success == 1.0`; the realized
  instruction is asserted via a small recording `LogSink` (reading
  `record.steps[0].observation.instruction`) **or** `realize_scene` directly — not
  via the `EvalLog`.
- kitchenbench has **no `__all__` snapshot test**; ruff/mypy catch a stale
  `__init__.__all__` when widening the API.

## 9. Build order (TDD, commit per step)

1. `distributions.py` + tests.
2. `instances.py` (TaskInstance/Realization/Validation + K_* constants) + tests.
3. `specs.py` rewrite (50 instances) + tests.
4. `tasks.py` (build_scenes/make_task/realize_scene, Epochs=5) + tests.
5. `embodiment.py` realize-on-reset + adjust its tests.
6. `__init__.py` API widen; full-suite 100% coverage; ruff + mypy green.
7. README + CLAUDE.md (root + package): document the task-instance model,
   distributions, the 5×5 defaults, and the validation-status caveat.
8. Commit/push in focused steps; CI green.

## 9a. Documentation caveats (README + CLAUDE.md)

- **The methodology output is the per-instance P̂[Yᵢ=1]** in
  `SceneResult.reduced["task_success"]`. `eval()` also reports a task-level
  `metrics["task_success"]` = *mean of P̂ over instances*; that is a **convenience
  aggregate, not a methodology quantity** (the methodology sorts P̂ into quantiles
  and fits pTQ/automation-halvings — out of scope here). Say so explicitly.
- **Mock caveat:** in the dependency-free `KitchenEmbodiment`, success depends only
  on the seed-drawn `goal_dir`, so the sampled *setup distributions have no causal
  effect* and P̂ is degenerately 1.0 for the scripted oracle. The realizations
  exercise the pipeline (and give operators concrete setups on real hardware), but
  the distribution *content* only bites on a real embodiment. Document this so no
  one mistakes the mock P̂ for a real measurement.

## 10. Out of scope

Running the human commissioning/validation pipeline (Prolific), the forecasting
plots (pTQ / automation halvings), and changing RoboInspect core (none needed).
