# 0003 — Simulation support: annotated blueprints + measurable success criteria

Revision 2. Revision 1 derived scene blueprints *generically* from setup-variable
naming conventions; a critique pass against all 50 shipped instances proved that
unsound (vocabulary gaps for jars/bottles/towels, `total_count` with no class
split, unnamed pour sources, tokenization collisions like the "scoop" verb, and
criteria that referenced knowledge no derivation could produce). Resolution,
confirmed with the owner: **sim semantics are explicit, per-instance
annotations in the specs themselves** — the single source of truth carries its
own machine-readable meaning; nothing is guessed.

Revision 3 (PR review follow-up): handoff requires a **direct**
gripper-to-gripper transfer — a set-down resets the tracked holder (a
bench-mediated regrasp is *placing*, not a handoff, and a transient false
"hold" can no longer poison the trial), a both-in-range tie is treated as
mid-transfer (holder kept), and a completed wrong-arm transfer re-establishes
the holder so a later correct handoff can still succeed. Count expansion:
counts below 1 are a blueprint-time error, and a `Var`-bound count always
numbers its copies (even when it resolves to 1) so names stay stable across
epochs; `size_order` factors are floored above zero.

Revision 4 (independent second-pass review): three checker/blueprint defects
fixed. (1) **Fold gates on flatness first** — footprint shrinks for the wrong
reasons too (a grasped, hanging cloth; a still-crumpled ball), so success now
also requires z-extent ≤ `FOLD_MAX_HEIGHT_M` (4 cm); this also stops
`fold_cloth/crumpled-start` from succeeding at t=0. (2) **Grasp distance is
measured to the item's AABB, not its center** — an end grasp of a 40 cm tool
is a grasp; center distance made `handoff/long-tool` unwinnable for 2 of 3
sampled lengths. (3) **Count-expansion spacing is clamped to the widest
copy's nominal footprint + 1 cm clearance** — sampled spreads narrower than
the copies themselves (26 cm plates at 8 cm) would interpenetrate at spawn,
which each engine's penetration resolution would resolve differently,
breaking same-seed-same-scene; the sampled value stays verbatim in
`blueprint.values`. Sort-tray compartments moved to ±10 cm for the same
reason. The instance sweep now asserts no expanded family interpenetrates.

## 1. Goal

Make KitchenBench runnable on physics simulators (a third embodiment class next
to the CI mock and real hardware) by shipping, in this repo:

1. **`SimSpec` annotations** on every `TaskInstance`: what to spawn (objects,
   with placements bound to the instance's setup variables), the **roles** the
   success checker needs (item/surface/vessel/lid/…), per-instance **success
   parameters** (target grams, fold counts, receiving arm), and which variables
   are physical **conditions**.
2. **`build_blueprint(scene, seed)`**: resolves the annotation against the same
   `Realization` the operator checklist uses → a structured `SceneBlueprint`.
3. **`WorldState` protocol + per-target-kind success checkers**: quantitative,
   embodiment-agnostic success — the benchmark's official semantics in sim
   (and usable by any pose-tracked real rig later).

Numpy-only, mypy-strict, 100 % coverage (fake world; no physics import).
Non-goals: the Isaac kitchen itself (assets/adapter — separate repo per
charter); changing distributions, the 5×5 methodology, the mock, or the
operator flow. One deliberate exception: the 5 scoop-task goal *templates*
change to expose the sampled target mass (§3.3) — success must be observable.

## 2. Grounding

(unchanged facts) 10 tasks × 5 instances; `Scene.target = Target(kind,
spec=static)`; `realize_scene(scene, seed)` reproduces per-epoch setups —
`eval()` passes `derive_seed(eval_seed, scene.init_seed, epoch)` into
`Embodiment.reset(scene, seed=...)`, so a sim embodiment calls
`build_blueprint(scene, seed)` with exactly that seed and matches the
checklist a human would have gotten (`seed=None` guard mirrors
`realize_scene`'s). Scoring stays untouched: a sim embodiment that detects
success sets `termination_reason="success"` and the existing `task_success`
scorer fires (the `inspect-robots-isaacsim` adapter already maps detected
success that way). R7: sim embodiments declare
`supported_target_kinds` for realizability gating.

## 3. Design

### 3.1 Annotation model (`kitchenbench/instances.py` + per-instance data in `specs.py`)

```python
@dataclass(frozen=True)
class Var:          # a reference to a setup variable, resolved at realization
    name: str

@dataclass(frozen=True)
class SimObject:
    name: str                       # canonical blueprint name ("spoon", "rack")
    asset: str | Var                # asset class, fixed or sampled ("cutlery" var)
    role: str | None = None         # checker role; None for distractors/props
    x_cm: float | Var = 0.0         # parent frame (bench unless parent set)
    y_cm: float | Var = 0.0
    yaw_deg: float | Var = 0.0
    parent: str | None = None       # spawn relative to / inside another object:
                                    # compartments ride the tray; pasta spawns
                                    # inside its source vessel
    count: int | Var = 1            # count>1 expands to name_1..name_n
    split: tuple[str, ...] = ()     # with count: round-robin asset classes
                                    # (sort's total_count -> spoon,fork,knife)
    spread_cm: float | Var = 0.0    # per-copy layout: copy k at
                                    # (x_cm + k*spread_cm, y_cm) — deterministic,
                                    # so stacks/piles never spawn coincident
    size_cm: float | Var | None = None   # characteristic size (stack/bowls
                                         # diameter_cm); with count>1 a
                                         # size_order Var ("largest_first" /
                                         # "shuffled") maps to a documented
                                         # deterministic per-copy size sequence
    size_order: Var | None = None
    amount_g: float | Var | None = None  # substances: grams spawned (fill_g)

@dataclass(frozen=True)
class SimSpec:
    objects: tuple[SimObject, ...]
    success: tuple[tuple[str, Scalar | Var], ...] = ()  # per-kind params (3.3)
    conditions: tuple[str, ...] = ()          # setup vars that are conditions
```

(Tuple-of-pairs rather than a dict keeps `SimSpec` a valid frozen/hashable
dataclass under mypy-strict — `TaskInstance` is frozen. `Var` resolution order:
`Realization.values` first, then `Target.spec` statics; a name in neither is an
import-time validation error.)

`TaskInstance` gains `sim: SimSpec | None = None` (annotated for all 50 shipped
instances in this PR; `None` stays legal for downstream authors — blueprint
building then raises a clear "instance not sim-annotated" error).

**Coverage invariant (the anti-guessing guarantee, tested per instance):**
every setup-variable name is referenced **at least once** somewhere in the
annotation (object binding, `success`, or `conditions`), and **at most once
among object bindings** (a var may legitimately serve both an object binding
and a success param — pour's `fill_g` is both the spawned `amount_g` and the
success total). Unreferenced or doubly-object-bound variables fail validation
at import time (`_validate_sim_spec` from `TaskInstance.__post_init__`), so an
instance edit that forgets the annotation is a red import, not silent drift.
Statics (`Target.spec`) may also be referenced (e.g. `substance="dry_pasta"`,
sort's category list).

**Reserved names:** `gripper_left`, `gripper_right` (rig frames a sim must
register), `bench` (the work surface). Numbered expansion (`cup_1..n`,
`distractor_1..n`) happens at blueprint time from the sampled count.

### 3.2 `SceneBlueprint` (`kitchenbench/sim/blueprint.py`)

`build_blueprint(scene, seed) -> SceneBlueprint`: looks up the instance (same
registry `realize_scene` uses), realizes values, resolves every `Var`, expands
counts/splits deterministically (order = annotation order; split round-robins
in the declared class order — pure functions of `values`, so sim and operator
runs of the same seed agree):

```python
@dataclass(frozen=True)
class SceneObject:
    name: str; asset: str; role: str | None
    x_cm: float; y_cm: float; yaw_deg: float   # parent frame if parent set
    parent: str | None                          # None = bench frame
    size_cm: float | None                       # characteristic size, if any
    amount_g: float | None                      # substances only

@dataclass(frozen=True)
class SceneBlueprint:
    contract_version: int            # SIM_CONTRACT_VERSION = 1
    scene_id: str; instruction: str; target_kind: str
    objects: tuple[SceneObject, ...]
    roles: Mapping[str, tuple[str, ...]]   # role -> object names (expanded)
    success_params: Mapping[str, Scalar]
    conditions: Mapping[str, Scalar]
    values: Mapping[str, Scalar]           # full realization, verbatim
```

The asset vocabulary is whatever the annotations name (closed set, exported
as `ASSETS: Mapping[str, AssetSpec]` — asset class → nominal dimensions
(footprint w×h cm, height, characteristic diameter). Adapters build catalogs
against it, and the **success checkers read nominal dims from it** where a
criterion needs a baseline the scene can't provide: fold's flat-area
denominator for the crumpled-start instance (sampled cloth asset → that
asset's nominal flat dims; one scalar pair per asset, so a sampled asset is
fine) and `size_order`'s deterministic per-copy size sequence (base size =
the asset's nominal diameter when the instance samples no `size_cm`/
`diameter_cm`). Includes jar, bottle, food_container, dish_towel,
measuring_cup, penne, rigatoni, … — the full set actually appearing in
specs, which revision 1's word-matching missed.

### 3.3 Success criteria (`kitchenbench/sim/success.py`)

```python
class WorldState(Protocol):
    def aabb(self, name: str) -> tuple[NDArray, NDArray]: ...          # meters
    def contained_fraction(self, item: str, container: str) -> float: ...
    def contained_mass_g(self, substance: str, container: str) -> float: ...
    def opening_fraction(self, name: str) -> float: ...
```

`make_success_checker(blueprint, world) -> SuccessChecker` (captures initial
state where a criterion needs it), checker returns
`Verdict(success, explanation)` and **never raises** — unknown object/queries
yield `Verdict(False, "unknown object 'x'")`. Required roles per kind are
declared in one table; the instance sweep asserts every annotation provides
them.

| kind | roles / params | success iff |
|---|---|---|
| `place_on` | item, surface | item xy-center inside surface footprint ∧ item bottom within `CONTACT_TOL_M` of surface top |
| `stack` | stack (multi) | sorted by bottom-z: each upper xy-center inside lower footprint ∧ upper bottom ∈ `[lower bottom, lower top + CONTACT_TOL_M]` (**nesting-aware**: cups/bowls sit inside; plates sit on top — both pass) |
| `place_in` | item, rack | `contained_fraction(item, rack) ≥ CONTAIN_FRACTION` |
| `pour_into` | vessel; params `substance`, `total_g` (=`fill_g`) | `contained_mass_g(substance, vessel) ≥ TRANSFER_FRACTION × total_g` |
| `open` | container | `opening_fraction ≥ OPEN_FRACTION` |
| `fold` | cloth; params `fold_count` or `flat_w_cm`+`flat_h_cm` | z-extent ≤ `FOLD_MAX_HEIGHT_M` (lying flat — a grasped/hanging or still-crumpled cloth must not fire, rev 4) ∧ xy-footprint area ≤ `FOLD_RATIO_PER_FOLD ** fold_count` × baseline area; baseline = initial capture normally, but crumpled-start annotates the cloth's **nominal flat dims** (`flat_w_cm × flat_h_cm`, part of the asset contract) as the denominator since its initial footprint is already shrunk |
| `seal` | lid, container | lid xy-center inside container footprint ∧ lid bottom within `SEAL_TOL_M` of container top (separate, looser than `CONTACT_TOL_M` — screw lids seat below the rim) |
| `handoff` | item; param `receiving_arm` ("left"/"right"/"either") | **transfer-detecting and stateful**: the checker tracks the holder per evaluation (holder = gripper whose center is within `GRASP_RADIUS_M` of the item's **AABB** — surface distance, not center distance, so end grasps of long tools count (rev 4) — and strictly nearer than the other); success once the holder has changed from an established holder to the other gripper **and the previous holder is beyond `GRASP_RADIUS_M`** (mid-transfer, with both grippers on the item, must not fire); the transfer must be **direct** — a set-down (item away from both grippers) resets the tracked holder, so bench-mediated regrasps don't count — and a completed wrong-arm transfer re-establishes the holder (rev 3) — matching `receiving_arm` when pinned ("left"/"right" for pickup/release-forced instances: utensil, cross-body), any direction for the 3 direction-symmetric instances ("either") |
| `sort` | compartments (multi, `compartment_<class>`), sortables (multi) | every sortable `contained_fraction(obj, compartment_<its class>) ≥ CONTAIN_FRACTION` (classes from the annotation's split/statics) |
| `scoop_transfer` | container; params `substance`, `target_g`, `tol_g` | `abs(contained_mass_g − target_g) ≤ tol_g` (default `SCOOP_TOL_G`, which dominates instruction rounding) — the *target amount*, not the whole pile. **Requires a small spec change** (owner-visible in the PR): the 5 scoop goal templates gain the target ("scoop about {fill_target_g:.0f} g of the {pasta} …" — format spec keeps the instruction readable); `fill_target_g` joins `language_vars`, instructions change, and the scoop `TaskSpec.version` bumps to "2" (a benchmark-definition change, named explicitly in milestone 2 — without it success depends on a value neither the policy nor a real-world operator can observe, a pre-existing benchmark bug this work surfaced) |

Thresholds are named constants in one table (benchmark parameters,
`SIM_CONTRACT_VERSION = 1`; the version is stamped on every blueprint so logs
can record which semantics produced a success).

Pour scenes gain an explicit **source** object in their annotations
(`measuring_cup`/`pasta_box` per instance — revision 1 had no spawnable source
holding the pasta). Sort scenes gain explicit `tray` + `compartment_*` objects.
Initial-state capture exists only where declared (fold, handoff), so checkers
stay evaluable from step 1 without false positives from default placements —
lid/stack annotations bind the real offset variables (`lid_offset_cm`,
`spread_cm`) so nothing spawns pre-satisfied.

### 3.4 Consumption story (README "Run it in simulation")

```python
bp = build_blueprint(scene, seed)          # in Embodiment.reset()
# spawn bp.objects (catalog keyed by bp asset names), apply bp.conditions
checker = make_success_checker(bp, world)
# per step: if checker(world).success -> terminate, reason="success"
```

Plus: declare `supported_target_kinds`; log `bp.contract_version`.

### 3.5 Alternatives considered

- Generic derivation from naming conventions (revision 1): rejected — proven
  unsound against the shipped instances (17-issue critique).
- Success detection per adapter: rejected — benchmark numbers must mean the
  same thing everywhere.
- Separate annotation file keyed by instance_id: rejected — drifts from
  specs.py; the annotation belongs on the instance it describes, enforced by
  the coverage invariant at import time.

## 4. Files

```
kitchenbench/
├── plans/0003-sim-support.md          (this doc)
├── src/kitchenbench/instances.py      (Var, SimObject, SimSpec, TaskInstance.sim,
│                                       _validate_sim_spec in __post_init__)
├── src/kitchenbench/specs.py          (sim=SimSpec(...) on all 50 instances)
├── src/kitchenbench/sim/__init__.py   (public API + SIM_CONTRACT_VERSION)
├── src/kitchenbench/sim/blueprint.py
├── src/kitchenbench/sim/success.py
├── src/kitchenbench/__init__.py       (re-exports; __all__ additions)
├── src/kitchenbench/CLAUDE.md         (module map + how to annotate)
├── tests/test_api_snapshot.py         (new: pin __all__, mirroring the framework)
├── tests/test_sim_blueprint.py
├── tests/test_sim_success.py
└── README.md                          ("Run it in simulation")
```

## 5. Testing (100 % coverage; positive assertions, no vacuous sweeps)

- **Coverage invariant** enforced in code at import; tests cover its failure
  modes (unreferenced var, doubly-object-bound var, unknown Var name, unknown
  or forward-referencing `parent` — parents must be declared earlier in the
  tuple, which makes cycles unrepresentable — missing required role for the
  kind) via deliberately broken `SimSpec`s.
- **Sweep over all 50 instances × 3 seeds**: blueprint builds; `roles`
  contains the kind's required roles; every object asset ∈ `ASSETS`; object
  names unique; counts expand with per-copy `spread_cm` layout
  (`stack/plates` count=n → `plate_1..n` from the annotated base + spread);
  `sort/overlapping`'s `total_count` split round-robins deterministically with
  ≥1 object per declared class; **spawnability**: no two same-parent resolved
  placements coincide (within 1 mm) — catches missing layout bindings and
  forces literal offsets where instances sample none (e.g.
  `place_cutlery/from-drawer`'s cutlery); resolved placements equal the
  corresponding `Realization.values` entries.
- **Pinned examples**: `place_cutlery/spoon-on-plate` seed-0 blueprint equals
  a hand-written expectation; `pour_pasta/*` has a source object; seal binds
  `lid_offset_cm`.
- **Success criteria per kind**: FakeWorld pass / fail / threshold-boundary
  cases; nesting stack (cup-in-cup) passes and side-by-side fails; scoop
  over-transfer (whole pile) **fails**; fold initial-capture (moving cloth
  between construction and evaluation flips verdict); handoff wrong-arm fails;
  unknown-object → `Verdict(False, …)`, never raises.
- **API snapshot** test pins `kitchenbench.__all__`.
- **Subagent audits** after implementation: test-vacuousness/mutation pass
  (flip a footprint check, drop initial capture, swap grippers, break the
  round-robin — each must be killed).

## 6. Milestones

1. Annotation model + validation in `instances.py` (+ unit tests).
2. `specs.py`: annotate all 50 instances; **includes the scoop goal-template/
   language_vars edit + scoop version bump to "2"** (the one non-mechanical,
   benchmark-visible change; canonical instructions in scene metadata change
   with it).
3. `sim/blueprint.py` + tests.
4. `sim/success.py` + tests.
5. Exports, api-snapshot test, CLAUDE.md, README; `uv lock` untouched (no dep
   changes); mutation audit; PR.
