"""Declarative specifications for the KitchenBench tasks — the single source of truth.

Each :class:`TaskSpec` carries exactly ``K_INSTANCES`` (5) :class:`TaskInstance`\\ s
(see :mod:`kitchenbench.instances`): self-contained scenarios whose environment
setups are **distributions**. :mod:`kitchenbench.tasks` turns each instance into a
RoboLens ``Scene`` and runs ``K_REALIZATIONS`` (5) realizations per instance.

Instances are AI-authored drafts (``Validation(source="opus-draft")``) and are
**not yet human-validated** — the methodology requires K_EXPERTS=3 reviewers
(representativeness & quality >= 4) before the instances are trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass

from kitchenbench.distributions import Categorical, Constant, Normal, Uniform
from kitchenbench.instances import TaskInstance


@dataclass(frozen=True)
class TaskSpec:
    """One KitchenBench task and its (distribution-based) task instances."""

    key: str
    title: str
    category: str
    bimanual: bool
    max_steps: int
    instances: tuple[TaskInstance, ...]
    version: str = "1"
    description: str = ""


# Common reusable distributions.
def _jitter(sigma_cm: float) -> Normal:
    """A 1-axis centroid jitter ~ N(0, sigma²) in cm."""
    return Normal(0.0, sigma_cm)


_PLACE_CUTLERY = (
    TaskInstance(
        instance_id="place_cutlery/spoon-on-plate",
        goal="place the {cutlery} on the {dishware}",
        setup={
            "cutlery": Categorical(("spoon", "fork", "knife")),
            "dishware": Categorical(("plate", "bowl", "napkin")),
            "cutlery_x_cm": Uniform(-20, -5),
            "dishware_x_cm": Uniform(5, 20),
            "jitter_y_cm": _jitter(2.0),
        },
        language_vars=("cutlery", "dishware"),
        target_kind="place_on",
    ),
    TaskInstance(
        instance_id="place_cutlery/from-drawer",
        goal="place the {cutlery} on the plate",
        setup={
            "cutlery": Categorical(("spoon", "fork", "knife")),
            "approach_angle_deg": Uniform(-30, 30),
            "plate_x_cm": Normal(8.0, 2.0),
            "plate_y_cm": _jitter(2.5),
        },
        language_vars=("cutlery",),
        target_kind="place_on",
    ),
    TaskInstance(
        instance_id="place_cutlery/cluttered-bench",
        goal="place the fork on the {dishware}",
        setup={
            "dishware": Categorical(("plate", "bowl")),
            "distractor_count": Categorical((1, 2, 3)),
            "fork_x_cm": Uniform(-18, -8),
            "fork_yaw_deg": Uniform(0, 90),
        },
        language_vars=("dishware",),
        target_kind="place_on",
    ),
    TaskInstance(
        instance_id="place_cutlery/napkin-soft",
        goal="place the {cutlery} on the napkin",
        setup={
            "cutlery": Categorical(("spoon", "fork", "knife")),
            "napkin_x_cm": Normal(10.0, 3.0),
            "napkin_y_cm": _jitter(3.0),
            "napkin_rotation_deg": Uniform(0, 45),
        },
        language_vars=("cutlery",),
        target_kind="place_on",
    ),
    TaskInstance(
        instance_id="place_cutlery/far-reach",
        goal="place the knife on the {dishware}",
        setup={
            "dishware": Categorical(("plate", "bowl", "napkin")),
            "dishware_x_cm": Uniform(20, 30),
            "knife_x_cm": Uniform(-25, -15),
            "jitter_y_cm": _jitter(2.0),
        },
        language_vars=("dishware",),
        target_kind="place_on",
    ),
)

_STACK = (
    TaskInstance(
        instance_id="stack/cups",
        goal="stack the cups",
        setup={
            "count": Categorical((2, 3, 4)),
            "spread_cm": Uniform(5, 15),
            "jitter_x_cm": _jitter(2.0),
            "jitter_y_cm": _jitter(2.0),
        },
        target_kind="stack",
    ),
    TaskInstance(
        instance_id="stack/bowls",
        goal="stack the bowls",
        setup={
            "count": Categorical((2, 3)),
            "diameter_cm": Categorical((12, 15, 18)),
            "spread_cm": Uniform(8, 18),
        },
        target_kind="stack",
    ),
    TaskInstance(
        instance_id="stack/plates",
        goal="stack the plates",
        setup={
            "count": Categorical((2, 3, 4, 5)),
            "plate_x_cm": Normal(0.0, 3.0),
            "plate_y_cm": _jitter(3.0),
        },
        target_kind="stack",
    ),
    TaskInstance(
        instance_id="stack/mixed-sizes",
        goal="stack the bowls",
        setup={
            "count": Categorical((3, 4)),
            "size_order": Categorical(("largest_first", "shuffled")),
            "spread_cm": Uniform(10, 20),
        },
        target_kind="stack",
    ),
    TaskInstance(
        instance_id="stack/tight-spacing",
        goal="stack the cups",
        setup={
            "count": Categorical((2, 3)),
            "spread_cm": Uniform(3, 8),
            "jitter_x_cm": _jitter(1.0),
        },
        target_kind="stack",
    ),
)

_PLACE_IN_RACK = (
    TaskInstance(
        instance_id="place_in_rack/plate",
        goal="place the {dishware} into the dish rack",
        setup={
            "dishware": Categorical(("plate", "bowl", "cup")),
            "slot_index": Categorical((1, 2, 3, 4)),
            "rack_x_cm": Normal(12.0, 2.0),
            "dishware_yaw_deg": Uniform(0, 90),
        },
        language_vars=("dishware",),
        target_kind="place_in",
    ),
    TaskInstance(
        instance_id="place_in_rack/bowl-tilt",
        goal="place the bowl into the dish rack",
        setup={
            "slot_index": Categorical((1, 2, 3)),
            "approach_tilt_deg": Uniform(0, 25),
            "rack_y_cm": _jitter(2.0),
        },
        target_kind="place_in",
    ),
    TaskInstance(
        instance_id="place_in_rack/cup-narrow",
        goal="place the cup into the dish rack",
        setup={
            "slot_index": Categorical((1, 2, 3, 4, 5)),
            "slot_width_cm": Categorical((6, 8)),
            "cup_x_cm": Uniform(-15, -5),
        },
        target_kind="place_in",
    ),
    TaskInstance(
        instance_id="place_in_rack/rightmost-slot",
        goal="place the {dishware} into the dish rack",
        setup={
            "dishware": Categorical(("plate", "bowl")),
            "slot_index": Constant(4),
            "rack_x_cm": Uniform(10, 18),
        },
        language_vars=("dishware",),
        target_kind="place_in",
    ),
    TaskInstance(
        instance_id="place_in_rack/wet-slippery",
        goal="place the plate into the dish rack",
        setup={
            "slot_index": Categorical((1, 2, 3, 4)),
            "surface_friction": Categorical(("low", "medium")),
            "plate_x_cm": Normal(-10.0, 2.0),
        },
        target_kind="place_in",
    ),
)

_POUR_PASTA = (
    TaskInstance(
        instance_id="pour_pasta/measuring-cup-to-bowl",
        goal="pour the dry pasta into the {vessel}",
        setup={
            "vessel": Categorical(("bowl", "cup", "pot")),
            "fill_g": Uniform(80, 200),
            "vessel_x_cm": Normal(0.0, 3.0),
            "vessel_y_cm": _jitter(3.0),
            "pour_height_cm": Uniform(8, 15),
        },
        language_vars=("vessel",),
        target_kind="pour_into",
        static={"substance": "dry_pasta"},
    ),
    TaskInstance(
        instance_id="pour_pasta/full-box",
        goal="pour the dry pasta into the pot",
        setup={
            "fill_g": Uniform(300, 500),
            "pour_angle_deg": Uniform(45, 80),
            "pot_x_cm": Normal(5.0, 2.0),
        },
        target_kind="pour_into",
        static={"substance": "dry_pasta"},
    ),
    TaskInstance(
        instance_id="pour_pasta/narrow-cup",
        goal="pour the dry pasta into the cup",
        setup={
            "fill_g": Uniform(40, 120),
            "cup_diameter_cm": Categorical((6, 8)),
            "pour_height_cm": Uniform(5, 10),
        },
        target_kind="pour_into",
        static={"substance": "dry_pasta"},
    ),
    TaskInstance(
        instance_id="pour_pasta/steady-and-pour",
        goal="pour the dry pasta into the {vessel}",
        setup={
            "vessel": Categorical(("bowl", "pot")),
            "fill_g": Uniform(150, 350),
            "steadying_force_n": Uniform(2, 6),
            "vessel_x_cm": _jitter(2.5),
        },
        language_vars=("vessel",),
        target_kind="pour_into",
        static={"substance": "dry_pasta"},
    ),
    TaskInstance(
        instance_id="pour_pasta/long-spaghetti",
        goal="pour the dry pasta into the pot",
        setup={
            "fill_g": Uniform(200, 400),
            "strand_length_cm": Categorical((24, 26)),
            "pot_y_cm": _jitter(3.0),
        },
        target_kind="pour_into",
        static={"substance": "dry_pasta"},
    ),
)

_OPEN_CONTAINER = (
    TaskInstance(
        instance_id="open_container/jar",
        goal="open the {container}",
        setup={
            "container": Categorical(("jar", "bottle", "food container")),
            "lid_torque_nm": Uniform(0.5, 2.5),
            "container_x_cm": Normal(0.0, 2.0),
            "container_yaw_deg": Uniform(0, 90),
        },
        language_vars=("container",),
        target_kind="open",
    ),
    TaskInstance(
        instance_id="open_container/bottle-cap",
        goal="open the bottle",
        setup={
            "cap_turns": Categorical((1, 2, 3)),
            "lid_torque_nm": Uniform(0.3, 1.5),
            "bottle_x_cm": _jitter(2.0),
        },
        target_kind="open",
    ),
    TaskInstance(
        instance_id="open_container/snap-lid",
        goal="open the food container",
        setup={
            "clip_count": Categorical((2, 4)),
            "pry_force_n": Uniform(5, 20),
            "container_y_cm": _jitter(2.5),
        },
        target_kind="open",
    ),
    TaskInstance(
        instance_id="open_container/stiff-jar",
        goal="open the jar",
        setup={
            "lid_torque_nm": Uniform(2.0, 4.0),
            "brace_force_n": Uniform(8, 20),
            "jar_diameter_cm": Categorical((6, 8, 10)),
        },
        target_kind="open",
    ),
    TaskInstance(
        instance_id="open_container/tilted",
        goal="open the {container}",
        setup={
            "container": Categorical(("jar", "bottle")),
            "tilt_deg": Uniform(0, 30),
            "lid_torque_nm": Uniform(0.5, 2.0),
        },
        language_vars=("container",),
        target_kind="open",
    ),
)

_FOLD_CLOTH = (
    TaskInstance(
        instance_id="fold_cloth/dish-towel",
        goal="fold the {cloth}",
        setup={
            "cloth": Categorical(("dish towel", "napkin", "cloth")),
            "size_cm": Categorical((30, 40, 50)),
            "initial_rotation_deg": Uniform(0, 90),
            "slack": Uniform(0.0, 0.4),
        },
        language_vars=("cloth",),
        target_kind="fold",
    ),
    TaskInstance(
        instance_id="fold_cloth/napkin-half",
        goal="fold the napkin",
        setup={
            "fold_count": Categorical((1, 2)),
            "napkin_x_cm": Normal(0.0, 2.0),
            "napkin_y_cm": _jitter(2.0),
        },
        target_kind="fold",
    ),
    TaskInstance(
        instance_id="fold_cloth/large-cloth",
        goal="fold the cloth",
        setup={
            "size_cm": Categorical((50, 60, 70)),
            "slack": Uniform(0.2, 0.6),
            "corner_offset_cm": _jitter(4.0),
        },
        target_kind="fold",
    ),
    TaskInstance(
        instance_id="fold_cloth/crumpled-start",
        goal="fold the {cloth}",
        setup={
            "cloth": Categorical(("dish towel", "cloth")),
            "crumple_level": Categorical(("light", "moderate")),
            "initial_rotation_deg": Uniform(0, 180),
        },
        language_vars=("cloth",),
        target_kind="fold",
    ),
    TaskInstance(
        instance_id="fold_cloth/thirds",
        goal="fold the dish towel",
        setup={
            "fold_count": Constant(2),
            "size_cm": Categorical((40, 50)),
            "slack": Uniform(0.0, 0.3),
        },
        target_kind="fold",
    ),
)

_SEAL_CONTAINER = (
    TaskInstance(
        instance_id="seal_container/food-container",
        goal="seal the {container} with its lid",
        setup={
            "container": Categorical(("food container", "pot", "jar")),
            "lid_offset_cm": _jitter(2.0),
            "press_force_n": Uniform(5, 20),
            "lid_yaw_deg": Uniform(0, 45),
        },
        language_vars=("container",),
        target_kind="seal",
    ),
    TaskInstance(
        instance_id="seal_container/twist-lid-jar",
        goal="seal the jar with its lid",
        setup={
            "thread_turns": Categorical((1, 2, 3)),
            "seat_torque_nm": Uniform(0.5, 2.0),
            "jar_x_cm": Normal(0.0, 2.0),
        },
        target_kind="seal",
    ),
    TaskInstance(
        instance_id="seal_container/snap-on-pot",
        goal="seal the pot with its lid",
        setup={
            "clip_count": Categorical((2, 4)),
            "press_force_n": Uniform(10, 30),
            "pot_y_cm": _jitter(2.5),
        },
        target_kind="seal",
    ),
    TaskInstance(
        instance_id="seal_container/misaligned-lid",
        goal="seal the {container} with its lid",
        setup={
            "container": Categorical(("food container", "pot")),
            "lid_offset_cm": Uniform(2, 5),
            "lid_yaw_deg": Uniform(15, 60),
        },
        language_vars=("container",),
        target_kind="seal",
    ),
    TaskInstance(
        instance_id="seal_container/tight-fit",
        goal="seal the food container with its lid",
        setup={
            "clearance_mm": Categorical((1, 2)),
            "press_force_n": Uniform(15, 35),
            "lid_offset_cm": _jitter(1.0),
        },
        target_kind="seal",
    ),
)

_HANDOFF = (
    TaskInstance(
        instance_id="handoff/utensil",
        goal="hand off the {item} from one arm to the other",
        setup={
            "item": Categorical(("utensil", "cup", "produce item")),
            "pickup_x_cm": Uniform(-25, -10),
            "handoff_height_cm": Uniform(15, 30),
            "item_yaw_deg": Uniform(0, 90),
        },
        language_vars=("item",),
        target_kind="handoff",
    ),
    TaskInstance(
        instance_id="handoff/cup-upright",
        goal="hand off the cup from one arm to the other",
        setup={
            "fill_level": Categorical(("empty", "half")),
            "handoff_x_cm": Normal(0.0, 2.0),
            "handoff_height_cm": Uniform(18, 28),
        },
        target_kind="handoff",
    ),
    TaskInstance(
        instance_id="handoff/produce-delicate",
        goal="hand off the produce item from one arm to the other",
        setup={
            "fragility": Categorical(("firm", "delicate")),
            "grip_force_n": Uniform(2, 8),
            "pickup_y_cm": _jitter(2.0),
        },
        target_kind="handoff",
    ),
    TaskInstance(
        instance_id="handoff/long-tool",
        goal="hand off the {item} from one arm to the other",
        setup={
            "item": Categorical(("utensil", "produce item")),
            "length_cm": Categorical((20, 30, 40)),
            "handoff_height_cm": Uniform(15, 25),
        },
        language_vars=("item",),
        target_kind="handoff",
    ),
    TaskInstance(
        instance_id="handoff/cross-body",
        goal="hand off the cup from one arm to the other",
        setup={
            "pickup_x_cm": Uniform(10, 25),
            "release_x_cm": Uniform(-25, -10),
            "handoff_height_cm": Uniform(20, 30),
        },
        target_kind="handoff",
    ),
)

_SORT_CUTLERY = (
    TaskInstance(
        instance_id="sort_cutlery/balanced-pile",
        goal="sort the cutlery into the correct tray compartments",
        setup={
            "spoon_count": Categorical((2, 3)),
            "fork_count": Categorical((2, 3)),
            "knife_count": Categorical((2, 3)),
            "pile_spread_cm": Uniform(8, 18),
        },
        target_kind="sort",
        static={"categories": "spoon,fork,knife"},
    ),
    TaskInstance(
        instance_id="sort_cutlery/spoon-heavy",
        goal="sort the cutlery into the correct tray compartments",
        setup={
            "spoon_count": Categorical((4, 5, 6)),
            "fork_count": Categorical((1, 2)),
            "knife_count": Categorical((1, 2)),
        },
        target_kind="sort",
        static={"categories": "spoon,fork,knife"},
    ),
    TaskInstance(
        instance_id="sort_cutlery/overlapping",
        goal="sort the cutlery into the correct tray compartments",
        setup={
            "total_count": Categorical((6, 8, 10)),
            "overlap_level": Categorical(("low", "high")),
            "pile_x_cm": Normal(0.0, 3.0),
        },
        target_kind="sort",
        static={"categories": "spoon,fork,knife"},
    ),
    TaskInstance(
        instance_id="sort_cutlery/tray-offset",
        goal="sort the cutlery into the correct tray compartments",
        setup={
            "tray_x_cm": Uniform(10, 20),
            "tray_yaw_deg": Uniform(0, 30),
            "total_count": Categorical((6, 9)),
        },
        target_kind="sort",
        static={"categories": "spoon,fork,knife"},
    ),
    TaskInstance(
        instance_id="sort_cutlery/sparse",
        goal="sort the cutlery into the correct tray compartments",
        setup={
            "spoon_count": Categorical((1, 2)),
            "fork_count": Categorical((1, 2)),
            "knife_count": Categorical((1, 2)),
            "pile_spread_cm": Uniform(12, 24),
        },
        target_kind="sort",
        static={"categories": "spoon,fork,knife"},
    ),
)

_SCOOP_PASTA = (
    TaskInstance(
        instance_id="scoop_pasta/spoon-penne",
        goal="scoop the {pasta} with the {tool} and transfer it to the container",
        setup={
            "pasta": Categorical(("penne", "rigatoni")),
            "tool": Categorical(("spoon", "measuring cup")),
            "fill_target_g": Uniform(30, 120),
            "pile_x_cm": Normal(-8.0, 2.0),
            "container_x_cm": Uniform(8, 18),
        },
        language_vars=("pasta", "tool"),
        target_kind="scoop_transfer",
    ),
    TaskInstance(
        instance_id="scoop_pasta/measuring-cup-level",
        goal="scoop the penne with the measuring cup and transfer it to the container",
        setup={
            "fill_target_g": Uniform(80, 160),
            "level_tolerance_g": Categorical((5, 10)),
            "pile_y_cm": _jitter(2.0),
        },
        target_kind="scoop_transfer",
    ),
    TaskInstance(
        instance_id="scoop_pasta/rigatoni-large",
        goal="scoop the rigatoni with the {tool} and transfer it to the container",
        setup={
            "tool": Categorical(("spoon", "measuring cup")),
            "fill_target_g": Uniform(40, 100),
            "pile_depth_cm": Categorical((3, 5, 7)),
        },
        language_vars=("tool",),
        target_kind="scoop_transfer",
    ),
    TaskInstance(
        instance_id="scoop_pasta/shallow-pile",
        goal="scoop the {pasta} with the spoon and transfer it to the container",
        setup={
            "pasta": Categorical(("penne", "rigatoni")),
            "pile_depth_cm": Categorical((1, 2)),
            "fill_target_g": Uniform(20, 60),
        },
        language_vars=("pasta",),
        target_kind="scoop_transfer",
    ),
    TaskInstance(
        instance_id="scoop_pasta/far-container",
        goal="scoop the penne with the measuring cup and transfer it to the container",
        setup={
            "container_x_cm": Uniform(20, 32),
            "fill_target_g": Uniform(60, 140),
            "transfer_height_cm": Uniform(10, 20),
        },
        target_kind="scoop_transfer",
    ),
)


SPECS: tuple[TaskSpec, ...] = (
    TaskSpec(
        key="place_cutlery",
        title="Place cutlery on dishware",
        category="pick_place",
        bimanual=False,
        max_steps=60,
        instances=_PLACE_CUTLERY,
        description="Pick a single piece of cutlery and place it on a target surface.",
    ),
    TaskSpec(
        key="stack",
        title="Stack dishware",
        category="stacking",
        bimanual=False,
        max_steps=80,
        instances=_STACK,
        description="Stack multiple like items into a single neat stack.",
    ),
    TaskSpec(
        key="place_in_rack",
        title="Place dishware in the dish rack",
        category="insertion",
        bimanual=False,
        max_steps=80,
        instances=_PLACE_IN_RACK,
        description="Drop a dish into the correct slot of a dish rack.",
    ),
    TaskSpec(
        key="pour_pasta",
        title="Pour dry pasta into a vessel",
        category="granular",
        bimanual=True,
        max_steps=100,
        instances=_POUR_PASTA,
        description="Pour dry pasta into a receiving vessel; one arm steadies, the other pours.",
    ),
    TaskSpec(
        key="open_container",
        title="Open a container",
        category="articulated",
        bimanual=True,
        max_steps=120,
        instances=_OPEN_CONTAINER,
        description="Remove or unscrew a lid — one arm braces while the other twists or pries.",
    ),
    TaskSpec(
        key="fold_cloth",
        title="Fold a cloth",
        category="deformable",
        bimanual=True,
        max_steps=120,
        instances=_FOLD_CLOTH,
        description="Deformable manipulation: grasp opposite corners and manage slack.",
    ),
    TaskSpec(
        key="seal_container",
        title="Seal a container with its lid",
        category="mating",
        bimanual=True,
        max_steps=120,
        instances=_SEAL_CONTAINER,
        description="Align and press-or-twist a matching lid onto a base while one arm holds it.",
    ),
    TaskSpec(
        key="handoff",
        title="Hand off an object between arms",
        category="coordination",
        bimanual=True,
        max_steps=80,
        instances=_HANDOFF,
        description="A pure handover that a single arm cannot do — the must-use-both-arms anchor.",
    ),
    TaskSpec(
        key="sort_cutlery",
        title="Sort cutlery into a utensil tray",
        category="classification",
        bimanual=False,
        max_steps=200,
        instances=_SORT_CUTLERY,
        description="Sort a mixed pile into spoon/fork/knife compartments — multi-instance.",
    ),
    TaskSpec(
        key="scoop_pasta",
        title="Scoop pasta with a tool and transfer it",
        category="granular_tool",
        bimanual=True,
        max_steps=120,
        instances=_SCOOP_PASTA,
        description="Tool-mediated granular handling: manage fill level, then transfer.",
    ),
)

SPEC_BY_KEY: dict[str, TaskSpec] = {spec.key: spec for spec in SPECS}
