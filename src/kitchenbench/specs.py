"""Declarative specifications for the KitchenBench tasks — the single source of truth.

Each :class:`TaskSpec` is turned into a registered RoboLens ``@task`` in
:mod:`kitchenbench.tasks`, whose scenes are the Cartesian product of its
variation ``axes``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    """One KitchenBench task and the axes its scenes vary over."""

    key: str
    title: str
    instruction: str  # ``str.format`` template over the axis names
    axes: dict[str, tuple[str, ...]]
    target_kind: str
    bimanual: bool
    category: str
    max_steps: int
    version: str = "1"
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)  # merged into each Target.spec


SPECS: tuple[TaskSpec, ...] = (
    TaskSpec(
        key="place_cutlery",
        title="Place cutlery on dishware",
        instruction="place the {cutlery} on the {dishware}",
        axes={
            "cutlery": ("spoon", "fork", "knife"),
            "dishware": ("plate", "bowl", "napkin"),
        },
        target_kind="place_on",
        bimanual=False,
        category="pick_place",
        max_steps=60,
        description="Pick a single piece of cutlery and place it on a target surface.",
    ),
    TaskSpec(
        key="stack",
        title="Stack dishware",
        instruction="stack the {items}",
        axes={"items": ("cups", "bowls", "plates")},
        target_kind="stack",
        bimanual=False,
        category="stacking",
        max_steps=80,
        description="Stack multiple like items into a single neat stack.",
    ),
    TaskSpec(
        key="place_in_rack",
        title="Place dishware in the dish rack",
        instruction="place the {dishware} into the dish rack",
        axes={"dishware": ("plate", "bowl", "cup")},
        target_kind="place_in",
        bimanual=False,
        category="insertion",
        max_steps=80,
        description="Drop a dish into the correct slot of a dish rack.",
    ),
    TaskSpec(
        key="pour_pasta",
        title="Pour dry pasta into a vessel",
        instruction="pour the dry pasta into the {vessel}",
        axes={"vessel": ("bowl", "cup", "pot")},
        target_kind="pour_into",
        bimanual=True,
        category="granular",
        max_steps=100,
        description="Pour dry pasta from a container into a receiving vessel; one arm "
        "steadies the vessel while the other pours.",
        extra={"substance": "dry_pasta"},
    ),
    TaskSpec(
        key="open_container",
        title="Open a container",
        instruction="open the {container}",
        axes={"container": ("jar", "bottle", "food container")},
        target_kind="open",
        bimanual=True,
        category="articulated",
        max_steps=120,
        description="Remove or unscrew a lid — one arm braces the body while the other "
        "twists or pries (contact-rich, coordinated force).",
    ),
    TaskSpec(
        key="fold_cloth",
        title="Fold a cloth",
        instruction="fold the {cloth}",
        axes={"cloth": ("dish towel", "napkin", "cloth")},
        target_kind="fold",
        bimanual=True,
        category="deformable",
        max_steps=120,
        description="Deformable manipulation: grasp opposite corners and manage slack.",
    ),
    TaskSpec(
        key="seal_container",
        title="Seal a container with its lid",
        instruction="seal the {container} with its lid",
        axes={"container": ("food container", "pot", "jar")},
        target_kind="seal",
        bimanual=True,
        category="mating",
        max_steps=120,
        description="Align and press-or-twist a matching lid onto a base (constrained "
        "insertion / part mating) while one arm holds the base.",
    ),
    TaskSpec(
        key="handoff",
        title="Hand off an object between arms",
        instruction="hand off the {item} from one arm to the other",
        axes={"item": ("utensil", "cup", "produce item")},
        target_kind="handoff",
        bimanual=True,
        category="coordination",
        max_steps=80,
        description="A pure handover that a single arm literally cannot do — the "
        "must-use-both-arms anchor task.",
    ),
    TaskSpec(
        key="sort_cutlery",
        title="Sort cutlery into a utensil tray",
        instruction="sort the cutlery into the correct tray compartments",
        axes={"layout": ("a", "b", "c")},  # seeds: different mixed-pile arrangements
        target_kind="sort",
        bimanual=False,
        category="classification",
        max_steps=200,
        description="Sort a mixed pile into spoons/forks/knives compartments — "
        "multi-instance, so it tests consistency, not a single lucky success.",
        extra={"categories": "spoon,fork,knife"},
    ),
    TaskSpec(
        key="scoop_pasta",
        title="Scoop pasta with a tool and transfer it",
        instruction="scoop the {pasta} with the {tool} and transfer it to the container",
        axes={
            "pasta": ("penne", "rigatoni"),
            "tool": ("spoon", "measuring cup"),
        },
        target_kind="scoop_transfer",
        bimanual=True,
        category="granular_tool",
        max_steps=120,
        description="Tool-mediated granular handling: control the tool's contact with a "
        "pile and manage fill level, then transfer to a container.",
    ),
)

SPEC_BY_KEY: dict[str, TaskSpec] = {spec.key: spec for spec in SPECS}
