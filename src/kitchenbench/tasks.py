"""The 10 KitchenBench tasks, generated from :data:`kitchenbench.specs.SPECS`.

Each task is registered with RoboLens under ``kitchenbench/<key>`` (the slash
namespaces KitchenBench within WorldEvals). The entry-point name, the ``@task``
name, and the returned ``Task.name`` are all identical so the CLI, the registry,
and ``import``-time registration agree.
"""

from __future__ import annotations

import re
from itertools import product

from robolens import Scene, Target, Task, episode_length, task

from kitchenbench.scoring import task_success
from kitchenbench.specs import SPEC_BY_KEY, TaskSpec

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")


def build_scenes(spec: TaskSpec) -> list[Scene]:
    """Build one scene per combination of the spec's variation axes."""
    names = list(spec.axes)
    scenes: list[Scene] = []
    for i, combo in enumerate(product(*(spec.axes[name] for name in names))):
        values = dict(zip(names, combo, strict=True))
        scene_id = "-".join([spec.key, *(_slug(v) for v in combo)])
        scenes.append(
            Scene(
                id=scene_id,
                instruction=spec.instruction.format(**values),
                target=Target(kind=spec.target_kind, spec={**values, **spec.extra}),
                init_seed=i,
                metadata={
                    "task": spec.key,
                    "category": spec.category,
                    "bimanual": spec.bimanual,
                    "version": spec.version,
                    "axes": values,
                },
            )
        )
    return scenes


def make_task(spec: TaskSpec) -> Task:
    """Assemble a RoboLens :class:`~robolens.Task` from a :class:`TaskSpec`."""
    return Task(
        name=f"kitchenbench/{spec.key}",
        scenes=build_scenes(spec),
        scorer=[task_success(), episode_length()],
        max_steps=spec.max_steps,
        metadata={
            "title": spec.title,
            "category": spec.category,
            "bimanual": spec.bimanual,
            "version": spec.version,
            "description": spec.description,
        },
    )


@task("kitchenbench/place_cutlery")
def place_cutlery() -> Task:
    return make_task(SPEC_BY_KEY["place_cutlery"])


@task("kitchenbench/stack")
def stack() -> Task:
    return make_task(SPEC_BY_KEY["stack"])


@task("kitchenbench/place_in_rack")
def place_in_rack() -> Task:
    return make_task(SPEC_BY_KEY["place_in_rack"])


@task("kitchenbench/pour_pasta")
def pour_pasta() -> Task:
    return make_task(SPEC_BY_KEY["pour_pasta"])


@task("kitchenbench/open_container")
def open_container() -> Task:
    return make_task(SPEC_BY_KEY["open_container"])


@task("kitchenbench/fold_cloth")
def fold_cloth() -> Task:
    return make_task(SPEC_BY_KEY["fold_cloth"])


@task("kitchenbench/seal_container")
def seal_container() -> Task:
    return make_task(SPEC_BY_KEY["seal_container"])


@task("kitchenbench/handoff")
def handoff() -> Task:
    return make_task(SPEC_BY_KEY["handoff"])


@task("kitchenbench/sort_cutlery")
def sort_cutlery() -> Task:
    return make_task(SPEC_BY_KEY["sort_cutlery"])


@task("kitchenbench/scoop_pasta")
def scoop_pasta() -> Task:
    return make_task(SPEC_BY_KEY["scoop_pasta"])


#: All task factories, keyed by their bare key (used by ``__init__`` + tests).
TASK_FACTORIES = {
    "place_cutlery": place_cutlery,
    "stack": stack,
    "place_in_rack": place_in_rack,
    "pour_pasta": pour_pasta,
    "open_container": open_container,
    "fold_cloth": fold_cloth,
    "seal_container": seal_container,
    "handoff": handoff,
    "sort_cutlery": sort_cutlery,
    "scoop_pasta": scoop_pasta,
}
