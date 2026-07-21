"""The 10 KitchenBench tasks, generated from :data:`kitchenbench.specs.SPECS`.

Each task instance (a stochastic setup + goal) becomes one Inspect Robots ``Scene``;
each task runs ``K_REALIZATIONS`` (5) realizations per instance via
``Epochs(count=5, reducer="mean")``, so the per-scene reduced ``task_success`` is
the methodology's instance success probability P̂[Yᵢ=1]. With ``K_INSTANCES`` (5)
instances per task that is 5 scenes x 5 epochs per task.

Each task is registered with Inspect Robots under ``kitchenbench/<key>`` (the slash
namespaces KitchenBench within WorldEvals). The entry-point name, the ``@task``
name, and the returned ``Task.name`` are all identical.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import asdict
from typing import Any

from inspect_robots import Epochs, Scene, Target, Task, episode_length, task
from inspect_robots.rollout import derive_seed

from kitchenbench.instances import K_INSTANCES, K_REALIZATIONS, Realization, TaskInstance
from kitchenbench.scoring import task_success
from kitchenbench.specs import SPEC_BY_KEY, TaskSpec

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")


def _validation_dict(spec_validation: Any) -> dict[str, Any]:
    """``asdict`` a Validation, casting rating tuples to lists for clean JSON."""
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(spec_validation).items()}


def build_scenes(spec: TaskSpec) -> list[Scene]:
    """Build one Scene per task instance (5 per task)."""
    scenes: list[Scene] = []
    for index, inst in enumerate(spec.instances):
        # Derive the scene seed from the instance id, not its sorted position.
        #
        # instance_id is "<task>/<name>" and is unique across the suite, so this
        # decorrelates the per-instance RNG streams. Seeding on `index` gave
        # instance i of EVERY task a byte-identical PCG64 stream (derive_seed
        # hashes only (eval_seed, scene_seed, epoch) -- task identity never
        # entered the payload), so same-shaped distributions at the same sorted
        # position realized the SAME number across tasks.
        scene_seed = zlib.crc32(inst.instance_id.encode()) & 0xFFFFFFFF
        # The displayed instruction is the realization of epoch 0 under the default
        # eval(seed=0), so it equals the first actual rollout's instruction. It has
        # to use the same scene seed the Scene carries, or the two drift apart.
        canonical = inst.realize(derive_seed(0, scene_seed, 0))
        scenes.append(
            Scene(
                id=_slug(inst.instance_id),
                instruction=canonical.instruction,
                target=Target(kind=inst.target_kind, spec=dict(inst.static)),
                init_seed=scene_seed,
                metadata={
                    "benchmark": "kitchenbench",
                    "task": spec.key,
                    "category": spec.category,
                    "bimanual": spec.bimanual,
                    "version": spec.version,
                    "instance_id": inst.instance_id,
                    "instance_index": index,
                    "setup": inst.setup_spec(),
                    "language_vars": list(inst.language_vars),
                    "validation": _validation_dict(inst.validation),
                },
            )
        )
    return scenes


def find_instance(scene: Scene) -> TaskInstance:
    """Recover the task instance behind a Scene (by stable ``instance_id``).

    A scene logged under an older task set either resolves to the same instance
    or fails loudly, never silently to a different one.
    """
    key = str(scene.metadata["task"])
    instance_id = str(scene.metadata["instance_id"])
    for inst in SPEC_BY_KEY[key].instances:
        if inst.instance_id == instance_id:
            return inst
    raise LookupError(f"no instance {instance_id!r} in task {key!r} (removed or renamed?)")


def realize_scene(scene: Scene, seed: int | None) -> Realization:
    """Realize the instance behind a Scene for ``seed``.

    The seam a real embodiment / operator tool uses to get the concrete setup
    for a given realization. ``seed=None`` (direct ``reset(scene)`` calls) is
    guarded to ``0`` for determinism.
    """
    return find_instance(scene).realize(seed if seed is not None else 0)


def make_task(spec: TaskSpec) -> Task:
    """Assemble a Inspect Robots :class:`~inspect_robots.Task` from a :class:`TaskSpec`."""
    return Task(
        name=f"kitchenbench/{spec.key}",
        scenes=build_scenes(spec),
        scorer=[task_success(), episode_length()],
        max_steps=spec.max_steps,
        epochs=Epochs(count=K_REALIZATIONS, reducer="mean"),
        metadata={
            "title": spec.title,
            "category": spec.category,
            "bimanual": spec.bimanual,
            "version": spec.version,
            "description": spec.description,
            "max_seconds": spec.max_seconds,
            "k_instances": K_INSTANCES,
            "k_realizations": K_REALIZATIONS,
        },
    )


@task("kitchenbench/place_cutlery")
def place_cutlery() -> Task:
    """Build the registered cutlery placement task from its stochastic instances."""
    return make_task(SPEC_BY_KEY["place_cutlery"])


@task("kitchenbench/stack")
def stack() -> Task:
    """Build the registered dishware stacking task from its stochastic instances."""
    return make_task(SPEC_BY_KEY["stack"])


@task("kitchenbench/place_in_rack")
def place_in_rack() -> Task:
    """Build the registered rack insertion task from its stochastic instances."""
    return make_task(SPEC_BY_KEY["place_in_rack"])


@task("kitchenbench/pour_pasta")
def pour_pasta() -> Task:
    """Build the registered granular pouring task from its stochastic instances."""
    return make_task(SPEC_BY_KEY["pour_pasta"])


@task("kitchenbench/open_container")
def open_container() -> Task:
    """Build the registered container opening task from its stochastic instances."""
    return make_task(SPEC_BY_KEY["open_container"])


@task("kitchenbench/fold_cloth")
def fold_cloth() -> Task:
    """Build the registered cloth folding task from its stochastic instances."""
    return make_task(SPEC_BY_KEY["fold_cloth"])


@task("kitchenbench/seal_container")
def seal_container() -> Task:
    """Build the registered lid sealing task from its stochastic instances."""
    return make_task(SPEC_BY_KEY["seal_container"])


@task("kitchenbench/handoff")
def handoff() -> Task:
    """Build the registered two-arm handoff task from its stochastic instances."""
    return make_task(SPEC_BY_KEY["handoff"])


@task("kitchenbench/sort_cutlery")
def sort_cutlery() -> Task:
    """Build the registered cutlery sorting task from its stochastic instances."""
    return make_task(SPEC_BY_KEY["sort_cutlery"])


@task("kitchenbench/scoop_pasta")
def scoop_pasta() -> Task:
    """Build the registered measured scooping task from its stochastic instances."""
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
