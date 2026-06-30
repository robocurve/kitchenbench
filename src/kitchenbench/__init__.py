"""KitchenBench — a bimanual kitchen-manipulation benchmark for VLA models.

Built on `RoboInspect <https://github.com/robocurve/roboinspect>`_; the first member of
`WorldEvals <https://github.com/robocurve/worldevals>`_. Importing this package
registers all 10 tasks, the mock :class:`~kitchenbench.embodiment.KitchenEmbodiment`,
and the mock policies with the RoboInspect registry (they are also discoverable via
entry points without importing).
"""

from __future__ import annotations

from kitchenbench.distributions import (
    Categorical,
    Constant,
    Distribution,
    Normal,
    Uniform,
)
from kitchenbench.embodiment import KitchenEmbodiment
from kitchenbench.instances import (
    K_EXPERTS,
    K_INSTANCES,
    K_REALIZATIONS,
    Realization,
    TaskInstance,
    Validation,
)
from kitchenbench.policies import (
    NoopKitchenPolicy,
    RandomKitchenPolicy,
    ScriptedKitchenPolicy,
)
from kitchenbench.scoring import task_success
from kitchenbench.specs import SPEC_BY_KEY, SPECS, TaskSpec
from kitchenbench.tasks import (
    TASK_FACTORIES,
    build_scenes,
    fold_cloth,
    handoff,
    make_task,
    open_container,
    place_cutlery,
    place_in_rack,
    pour_pasta,
    realize_scene,
    scoop_pasta,
    seal_container,
    sort_cutlery,
    stack,
)

__version__ = "0.2.0"

__all__ = [
    "K_EXPERTS",
    "K_INSTANCES",
    "K_REALIZATIONS",
    "SPECS",
    "SPEC_BY_KEY",
    "TASK_FACTORIES",
    "Categorical",
    "Constant",
    "Distribution",
    "KitchenEmbodiment",
    "NoopKitchenPolicy",
    "Normal",
    "RandomKitchenPolicy",
    "Realization",
    "ScriptedKitchenPolicy",
    "TaskInstance",
    "TaskSpec",
    "Uniform",
    "Validation",
    "__version__",
    "build_scenes",
    "fold_cloth",
    "handoff",
    "make_task",
    "open_container",
    "place_cutlery",
    "place_in_rack",
    "pour_pasta",
    "realize_scene",
    "scoop_pasta",
    "seal_container",
    "sort_cutlery",
    "stack",
    "task_success",
]
