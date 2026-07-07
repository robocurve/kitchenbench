"""KitchenBench — a bimanual kitchen-manipulation benchmark for VLA models.

Built on `Inspect Robots <https://github.com/robocurve/inspect-robots>`_; the first member of
`WorldEvals <https://github.com/robocurve/worldevals>`_. Importing this package
registers all 10 tasks with the Inspect Robots registry (via the ``@task``
decorator). The mock :class:`~kitchenbench.embodiment.KitchenEmbodiment` and the
mock policies are not registered on import — Inspect Robots resolves them through
this package's entry points when it is installed (as it does the tasks, so no
import is needed for ``inspect-robots list``).
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

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("kitchenbench")
except PackageNotFoundError:  # pragma: no cover - only hit in a non-installed tree
    __version__ = "0.0.0+unknown"

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
