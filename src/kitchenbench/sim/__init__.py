"""KitchenBench sim contract: annotated blueprints + measurable success.

See ``plans/0003-sim-support.md``. A physics-sim embodiment:

1. calls :func:`build_blueprint` in ``reset()`` (with the same per-epoch seed
   the ``inspect_robots.eval`` runner hands it),
2. spawns ``blueprint.objects`` from a catalog built against :data:`ASSETS`,
3. builds a checker via :func:`make_success_checker` and, per step, terminates
   with ``termination_reason="success"`` once it fires — the existing
   ``task_success`` scorer needs no changes.
"""

from kitchenbench.sim.blueprint import (
    ASSETS,
    SIM_CONTRACT_VERSION,
    AssetSpec,
    SceneBlueprint,
    SceneObject,
    build_blueprint,
)
from kitchenbench.sim.success import (
    CONTACT_TOL_M,
    CONTAIN_FRACTION,
    FOLD_MAX_HEIGHT_M,
    FOLD_RATIO_PER_FOLD,
    GRASP_RADIUS_M,
    OPEN_FRACTION,
    SCOOP_TOL_G,
    SEAL_TOL_M,
    TRANSFER_FRACTION,
    SuccessChecker,
    Verdict,
    WorldState,
    make_success_checker,
)

__all__ = [
    "ASSETS",
    "CONTACT_TOL_M",
    "CONTAIN_FRACTION",
    "FOLD_MAX_HEIGHT_M",
    "FOLD_RATIO_PER_FOLD",
    "GRASP_RADIUS_M",
    "OPEN_FRACTION",
    "SCOOP_TOL_G",
    "SEAL_TOL_M",
    "SIM_CONTRACT_VERSION",
    "TRANSFER_FRACTION",
    "AssetSpec",
    "SceneBlueprint",
    "SceneObject",
    "SuccessChecker",
    "Verdict",
    "WorldState",
    "build_blueprint",
    "make_success_checker",
]
