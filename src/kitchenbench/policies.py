"""Mock policies for the KitchenEmbodiment.

- :class:`ScriptedKitchenPolicy` — reads the privileged ``goal_dir`` and drives
  straight to success (the CI oracle / template for a real VLA policy).
- :class:`RandomKitchenPolicy` — random actions; effectively never aligns, so it
  fails. Deterministic given the construction seed.
- :class:`NoopKitchenPolicy` — zero actions; never moves.

All emit :class:`~roboinspect.ActionChunk`\\ s with ``H > 1`` to exercise open-loop
chunk execution.
"""

from __future__ import annotations

import numpy as np
from roboinspect import (
    Action,
    ActionChunk,
    ActionSemantics,
    Box,
    Observation,
    ObservationSpace,
    PolicyConfig,
    PolicyInfo,
    Scene,
)

_ACTION_SPACE = Box(
    shape=(8,),
    semantics=ActionSemantics(control_mode="eef_delta_pos", gripper="continuous", frame="world"),
)
# The scripted oracle needs the privileged goal direction.
_SCRIPTED_OBS = ObservationSpace(state_keys=frozenset({"goal_dir"}))


class ScriptedKitchenPolicy:
    """Drive both arms along the privileged ``goal_dir`` (deterministic success)."""

    def __init__(self, *, chunk_size: int = 4):
        self.chunk_size = chunk_size
        self.num_inferences = 0
        self.info = PolicyInfo(
            name="kitchen_scripted", action_space=_ACTION_SPACE, observation_space=_SCRIPTED_OBS
        )
        self.config = PolicyConfig(action_horizon=chunk_size)

    def reset(self, scene: Scene) -> None:
        self.num_inferences = 0

    def act(self, observation: Observation) -> ActionChunk:
        self.num_inferences += 1
        goal = np.asarray(observation.state["goal_dir"], dtype=np.float64)
        data = np.concatenate([goal, np.array([1.0, 1.0])])  # arms aligned; grippers closed
        return ActionChunk(actions=[Action(data=data.copy()) for _ in range(self.chunk_size)])


class RandomKitchenPolicy:
    """Emit random actions. Deterministic given the construction seed."""

    def __init__(self, *, chunk_size: int = 4, seed: int = 0):
        self.chunk_size = chunk_size
        self.num_inferences = 0
        self._base_seed = seed
        self._reset_count = 0
        self._rng = np.random.RandomState(seed)
        self.info = PolicyInfo(name="kitchen_random", action_space=_ACTION_SPACE)
        self.config = PolicyConfig(action_horizon=chunk_size)

    def reset(self, scene: Scene) -> None:
        self._rng = np.random.RandomState(self._base_seed + self._reset_count)
        self._reset_count += 1
        self.num_inferences = 0

    def act(self, observation: Observation) -> ActionChunk:
        self.num_inferences += 1
        actions = [
            Action(data=self._rng.uniform(-1.0, 1.0, size=8)) for _ in range(self.chunk_size)
        ]
        return ActionChunk(actions=actions)


class NoopKitchenPolicy:
    """Emit zero actions; never moves."""

    def __init__(self, *, chunk_size: int = 1):
        self.chunk_size = chunk_size
        self.num_inferences = 0
        self.info = PolicyInfo(name="kitchen_noop", action_space=_ACTION_SPACE)
        self.config = PolicyConfig(action_horizon=chunk_size)

    def reset(self, scene: Scene) -> None:
        self.num_inferences = 0

    def act(self, observation: Observation) -> ActionChunk:
        self.num_inferences += 1
        return ActionChunk(actions=[Action(data=np.zeros(8)) for _ in range(self.chunk_size)])
