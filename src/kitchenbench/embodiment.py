"""KitchenEmbodiment — a dependency-free abstract bimanual mock kitchen.

It does **not** simulate physics; like RoboLens's ``CubePick`` it models *progress
toward the scene goal* so the whole pipeline (scenes → chunked rollout → score →
log) runs in CI with no hardware. The value of KitchenBench is the task
definitions; this mock exists to exercise them and to serve as the template for a
real YAM-arms embodiment.

Each reset draws a hidden unit "goal direction" in the 6-D dual-arm command
subspace. Progress advances only when a normalized action aligns with that
direction (cosine ≥ ``align_threshold``) — so the scripted oracle (which reads the
privileged ``goal_dir``) succeeds, while random/no-op policies do not.

The action space is bimanual on purpose: ``(8,)`` = two arms x ``[dx, dy, dz,
gripper]``. A real YAM embodiment is a higher-DoF analog (compatibility checking
matches exact dims, so the real arm pairs with a real VLA, not with this mock).
"""

from __future__ import annotations

import numpy as np
from robolens import (
    Action,
    ActionSemantics,
    Box,
    CameraSpec,
    EmbodimentInfo,
    Observation,
    ObservationSpace,
    Scene,
    StepResult,
)
from robolens.embodiment import PRIVILEGED_SUCCESS, RENDERABLE, SEEDABLE

from kitchenbench.tasks import realize_scene

_IMG = 24

_ACTION_SPACE = Box(
    shape=(8,),
    low=np.full(8, -1.0),
    high=np.full(8, 1.0),
    semantics=ActionSemantics(control_mode="eef_delta_pos", gripper="continuous", frame="world"),
)


class KitchenEmbodiment:
    """Abstract bimanual mock kitchen world."""

    def __init__(
        self,
        *,
        step_size: float = 0.25,
        align_threshold: float = 0.99,
        goal_threshold: float = 1.0,
    ):
        self.step_size = step_size
        self.align_threshold = align_threshold
        self.goal_threshold = goal_threshold
        self.num_steps = 0

        self._progress = 0.0
        self._goal = np.zeros(6, dtype=np.float64)
        self._last = np.zeros(8, dtype=np.float64)
        self._instruction: str | None = None
        self._rng = np.random.RandomState(0)

        self.info = EmbodimentInfo(
            name="kitchen",
            action_space=_ACTION_SPACE,
            observation_space=ObservationSpace(
                cameras=(CameraSpec(name="overhead", height=_IMG, width=_IMG, channels=3),),
                state_keys=frozenset({"progress", "goal_dir", "left_eef", "right_eef"}),
            ),
            control_hz=10.0,
            is_simulated=True,
            capabilities=frozenset({SEEDABLE, RENDERABLE, PRIVILEGED_SUCCESS}),
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        self._rng = np.random.RandomState(seed if seed is not None else 0)
        goal = self._rng.normal(size=6)
        self._goal = goal / (np.linalg.norm(goal) or 1.0)
        self._progress = 0.0
        self._last = np.zeros(8, dtype=np.float64)
        self.num_steps = 0
        # For a KitchenBench scene (the "benchmark" marker keeps scenes from other
        # plugins out), realize the task instance for this seed so the observed
        # instruction reflects the per-epoch realization (a real embodiment /
        # operator would also arrange the sampled setup). The realization rng is
        # independent of the goal_dir rng above. Bare scenes fall back unchanged.
        # The instruction is carried on every subsequent observation — real VLA
        # policies re-condition on it at each act() call.
        self._instruction = scene.instruction
        if scene.metadata.get("benchmark") == "kitchenbench":
            self._instruction = realize_scene(scene, seed).instruction
        return self._observe()

    def step(self, action: Action) -> StepResult:
        self.num_steps += 1
        data = np.clip(np.asarray(action.data, dtype=np.float64), -1.0, 1.0)
        self._last = data
        arm = data[:6]
        norm = float(np.linalg.norm(arm))
        if norm > 0.0:
            cosine = float(np.dot(arm / norm, self._goal))
            if cosine >= self.align_threshold:
                self._progress = min(self.goal_threshold, self._progress + self.step_size)
        success = self._progress >= self.goal_threshold
        return StepResult(
            observation=self._observe(),
            reward=self._progress - 1.0,
            terminated=success,
            termination_reason="success" if success else None,
            truncated=False,
            info={"success": success, "progress": self._progress},
        )

    def close(self) -> None:
        return None

    def _observe(self) -> Observation:
        return Observation(
            images={"overhead": self._render()},
            state={
                "progress": np.array([self._progress], dtype=np.float64),
                "goal_dir": self._goal.copy(),
                "left_eef": self._last[:3].copy(),
                "right_eef": self._last[3:6].copy(),
            },
            instruction=self._instruction,
        )

    def _render(self) -> np.ndarray:
        img = np.zeros((_IMG, _IMG, 3), dtype=np.uint8)
        filled = round(self._progress * _IMG)
        img[_IMG - filled :, :, 1] = 200  # a green progress bar rising from the bottom
        return img
