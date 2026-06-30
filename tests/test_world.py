"""Mock embodiment, policies, scoring, and end-to-end evaluation."""

from __future__ import annotations

import numpy as np
import pytest
from roboinspect import Action, eval
from roboinspect.rollout import TrialRecord
from roboinspect.scene import Scene

from kitchenbench.embodiment import KitchenEmbodiment
from kitchenbench.policies import (
    NoopKitchenPolicy,
    RandomKitchenPolicy,
    ScriptedKitchenPolicy,
)
from kitchenbench.scoring import task_success
from kitchenbench.tasks import TASK_FACTORIES, place_cutlery

_SCENE = Scene(id="s", instruction="do the thing", init_seed=0)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def _record(*, reason: str | None = None, operator: str | None = None) -> TrialRecord:
    rec = TrialRecord(scene_id="s", epoch=0, seed=0)
    rec.termination_reason = reason
    rec.operator_judgement = operator
    return rec


def test_task_success_branches() -> None:
    scorer = task_success()
    assert scorer(_record(reason="success"), None).value is True
    assert scorer(_record(operator="success"), None).value is True
    assert scorer(_record(operator="fail"), None).value is False
    assert scorer(_record(), None).value is False


# --------------------------------------------------------------------------- #
# embodiment
# --------------------------------------------------------------------------- #
def test_embodiment_reset_exposes_goal() -> None:
    emb = KitchenEmbodiment()
    obs = emb.reset(_SCENE, seed=3)
    assert "goal_dir" in obs.state and obs.state["goal_dir"].shape == (6,)
    assert np.isclose(np.linalg.norm(obs.state["goal_dir"]), 1.0)
    assert obs.state["progress"][0] == 0.0
    assert obs.images["overhead"].shape == (24, 24, 3)


def test_embodiment_aligned_action_makes_progress_to_success() -> None:
    emb = KitchenEmbodiment()
    obs = emb.reset(_SCENE, seed=1)
    goal = obs.state["goal_dir"]
    aligned = Action(data=np.concatenate([goal, [1.0, 1.0]]))
    results = [emb.step(aligned) for _ in range(4)]
    assert results[-1].terminated is True
    assert results[-1].termination_reason == "success"
    assert results[-1].info["success"] is True
    assert emb.num_steps == 4


def test_embodiment_zero_and_misaligned_actions_make_no_progress() -> None:
    emb = KitchenEmbodiment()
    obs = emb.reset(_SCENE, seed=2)
    zero = emb.step(Action(data=np.zeros(8)))  # norm 0 branch
    assert zero.info["progress"] == 0.0
    opposite = Action(data=np.concatenate([-obs.state["goal_dir"], [0.0, 0.0]]))
    assert emb.step(opposite).info["progress"] == 0.0
    assert not emb.step(opposite).terminated


def test_embodiment_close_and_render_bar() -> None:
    emb = KitchenEmbodiment()
    obs = emb.reset(_SCENE, seed=1)
    for _ in range(2):
        emb.step(Action(data=np.concatenate([obs.state["goal_dir"], [0.0, 0.0]])))
    assert emb._render().sum() > 0  # progress bar has green pixels
    assert emb.close() is None


# --------------------------------------------------------------------------- #
# policies
# --------------------------------------------------------------------------- #
def test_scripted_policy_aligns_with_goal() -> None:
    emb = KitchenEmbodiment()
    obs = emb.reset(_SCENE, seed=5)
    policy = ScriptedKitchenPolicy(chunk_size=4)
    policy.reset(_SCENE)
    chunk = policy.act(obs)
    assert len(chunk) == 4
    assert chunk.actions[0].data.shape == (8,)
    assert np.allclose(chunk.actions[0].data[:6], obs.state["goal_dir"])
    assert policy.num_inferences == 1


def test_random_and_noop_policies() -> None:
    obs = KitchenEmbodiment().reset(_SCENE, seed=0)
    rnd = RandomKitchenPolicy(chunk_size=3, seed=7)
    rnd.reset(_SCENE)
    chunk = rnd.act(obs)
    assert len(chunk) == 3 and chunk.actions[0].data.shape == (8,)
    noop = NoopKitchenPolicy()
    noop.reset(_SCENE)
    assert np.all(noop.act(obs).actions[0].data == 0.0)


# --------------------------------------------------------------------------- #
# end to end over all 10 tasks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", list(TASK_FACTORIES), ids=list(TASK_FACTORIES))
def test_scripted_solves_every_task(key: str, tmp_path: object) -> None:
    task = TASK_FACTORIES[key]()
    (log,) = eval(task, ScriptedKitchenPolicy(), KitchenEmbodiment(), log_dir=str(tmp_path))
    assert log.status == "success"
    assert log.results.metrics["task_success"] == 1.0


def test_random_and_noop_do_not_solve(tmp_path: object) -> None:
    task = place_cutlery()
    (rnd,) = eval(task, RandomKitchenPolicy(), KitchenEmbodiment(), log_dir=str(tmp_path), seed=0)
    assert rnd.results.metrics["task_success"] < 1.0
    (noop,) = eval(task, NoopKitchenPolicy(), KitchenEmbodiment(), log_dir=str(tmp_path))
    assert noop.results.metrics["task_success"] == 0.0


def test_run_via_registry_strings(tmp_path: object) -> None:
    (log,) = eval("kitchenbench/handoff", "kitchen_scripted", "kitchen", log_dir=str(tmp_path))
    assert log.status == "success"
    assert log.results.metrics["task_success"] == 1.0
