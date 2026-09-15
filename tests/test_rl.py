"""rl/: masks, the network, a tiny training run, the policy adapter and evaluation.

Skipped when the extras in requirements-rl.txt aren't installed.
"""

import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("sb3_contrib")

import torch  # noqa: E402
from gymnasium import spaces  # noqa: E402

from agent.strategy import Options, decide  # noqa: E402
from agent.world import World  # noqa: E402
from gym_env import rules  # noqa: E402
from gym_env.env import ACTIONS, action_mask, behavior_tree_policy, bt_policy  # noqa: E402
from rl.agent import RLPolicy  # noqa: E402
from rl.envs import make_env  # noqa: E402
from rl.evaluate import evaluate, report  # noqa: E402
from rl.model import SnakeCNN  # noqa: E402
from rl.train import train  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import explain_game  # noqa: E402

TRAPPED_AGENT = [(0, 0), (0, 1), (1, 1), (1, 0), (1, 0)]  # every move is fatal


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """One very short CPU training run, shared by the tests below."""
    runs = tmp_path_factory.mktemp("runs")
    return train(
        "solo", timesteps=512, n_envs=2, device="cpu", subprocess=False,
        runs_dir=runs, eval_every=10**9, eval_games=1,
    )


def test_mask_matches_the_environment_when_some_move_is_allowed():
    env = make_env("solo")
    env.reset(seed=0)
    state = rules.GameState(11, 11, [rules.Snake("agent", [(0, 0), (0, 1), (0, 2)])])
    env.reset(seed=0, options={"state": state})
    assert env.action_masks().tolist() == [False, False, False, True]  # only right


def test_mask_is_never_empty():
    env = make_env("solo")
    state = rules.GameState(11, 11, [rules.Snake("agent", TRAPPED_AGENT)])
    env.reset(seed=0, options={"state": state})
    assert not action_mask(env.unwrapped.state).any()
    assert env.action_masks().all()


def test_bt_policy_plays_with_the_given_options():
    state = rules.new_game(["a", "b"], random.Random(4))
    game_state = rules.to_api_json(state, "a")
    world = World.from_json(game_state)
    assert bt_policy(Options())(game_state, random.Random(1)) == decide(world, random.Random(1), Options())
    assert bt_policy()(game_state, random.Random(1)) == behavior_tree_policy(game_state, random.Random(1))


def test_stage_environments_have_the_right_number_of_opponents():
    for stage, snakes in (("solo", 1), ("duel_weak", 2), ("four_weak", 4)):
        env = make_env(stage)
        env.reset(seed=0)
        assert len(env.unwrapped.state.snakes) == snakes


def test_snake_cnn_keeps_the_batch_and_gives_256_features():
    space = spaces.Box(0.0, 1.0, shape=(6, 11, 11))
    assert SnakeCNN(space)(torch.zeros(3, 6, 11, 11)).shape == (3, 256)


def test_training_saves_a_model_and_a_summary(tiny_model):
    assert tiny_model.exists()
    assert (tiny_model.parent / "stage.json").exists()


def test_rl_policy_picks_an_allowed_move(tiny_model):
    policy = RLPolicy(tiny_model)
    state = rules.GameState(11, 11, [rules.Snake("me", [(0, 0), (0, 1), (0, 2)])], food=[(5, 5)])
    game_state = rules.to_api_json(state, "me")
    explained = policy.explain(game_state)
    assert explained["move"] == "right"  # the only move that isn't a wall or our neck
    assert policy(game_state) == "right"
    assert explained["policy"]["up"] == 0.0
    assert sum(explained["policy"].values()) == pytest.approx(1.0, abs=1e-3)
    assert set(explained["mask"]) == set(ACTIONS)


def test_replay_with_an_rl_snake_explains_it_by_probabilities(tiny_model):
    replay = explain_game.replay_from_simulation(
        seed=2, opponents=1, options=Options(), max_turns=30, agent=RLPolicy(tiny_model)
    )
    assert replay["snakes"][0]["name"] == "snake1 (RL)"
    rl_turns = [frame["decisions"]["snake1"] for frame in replay["turns"] if "snake1" in frame["decisions"]]
    assert rl_turns
    for decision in rl_turns:
        assert decision["actual"] == decision["move"]
        assert set(decision["policy"]) == set(ACTIONS)
        assert "trace" not in decision
    bt_turns = [frame["decisions"]["snake2"] for frame in replay["turns"] if "snake2" in frame["decisions"]]
    assert all("trace" in decision for decision in bt_turns)


def test_evaluate_plays_exactly_the_requested_games(tiny_model):
    stats = evaluate(tiny_model, "duel_weak", games=5, first_seed=0, n_envs=2, device="cpu", subprocess=False)
    assert stats["games"] == 5
    assert 0 <= stats["wins"] <= 5
    assert stats["fair_share"] == 0.5
    assert "win rate" in report(stats)
