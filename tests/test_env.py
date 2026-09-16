"""SnakeEnv: the Gymnasium interface, observations, rewards and episode ends."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from gym_env import rules
from gym_env.env import (
    ACTIONS,
    AGENT,
    ENEMY_BODIES,
    ENEMY_HEADS,
    FOOD,
    MY_BODY,
    MY_HEAD,
    MY_HEALTH,
    MY_REACH,
    OWNER,
    RIVAL_REACH,
    SnakeEnv,
    action_mask,
    encode_observation,
    territory_share,
)


def board(snakes, food=(), health=None, size=11):
    health = health or {}
    return rules.GameState(
        size,
        size,
        [rules.Snake(sid, list(body), health.get(sid, 100)) for sid, body in snakes.items()],
        list(food),
    )


def always(move):
    """An opponent policy that makes the same move every turn."""
    return lambda game_state, rng: move


# --- Gymnasium interface ----------------------------------------------------


@pytest.mark.parametrize("opponents", [0, 3])
def test_passes_the_gymnasium_env_checker(opponents):
    check_env(SnakeEnv(opponents=opponents), skip_render_check=True)


def test_reset_gives_an_observation_in_the_space_and_a_mask():
    env = SnakeEnv(opponents=2)
    obs, info = env.reset(seed=0)
    assert obs.shape == (6, 11, 11)
    assert env.observation_space.contains(obs)
    assert info["action_mask"].shape == (4,)
    assert info["turn"] == 0


def test_the_same_seed_replays_the_same_game_with_opponents():
    def play(seed):
        env = SnakeEnv(opponents=2)
        obs, info = env.reset(seed=seed)
        frames = [obs]
        for _ in range(60):
            allowed = np.flatnonzero(info["action_mask"])
            obs, _, terminated, truncated, info = env.step(int(allowed[0]) if len(allowed) else 0)
            frames.append(obs)
            if terminated or truncated:
                break
        return frames

    first, second = play(3), play(3)
    assert len(first) == len(second)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_gymnasium_make_finds_the_registered_env():
    import gymnasium

    import gym_env  # noqa: F401  (registers PoisonSnake-v0)

    env = gymnasium.make("PoisonSnake-v0", opponents=1)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (6, 11, 11)


# --- observation and mask ---------------------------------------------------


def test_observation_channels():
    s = board(
        {
            AGENT: [(5, 5), (5, 4), (5, 3), (5, 3)],  # length 4, just ate
            "long": [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6)],  # length 5
            "short": [(8, 8), (8, 9)],  # length 2
        },
        food=[(0, 0)],
        health={AGENT: 50},
    )
    obs = encode_observation(s)
    assert obs[MY_HEAD, 5, 5] == 1 and obs[MY_HEAD].sum() == 1
    assert obs[MY_BODY, 4, 5] == pytest.approx(3 / 4)  # neck
    assert obs[MY_BODY, 3, 5] == pytest.approx(2 / 4)  # doubled tail keeps the larger value
    assert obs[ENEMY_HEADS, 2, 2] == 1.0  # at least as long as us
    assert obs[ENEMY_HEADS, 8, 8] == 0.5  # shorter than us
    assert obs[ENEMY_BODIES, 3, 2] == pytest.approx(4 / 5)
    assert obs[ENEMY_BODIES, 6, 2] == pytest.approx(1 / 5)
    assert obs[FOOD, 0, 0] == 1 and obs[FOOD].sum() == 1
    assert np.all(obs[MY_HEALTH] == 0.5)


def test_action_mask_rules_out_walls_and_bodies():
    s = board({AGENT: [(0, 0), (0, 1), (0, 2)]})
    assert action_mask(s).tolist() == [0, 0, 0, 1]  # up, down, left, right


# --- rewards and episode ends -----------------------------------------------


def test_dying_ends_the_episode_with_the_death_penalty():
    env = SnakeEnv()
    env.reset(seed=0, options={"state": board({AGENT: [(0, 0), (0, 1), (0, 2)]})})
    _, reward, terminated, truncated, info = env.step(ACTIONS.index("left"))
    assert reward == -1.0
    assert terminated and not truncated
    assert info["cause"] == rules.WALL_COLLISION


def test_surviving_and_eating_are_rewarded():
    env = SnakeEnv()
    start = board({AGENT: [(5, 5), (5, 4), (5, 3)]}, food=[(5, 6)], health={AGENT: 50})
    env.reset(seed=0, options={"state": start})
    _, reward, terminated, _, _ = env.step(ACTIONS.index("up"))
    assert reward == pytest.approx(0.01 + 0.1)
    assert not terminated
    _, reward, _, _, _ = env.step(ACTIONS.index("up"))
    assert reward == pytest.approx(0.01)


def test_outliving_the_last_opponent_is_a_win():
    env = SnakeEnv(opponent_policy=always("left"))
    start = board({AGENT: [(5, 5), (5, 4), (5, 3)], "opponent1": [(0, 8), (1, 8), (2, 8)]})
    env.reset(seed=0, options={"state": start})
    _, reward, terminated, _, info = env.step(ACTIONS.index("up"))
    assert reward == pytest.approx(0.01 + 1.0)
    assert terminated and info["won"]


def test_a_draw_counts_as_death():
    env = SnakeEnv(opponent_policy=always("left"))
    start = board({AGENT: [(4, 5), (3, 5), (2, 5)], "opponent1": [(6, 5), (7, 5), (8, 5)]})
    env.reset(seed=0, options={"state": start})
    _, reward, terminated, _, info = env.step(ACTIONS.index("right"))
    assert reward == -1.0
    assert terminated and not info["won"]
    assert info["cause"] == rules.HEAD_COLLISION


def test_episodes_are_truncated_after_max_turns():
    env = SnakeEnv(max_turns=3)
    obs, info = env.reset(seed=0, options={"state": board({AGENT: [(5, 5), (5, 4), (5, 3)]})})
    for turn in range(3):
        allowed = np.flatnonzero(info["action_mask"])
        _, _, terminated, truncated, info = env.step(int(allowed[0]))
        assert not terminated
        assert truncated == (turn == 2)


def test_reward_weights_can_be_changed():
    env = SnakeEnv(rewards={"survive": 0.5})
    env.reset(seed=0, options={"state": board({AGENT: [(5, 5), (5, 4), (5, 3)]})})
    _, reward, _, _, _ = env.step(ACTIONS.index("up"))
    assert reward == 0.5
    assert env.rewards["death"] == -1.0  # other weights keep their defaults


def test_territory_is_the_whole_board_when_we_are_alone():
    state = board({AGENT: [(5, 5), (5, 4), (5, 3)]})
    assert territory_share(state) == 1.0


def test_territory_splits_between_two_snakes_and_favours_the_freer_side():
    # Both snakes lie on row 5; ours on the left, theirs on the right.
    even = board({AGENT: [(2, 5), (1, 5), (0, 5)], "o": [(8, 5), (9, 5), (10, 5)]})
    assert 0.3 < territory_share(even) < 0.7

    # Now ours sits in the far corner, so the enemy reaches most cells first.
    cornered = board({AGENT: [(0, 0), (1, 0), (2, 0)], "o": [(5, 5), (5, 6), (5, 7)]})
    assert territory_share(cornered) < territory_share(even)


def test_the_territory_reward_pays_for_holding_space():
    state = board({AGENT: [(5, 5), (5, 4), (5, 3)], "o": [(9, 9), (9, 10), (8, 10)]})
    env = SnakeEnv(opponents=1, rewards={"territory": 1.0}, opponent_policy=always("down"))
    env.reset(seed=0, options={"state": state})
    _, reward, _, _, _ = env.step(ACTIONS.index("up"))
    share = territory_share(env.state)
    assert reward == pytest.approx(0.01 + share)
    assert 0 < share < 1


def test_space_channels_mark_who_reaches_each_cell_first():
    state = board({AGENT: [(1, 5), (0, 5), (0, 4)], "o": [(9, 5), (10, 5), (10, 4)]})
    obs = encode_observation(state, spatial=True)
    assert obs.shape == (9, 11, 11)

    near_me, near_them = (2, 5), (8, 5)
    assert obs[OWNER][near_me[1]][near_me[0]] == 1.0
    assert obs[OWNER][near_them[1]][near_them[0]] == 0.0
    assert obs[MY_REACH][near_me[1]][near_me[0]] > obs[MY_REACH][near_them[1]][near_them[0]]
    assert obs[RIVAL_REACH][near_them[1]][near_them[0]] > obs[RIVAL_REACH][near_me[1]][near_me[0]]
    # The middle column is the same distance from both, so it's a tie.
    assert obs[OWNER][5][5] == 0.5


def test_space_channels_are_off_by_default():
    state = board({AGENT: [(5, 5), (5, 4), (5, 3)]})
    assert encode_observation(state).shape == (6, 11, 11)
    assert SnakeEnv(spatial=True).observation_space.shape == (9, 11, 11)
    assert SnakeEnv().observation_space.shape == (6, 11, 11)


def test_stepping_after_the_episode_ended_is_an_error():
    env = SnakeEnv()
    env.reset(seed=0, options={"state": board({AGENT: [(0, 0), (0, 1), (0, 2)]})})
    env.step(ACTIONS.index("left"))
    with pytest.raises(RuntimeError):
        env.step(0)


def test_ansi_render_draws_the_board():
    env = SnakeEnv(render_mode="ansi")
    env.reset(seed=0, options={"state": board({AGENT: [(0, 0), (1, 0), (2, 0)]}, food=[(10, 10)])})
    lines = env.render().splitlines()
    assert lines[0] == "turn 0, health 100"
    assert lines[1] == ". . . . . . . . . . *"  # top row, y = 10
    assert lines[-1] == "A a a . . . . . . . ."  # bottom row, y = 0
