"""The deployed brain: numpy weights must play exactly like the trained network.

If these two ever disagree, the snake on the server isn't the snake we
measured. Skipped when the training extras aren't installed.
"""

import importlib
import random
import time

import numpy as np
import pytest

pytest.importorskip("sb3_contrib")

from agent.world import MOVES  # noqa: E402
from boards import make_state  # noqa: E402
from gym_env import rules  # noqa: E402
from rl.agent import RLPolicy  # noqa: E402
from rl.envs import make_env  # noqa: E402
from rl.export import export  # noqa: E402
from rl.model import new_model  # noqa: E402
from rl.numpy_agent import NumpyPolicy  # noqa: E402


@pytest.fixture(scope="module")
def brains(tmp_path_factory):
    """An untrained network saved both ways: random weights exercise the same math."""
    folder = tmp_path_factory.mktemp("weights")
    env = make_env("solo", monitor=False, spatial=True)
    model = new_model(env, device="cpu", seed=0)
    model_path = folder / "model.zip"
    model.save(model_path)
    env.close()
    return RLPolicy(model_path), NumpyPolicy(export(model_path, folder / "snake.npz"))


def positions(games: int = 6, turns: int = 25) -> list[dict]:
    """Game states from a few random four-snake games."""
    rng = random.Random(5)
    seen = []
    for _ in range(games):
        state = rules.new_game([f"s{i}" for i in range(1, 5)], rng)
        for _ in range(turns):
            if not state.snakes:
                break
            seen.append(rules.to_api_json(state, state.snakes[0].id))
            moves = {}
            for snake in state.snakes:
                allowed = rules.non_colliding_moves(state, snake.id)
                moves[snake.id] = allowed[0] if allowed else "up"
            state, _ = rules.step(state, moves, rng)
    return seen


def test_numpy_plays_exactly_like_the_trained_network(brains):
    torch_brain, numpy_brain = brains
    boards = positions()
    assert len(boards) > 100
    for game_state in boards:
        theirs, ours = torch_brain.explain(game_state), numpy_brain.explain(game_state)
        assert ours["move"] == theirs["move"]
        assert ours["policy"] == pytest.approx(theirs["policy"], abs=1e-3)
        assert ours["mask"] == theirs["mask"]


def test_it_never_picks_a_move_that_hits_a_wall_or_a_body(brains):
    _, numpy_brain = brains
    for game_state in positions():
        decision = numpy_brain.explain(game_state)
        assert decision["mask"][decision["move"]]
        assert all(p == 0 for move, p in decision["policy"].items() if not decision["mask"][move])


def test_a_move_takes_well_under_the_engines_time_limit(brains):
    _, numpy_brain = brains
    boards = positions()[:200]
    start = time.perf_counter()
    for game_state in boards:
        numpy_brain(game_state)
    per_move = (time.perf_counter() - start) / len(boards)
    assert per_move < 0.01, f"{per_move * 1000:.1f} ms per move"


def test_the_trapped_board_still_answers(brains):
    _, numpy_brain = brains
    state = rules.GameState(11, 11, [rules.Snake("me", [(0, 0), (0, 1), (1, 1), (1, 0), (1, 0)])])
    assert numpy_brain(rules.to_api_json(state, "me")) in MOVES


def test_the_server_can_play_with_the_network(brains, monkeypatch):
    _, numpy_brain = brains
    monkeypatch.setenv("SNAKE_BRAIN", "rl")
    monkeypatch.setenv("SNAKE_MODEL", numpy_brain.name)
    import server

    try:
        rl_server = importlib.reload(server)
        response = rl_server.app.test_client().post("/move", json=make_state([(5, 5), (5, 4), (5, 3)]))
        assert response.status_code == 200
        assert response.get_json()["move"] in MOVES
    finally:
        monkeypatch.undo()
        importlib.reload(server)  # leave the behavior tree in place for other tests


def test_exported_weights_carry_the_board_shape(brains):
    _, numpy_brain = brains
    assert numpy_brain.channels == 9
    assert numpy_brain.spatial is True
    assert numpy_brain.shape == (9, 11, 11)
    assert np.array_equal(numpy_brain.w["shape"], [9, 11, 11])


def test_the_behavior_tree_takes_over_on_boards_the_network_never_saw(brains, monkeypatch):
    _, numpy_brain = brains
    monkeypatch.setenv("SNAKE_BRAIN", "rl")
    monkeypatch.setenv("SNAKE_MODEL", numpy_brain.name)
    import server

    try:
        rl_server = importlib.reload(server)
        standard = make_state([(5, 5), (5, 4), (5, 3)])
        assert rl_server.network_can_play(standard)

        bigger = make_state([(5, 5), (5, 4), (5, 3)])
        bigger["board"]["width"] = bigger["board"]["height"] = 19
        assert not rl_server.network_can_play(bigger)
        assert rl_server.app.test_client().post("/move", json=bigger).get_json()["move"] in MOVES

        hazardous = make_state([(5, 5), (5, 4), (5, 3)])
        hazardous["board"]["hazards"] = [{"x": 0, "y": 0}]
        assert not rl_server.network_can_play(hazardous)
        assert rl_server.app.test_client().post("/move", json=hazardous).get_json()["move"] in MOVES
    finally:
        monkeypatch.undo()
        importlib.reload(server)
