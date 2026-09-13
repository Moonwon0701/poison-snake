"""Smoke tests for Milestone 1: endpoints respond, placeholder isn't suicidal."""

from agent.strategy import MOVES, decide
from server import app


def make_state(my_body, other_snakes=(), width=11, height=11):
    """Build a minimal game-state dict. Bodies are lists of (x, y), head first."""

    def snake(snake_id, body):
        return {
            "id": snake_id,
            "head": {"x": body[0][0], "y": body[0][1]},
            "body": [{"x": x, "y": y} for x, y in body],
            "health": 100,
            "length": len(body),
        }

    me = snake("me", my_body)
    others = [snake(f"enemy{i}", b) for i, b in enumerate(other_snakes)]
    return {
        "game": {"id": "test-game"},
        "turn": 1,
        "board": {
            "width": width,
            "height": height,
            "food": [],
            "hazards": [],
            "snakes": [me, *others],
        },
        "you": me,
    }


def test_info_endpoint():
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert response.get_json()["apiversion"] == "1"


def test_move_endpoint_returns_valid_move():
    state = make_state([(5, 5), (5, 4), (5, 3)])
    response = app.test_client().post("/move", json=state)
    assert response.status_code == 200
    assert response.get_json()["move"] in MOVES


def test_start_and_end_endpoints():
    client = app.test_client()
    state = make_state([(5, 5), (5, 4), (5, 3)])
    assert client.post("/start", json=state).status_code == 200
    assert client.post("/end", json=state).status_code == 200


def test_corner_only_one_safe_move():
    # Head in bottom-left corner, body going up: left/down are walls,
    # up is our own neck. Only "right" survives. Moves are random, so repeat.
    state = make_state([(0, 0), (0, 1), (0, 2)])
    for _ in range(50):
        assert decide(state) == "right"


def test_avoids_enemy_body():
    # Our head at (5, 5) with neck below. Enemy body blocks left and right.
    state = make_state(
        [(5, 5), (5, 4), (5, 3)],
        other_snakes=[[(4, 6), (4, 5), (4, 4)], [(6, 6), (6, 5), (6, 4)]],
    )
    for _ in range(50):
        assert decide(state) == "up"


def test_trapped_still_returns_a_move():
    # Nothing is safe; we must still answer with a legal string, not crash.
    state = make_state(
        [(0, 0), (0, 1), (1, 1), (1, 0)],
    )
    assert decide(state) in MOVES
