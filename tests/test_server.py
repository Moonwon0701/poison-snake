"""Smoke tests: the HTTP endpoints respond with what the engine expects."""

from agent.world import MOVES
from boards import make_state
from server import app


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
