"""HTTP layer for the Battlesnake API: https://docs.battlesnake.com/api

Deliberately thin: read the JSON the game engine sends, hand it to the
agent, send back JSON. No strategy lives here.

Environment variables:
    SNAKE_BRAIN   "bt" (default) for the behavior tree, "rl" for a trained network
    SNAKE_MODEL   weights for the "rl" brain (default models/snake.npz)
    SNAKE_AUTHOR  your Battlesnake username, shown on the profile
    SNAKE_COLOR   the snake's color, e.g. "#7b2cbf"
    SNAKE_HEAD    head style from https://play.battlesnake.com/customizations
    SNAKE_TAIL    tail style, likewise
    HOST, PORT    where to listen (0.0.0.0 and the platform's PORT when deployed)
"""

import logging
import os

from flask import Flask, jsonify, request
from waitress import serve

from agent.strategy import decide_with_trace
from agent.world import World

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("snake")

app = Flask(__name__)

# SNAKE_BRAIN=rl plays with a trained network instead of the behavior tree.
# The weights are a .npz from rl/export.py, and numpy runs them, so a
# deployment needs no PyTorch. Still no strategy in this file: both brains
# take the game state and hand back a move.
BRAIN = os.environ.get("SNAKE_BRAIN", "bt")
MODEL = os.environ.get("SNAKE_MODEL", "models/snake.npz")
_network = None


def network():
    global _network
    if _network is None:
        from rl.numpy_agent import NumpyPolicy

        _network = NumpyPolicy(MODEL)
        log.info("brain: %s", _network.name)
    return _network


if BRAIN == "rl":  # fail at startup, not on the first move of a real game
    network()
elif BRAIN != "bt":
    raise SystemExit(f'SNAKE_BRAIN must be "bt" or "rl", not "{BRAIN}"')


@app.get("/")
def info():
    # The engine calls this to learn our API version and how to draw us.
    return jsonify(
        {
            "apiversion": "1",
            "author": os.environ.get("SNAKE_AUTHOR", ""),
            "color": os.environ.get("SNAKE_COLOR", "#7b2cbf"),
            "head": os.environ.get("SNAKE_HEAD", "default"),
            "tail": os.environ.get("SNAKE_TAIL", "default"),
            "version": "0.1.0",
        }
    )


@app.post("/start")
def start():
    game_state = request.get_json()
    log.info("GAME START %s", game_state["game"]["id"])
    return "ok"


@app.post("/move")
def move():
    game_state = request.get_json()
    if BRAIN == "rl":
        decision = network().explain(game_state)
        chosen = decision["move"]
        why = " ".join(f"{m} {p:.0%}" for m, p in decision["policy"].items() if p)
    else:
        chosen, trace = decide_with_trace(World.from_json(game_state))
        why = " > ".join(trace[1:])  # trace[0] is the root node
    log.info("TURN %s -> %s [%s]", game_state["turn"], chosen, why)
    return jsonify({"move": chosen})


@app.post("/end")
def end():
    game_state = request.get_json()
    log.info("GAME OVER %s", game_state["game"]["id"])
    return "ok"


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    # Waitress instead of Flask's built-in dev server. The dev server closes
    # the connection after every response, so the game engine opens a new
    # TCP connection every turn. On Windows each closed connection holds a
    # local port for ~2 minutes, and long batches of CLI games run out of
    # ports. Waitress keeps connections open so the engine can reuse them.
    serve(app, host=host, port=port)
