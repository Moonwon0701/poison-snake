"""HTTP layer for the Battlesnake API: https://docs.battlesnake.com/api

Deliberately thin: read the JSON the game engine sends, hand it to the
agent, send back JSON. No strategy lives here.
"""

import logging
import os

from flask import Flask, jsonify, request

from agent.strategy import decide
from agent.world import World

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("snake")

app = Flask(__name__)


@app.get("/")
def info():
    # The engine calls this to learn our API version and how to draw us.
    return jsonify(
        {
            "apiversion": "1",
            "author": "",
            "color": "#7b2cbf",
            "head": "default",
            "tail": "default",
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
    chosen = decide(World.from_json(game_state))
    log.info("TURN %s -> %s", game_state["turn"], chosen)
    return jsonify({"move": chosen})


@app.post("/end")
def end():
    game_state = request.get_json()
    log.info("GAME OVER %s", game_state["game"]["id"])
    return "ok"


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    app.run(host=host, port=port)
