# poison-snake

A [Battlesnake](https://play.battlesnake.com) agent, built as a learning
project for behavior trees, pathfinding, and (later) RL environments.

## Layout

```
server.py            HTTP endpoints (GET /, POST /start /move /end). No strategy.
agent/world.py       World.from_json(game_state): board, my snake, enemies, food
agent/strategy.py    decide(world) -> "up" | "down" | "left" | "right"
agent/pathfind.py    shortest_path (BFS) and flood_fill (area counting)
tests/               pytest tests
```

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `Activate.ps1` is blocked, run this once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

On macOS/Linux, use `python3 -m venv .venv` and `source .venv/bin/activate` instead.

## Run the tests

```powershell
pytest -v
```

All tests should pass.

## Run the snake server

```powershell
python server.py
```

The server runs on [waitress](https://docs.pylonsproject.org/projects/waitress/)
rather than Flask's development server. Waitress keeps connections open
between turns, so running many CLI games back-to-back doesn't use up local
ports. It listens on `http://127.0.0.1:8000`. You can change this with the
`PORT` and `HOST` environment variables. For a quick check, open
http://127.0.0.1:8000/ in a browser. You should see JSON containing
`"apiversion": "1"`.

## Play a local game with the Battlesnake CLI

1. Download the CLI from
   https://github.com/BattlesnakeOfficial/rules/releases (on Windows, pick
   `battlesnake_<version>_Windows_x86_64.tar.gz`). Extract it with
   `tar -xzf <file>.tar.gz` and put `battlesnake.exe` on your PATH or in this
   folder. If you have Go installed, you can use this instead:
   `go install github.com/BattlesnakeOfficial/rules/cli/battlesnake@latest`.
2. Leave the server running in one terminal. In a second terminal, run:

```powershell
# Solo game, with the board printed in the terminal each turn
battlesnake play -W 11 -H 11 --name poison --url http://127.0.0.1:8000 -g solo -v

# Same game, shown in the web board viewer
battlesnake play -W 11 -H 11 --name poison --url http://127.0.0.1:8000 -g solo --browser

# Our snake against itself (standard rules, 2 snakes)
battlesnake play --name A --url http://127.0.0.1:8000 --name B --url http://127.0.0.1:8000
```

The CLI prints `Game completed after N turns.` at the end (in a multi-snake
game it also names the winner). The
server terminal logs `GAME START`, one `TURN n -> move` line per turn, and
`GAME OVER`.

At this stage (Milestone 4) the snake decides in priority order:

1. **Safety.** It never runs into a wall or a body. It knows a tail moves
   out of the way unless that snake just ate. It avoids cells where an enemy
   of equal or greater length could move its head.
2. **Space.** For each safe move it flood-fills the area it would be left
   in, and skips areas with fewer cells than its body is long.
3. **Food.** At health 50 or below it follows the shortest path to the
   nearest food, starting only with moves that passed steps 1 and 2.
4. **Room.** Otherwise it heads for the side with the most space.

Milestone 5 restructures this as an explicit behavior tree.
