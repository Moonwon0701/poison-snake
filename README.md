# poison-snake

A [Battlesnake](https://play.battlesnake.com) agent, built as a learning
project for behavior trees, pathfinding, and (later) RL environments.

## Layout

```
server.py            HTTP endpoints (GET /, POST /start /move /end). No strategy.
agent/world.py       World.from_json(game_state): board, my snake, enemies, food
agent/bt.py          hand-written behavior tree nodes (Sequence, Selector, ...)
agent/strategy.py    decide(world) -> "up" | "down" | "left" | "right", as a BT
agent/pathfind.py    shortest_path (BFS) and flood_fill (area counting)
gym_env/rules.py     pure-Python Battlesnake rules for simulation
gym_env/env.py       SnakeEnv: those rules as a Gymnasium environment
tools/               rule check against CLI logs, simulator benchmark
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

At this stage (Milestone 5) the snake's decision is a behavior tree, built in
`agent/strategy.py` from the node classes in `agent/bt.py`. The tree makes
these checks in priority order:

1. **Safety.** It never runs into a wall or a body. It knows a tail moves
   out of the way unless that snake just ate. It avoids cells where an enemy
   of equal or greater length could move its head.
2. **Space.** For each safe move it flood-fills the area it would be left
   in, and skips areas with fewer cells than its body is long.
3. **Food.** At health 50 or below it follows the shortest path to the
   nearest food, starting only with moves that passed steps 1 and 2.
4. **Room.** Otherwise it heads for the side with the most space.

Each `TURN` line in the server log ends with the branch of the tree that
chose the move, for example:

```
TURN 57 -> left [normal turn > pick one > eat > step toward food]
```

## Gymnasium environment (Milestone 6)

`gym_env/` re-implements the standard Battlesnake rules in plain Python and
wraps them as a [Gymnasium](https://gymnasium.farama.org/) environment. It
covers building the environment only; there's no training code.

```python
from gym_env.env import SnakeEnv

env = SnakeEnv(opponents=2)  # 0-3 opponents, played by our behavior tree
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(0)  # 0 up, 1 down, 2 left, 3 right
```

- **Observation:** a float32 array of shape `(6, 11, 11)`, indexed
  `[channel, y, x]`. The channels are my head, my body, enemy heads (1 if at
  least as long as me, else 0.5), enemy bodies, food, and my health / 100.
  Body cells hold (turns until the cell frees up) / length.
- **Reward:** +0.01 per turn survived, +0.1 for eating, −1 for being
  eliminated (draws included), and +1 for outliving every opponent. Change
  any of these with `SnakeEnv(rewards={...})`.
- **`info["action_mask"]`:** 1 for moves that don't hit a wall or a body this turn.
- **Episode end:** `terminated` when our snake is eliminated or wins,
  `truncated` after `max_turns` steps (default 1000).
- **Extras:** `reset(options={"state": ...})` starts from any
  `gym_env.rules.GameState`, and `render_mode="ansi"` prints the board.

To check the simulator:

```powershell
pytest -v
python tools\replay_cli_logs.py <folder of CLI "-o" logs>   # our rules vs the official engine
python tools\sim_benchmark.py 100                           # our snake inside the simulator
```

The rules follow the official engine's order, but use Python's random
numbers. Food placement therefore differs from a CLI game with the same
seed. Hazards and non-standard game modes aren't included.
