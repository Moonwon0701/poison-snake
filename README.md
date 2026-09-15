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

### Head-to-head improvements

After Milestone 6, three options were added to the tree to cut down
head-to-head losses. Before that, head-to-heads caused half of all deaths in
four-snake games:

| Option | What it does |
|---|---|
| `lookahead` | Keeps moves that still leave an exit the turn after, away from longer enemy heads |
| `length_race` | Also looks for food while any enemy is at least as long as us |
| `hunt` | Closes in on a shorter snake's head within 2 moves |

`tools/ab_test.py` pits one snake with some options against three snakes
with none, for 2,000 simulated games per combination. With all three on, the
snake won 57.6% of games, against 22.4% with none.

### Trap improvements

With head-to-heads handled, most deaths were traps. A region split at a
narrow gap, usually closed by an enemy, and left the snake on the small side.
Four more options target that:

| Option | What it does |
|---|---|
| `voronoi_pick` | Goes where our territory (cells we reach before any enemy) is biggest |
| `voronoi_gate` | Chases food or prey only through moves whose territory fits our body |
| `timed_area` | Counts cells that bodies will have left by the time we arrive |
| `worst_case_filter` | Keeps moves our body fits whatever nearby enemy heads do |

The test opponents were three snakes with the head-to-head options. Every
combination was screened over 1,000 games, and the best were confirmed on
4,000 fresh seeds:

- `voronoi_pick` + `voronoi_gate` + `timed_area` won 58.1% (95% CI 56.6-59.6%)
  and was trapped in 26.7% of games, against 24.4% and 50.3% without them.
- Adding the worst-case filter did worse, so it stays off.

The default (`DEFAULT_OPTIONS` in `agent/strategy.py`) is the three
head-to-head options plus those three.

```powershell
python tools\ab_test.py "territory:*" --games 1000              # screen every trap combination
python tools\ab_test.py lookahead hunt --baseline none          # against snakes with no options
python tools\ab_test.py default --games 4000 --first-seed 1000  # confirm on fresh seeds
```

### Decision replay viewer

`tools/explain_game.py` turns a game into one HTML file that shows why each
snake moved the way it did. Open the file in any browser; it needs no server.

```powershell
# A game recorded by the CLI (battlesnake play ... -o game.jsonl)
python tools\explain_game.py game.jsonl -o replay.html

# A new game played in the simulator, 1 snake against 3
python tools\explain_game.py --simulate --opponents 3 --seed 7 -o replay.html
```

For each turn, click a snake to see:

- Its four moves as arrows, colored by the filter that dropped them
  (safety, space, escape) or by whether they were chosen, each with a reason.
- Overlays you can switch on and off: danger cells, open area per move, the
  food path, escape cells, and hunt targets.
- The behavior tree, with the nodes that ran and the branch that decided.

Decisions for a CLI log are re-computed with the current code. A random
tie-break can then pick a different move from the recorded one, and the
viewer shows both. Simulated games record exactly the decisions made.
Keyboard shortcuts match the official board viewer: Space, ←/→, q/e, r/t.

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
