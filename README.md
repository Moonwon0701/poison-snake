English | [한국어](README.ko.md)

# poison-snake

A [Battlesnake](https://play.battlesnake.com) agent, built as a learning
project for behavior trees, pathfinding, and (later) RL environments.

The Korean page is a shorter tour: what's here, how to run it, and the
behavior-tree and RL ideas worth reusing in a hackathon.

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
- **Territory reward (off by default):** `rewards={"territory": 0.02}` also
  pays, every turn, that much times `territory_share(state)` — the fraction
  of the free board our head reaches strictly before any other head (our
  Voronoi area). It shrinks several turns before a snake is trapped, so it
  rewards keeping space early. It costs two BFS passes, about 0.1 ms a step.
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

## Reinforcement learning (rl/)

`rl/` trains a snake from scratch with
[MaskablePPO](https://sb3-contrib.readthedocs.io/) on the Gymnasium
environment above, against opponents that get harder stage by stage. The
extras it needs are separate from the server's:

```powershell
pip install -r requirements-rl.txt   # torch (CUDA 12.8), stable-baselines3, sb3-contrib
```

```powershell
# One stage at a time, each continuing from the last
python -m rl.train --stage solo      --timesteps 10000000 --max-minutes 25 --envs 24 --device cuda --threads 4
python -m rl.train --stage duel_weak --from runs\solo\model.zip      --timesteps 10000000 --max-minutes 25
python -m rl.train --stage four_weak --from runs\duel_weak\model.zip --timesteps 10000000 --max-minutes 25

# Against the tuned trees, optionally paying for territory as well as survival
python -m rl.train --stage four_default --from runs\four_weak\model.zip --max-minutes 25 --territory-reward 0.02

# How often it wins, on seeds training never saw
python -m rl.evaluate runs\four_weak\model.zip --stage four_weak --games 1000 --first-seed 100000

# Watch it play, with its move probabilities (see the replay viewer above)
python tools\explain_game.py --simulate --agent runs\four_weak\model.zip --opponents 3 --opponent-options none -o replay.html

tensorboard --logdir runs
```

- **Stages** (`rl/envs.py`): `solo` alone, `duel_weak` against one behavior
  tree with every option off, `four_weak` against three of them, and
  `four_default` against three of today's default trees.
- **Network** (`rl/model.py`): three padded 3x3 convolutions (6→32→64→64)
  keep all 11x11 cells, then a 7,744→256 layer sees the whole board, then
  separate move and value heads. 2.2M parameters.
- **Action masks**: moves that hit a wall or a body get zero probability, so
  the agent never has to learn that they kill. `MaskCachingVecEnv` answers
  those masks from the step infos; asking each worker process separately
  roughly halved training speed on Windows.
- **`--max-minutes`** stops a stage on time. It counts only time spent
  training, so a laptop that sleeps mid-run doesn't burn the budget.
- Training lives in the `runs/` folder (gitignored): `model.zip`,
  `best/`, `checkpoints/` every 250k steps, `eval/`, and TensorBoard logs.

Results of the first curriculum, about 63 minutes of training in total on an
RTX 5070 Ti laptop (9.7M steps, roughly 2,000-2,700 steps per second):

| Stage | Steps | Result on fresh seeds |
|---|---|---|
| solo | 3.8M | survives ~850 turns (our behavior tree: 690) |
| duel_weak | 4.2M | wins 67.8% [63.0, 72.1] against one weak tree (even: 50%) |
| four_weak | 1.7M | wins 59.7% [56.6, 62.7] against three weak trees (even: 25%) |
| four_default | 2.4M | wins 7.6% [6.1, 9.4] against three default trees (even: 25%) |
| four_default, with the territory reward and space channels | 36M | wins **35.6% [32.7, 38.6]** against three default trees |

Numbers in brackets are 95% confidence intervals. The last row is the model
to use: `runs/overnight_6/model.zip`, after nine hours of training in all.

Getting there took three things, and the order matters:

1. **The territory reward** (2.0% → 7.6% → 11.6%). Being trapped caused 82%
   of losses, and paying per turn for space attacks that directly.
2. **The space channels** (11.6% → 12.2%, then 20.2%). The first 25 minutes
   showed nothing; the gain only appeared with more training. Reading that
   flat result as failure would have thrown away a good idea.
3. **Time** (20.2% → 35.6%). Overnight, 30 minutes at a time, the agent
   passed an even share (25%) after about 90 more minutes and peaked at three
   hours. Training past that point did not help: at six hours it scored 32.1%.

Each 30-minute chunk was scored over 1,000 games. Since the peak was picked
by looking at twelve such scores, it was re-measured on seeds none of them
had seen; it held (36.7% became 35.6%). Being trapped now ends 58% of its
games, down from 82%.

### Self-play

Watching a replay showed the agent walking into cells right beside an
equal-length enemy head to take the food. That works because *our* behavior
tree always gives way there: refusing a head-to-head it can't win is in its
core safety filter, not an option, so in every game the agent had ever
played, the opponent backed off. Against a snake that doesn't, both die.

`rl/opponents.py` fixes that by drawing each opponent, once per game, from
earlier saved versions of the agent and the behavior tree:

```powershell
python -m rl.train --stage four_default --from runs/four/model.zip --run-name selfplay `
  --opponent-models runs/four/model.zip,runs/overnight_6/model.zip --bt-share 0.5 `
  --spatial --territory-reward 0.02 --max-minutes 30

# How a model does against another model, rather than against the tree
python -m rl.evaluate runs/selfplay/model.zip --stage four_default --games 600 `
  --first-seed 900000 --opponent-models runs/four/model.zip
```

Half an hour of it dropped the win rate against the tree from 35.6% to
25.4% — and that drop was the point. Played against each other, the
self-play model won 58.0% and the old champion 21.2%, against a 25% even
share. The 10 points it lost were the exploit, not skill.

Raising the tree's share to 50% then brought both up: 31.8% against the
tree and 35.7% against the model before it. Another half hour after that
made it worse (23.0% against its own predecessor), so training stopped
there.

**A win rate against one fixed opponent measures how well you exploit that
opponent as much as how well you play.** Against our own earlier models, it
can't.

## Playing on the real platform

The deployed snake plays with numpy alone: `rl/export.py` writes the policy
weights (7.5 MB, against a 26 MB checkpoint) and `rl/numpy_agent.py` runs
them. It picks the same moves as PyTorch — a test checks that — in 0.76 ms
rather than 2.07 ms, and the image needs no torch at all.

```powershell
python -m rl.export runs/selfplay/model.zip models\snake.npz
$env:SNAKE_BRAIN="rl"; python server.py     # or leave it unset for the tree
```

| Variable | What it does |
|---|---|
| `SNAKE_BRAIN` | `bt` (default) or `rl` |
| `SNAKE_MODEL` | weights for `rl` (default `models/snake.npz`) |
| `SNAKE_AUTHOR`, `SNAKE_COLOR`, `SNAKE_HEAD`, `SNAKE_TAIL` | how the snake appears |
| `HOST`, `PORT` | where to listen |

The network's observation is fixed to one board size and our rules have no
hazards, so the server falls back to the behavior tree on any board that
isn't 11x11 or that has hazards. The tree plays any board.

To deploy on [Fly.io](https://fly.io) with the included `Dockerfile` and
`fly.toml`:

```powershell
fly auth login
fly apps create <name>        # then set app = "<name>" in fly.toml
fly deploy --ha=false
fly secrets set SNAKE_AUTHOR=<your username> SNAKE_HEAD=silly
```

`min_machines_running = 1` keeps the machine awake: the engine gives up on a
move after about 500 ms, which a cold start would blow. Deciding takes under
a millisecond, so the rest of that budget is network travel — put the app in
a region near the game servers, and pick the engine region closest to it.
