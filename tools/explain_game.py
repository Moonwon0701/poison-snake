"""Turn a game into a self-contained HTML replay that shows each snake's reasoning.

    python tools/explain_game.py game.jsonl [-o replay.html]
    python tools/explain_game.py --simulate [--opponents 3] [--seed 0] [-o replay.html]

With a CLI log (from `battlesnake play ... -o game.jsonl`), every snake's
decision is worked out again, turn by turn, with the CURRENT code. That shows
what today's behavior tree thinks, even about a game an older version played.
It also means a random tie-break can pick a different move than the one
recorded; the viewer shows both when that happens.

With --simulate, our behavior tree plays a new game under gym_env.rules, so
every explanation is exactly the decision that was made.

    python tools/explain_game.py --simulate --agent runs/four_weak/model.zip --opponent-options none

With --agent, snake1 is a trained RL model (see rl/) and the viewer shows its
move probabilities instead of a behavior tree.

Open the HTML file in any browser. It needs no server and no internet.
"""

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run from anywhere

from agent.strategy import DEFAULT_OPTIONS, Options, build_tree, explain  # noqa: E402
from agent.world import MOVES, World  # noqa: E402
from gym_env import rules  # noqa: E402

TEMPLATE = ROOT / "viewer" / "template.html"
PLACEHOLDER = "__REPLAY_DATA__"


def replay_from_log(path: Path, options: Options = DEFAULT_OPTIONS) -> dict:
    """Explain every turn of a CLI game log, re-deciding with the current code."""
    lines = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    states = [rules.from_api_json(line) for line in lines if "board" in line]
    names = {}
    for line in lines:
        for snake in line.get("board", {}).get("snakes", []):
            names.setdefault(snake["id"], snake["name"])

    rng = random.Random(0)
    turns = []
    for i, state in enumerate(states):
        following = states[i + 1] if i + 1 < len(states) else None
        frame = _frame(state)
        frame["decisions"] = _explain_all(state, rng, options)
        for snake in state.snakes:
            after = following.snake(snake.id) if following else None
            frame["decisions"][snake.id]["actual"] = (
                _move_between(snake.head, after.head) if after else None
            )
        # The log doesn't say why a snake was eliminated, only that it's gone.
        frame["eliminated"] = {
            s.id: "" for s in state.snakes if following and following.snake(s.id) is None
        }
        turns.append(frame)

    return _replay(f"CLI log {Path(path).name}", False, states[0], names, turns, options)


def replay_from_simulation(
    seed: int = 0,
    opponents: int = 3,
    options: Options = DEFAULT_OPTIONS,
    max_turns: int = 1000,
    agent=None,
) -> dict:
    """Play a new game with our behavior tree for every snake, explaining each move.

    With `agent` (an rl.agent.RLPolicy), snake1 is that trained model instead.
    Its decisions hold the policy's move probabilities and value estimate
    rather than a behavior-tree explanation.
    """
    rng = random.Random(seed)
    ids = [f"snake{i}" for i in range(1, opponents + 2)]
    state = rules.new_game(ids, rng)
    first = state
    turns = []
    while True:
        frame = _frame(state)
        decisions = _explain_all(state, rng, options, skip=ids[0] if agent else None)
        if agent and state.snake(ids[0]):
            decisions[ids[0]] = agent.explain(rules.to_api_json(state, ids[0]))
        for decision in decisions.values():
            decision["actual"] = decision["move"]
        frame["decisions"] = decisions
        game_over = not state.snakes if opponents == 0 else len(state.snakes) <= 1
        if game_over or state.turn >= max_turns:
            frame["eliminated"] = {}
            turns.append(frame)
            break
        moves = {sid: decision["move"] for sid, decision in decisions.items()}
        state, eliminated = rules.step(state, moves, rng)
        frame["eliminated"] = eliminated
        turns.append(frame)

    names = {sid: sid for sid in ids}
    source = f"simulated game, seed {seed}, {opponents} opponent{'s' if opponents != 1 else ''}"
    if agent:
        names[ids[0]] = f"{ids[0]} (RL)"
        source += f", {ids[0]} is the trained model {agent.name}"
    return _replay(source, True, first, names, turns, options)


def render_html(replay: dict, template: Path = TEMPLATE) -> str:
    """The viewer template with `replay` embedded in it."""
    html = template.read_text(encoding="utf-8")
    if PLACEHOLDER not in html:
        raise ValueError(f"{template} has no {PLACEHOLDER} placeholder")
    # "</" inside the JSON would end the <script> element early; "<\/" is the
    # same string to a JSON parser.
    data = json.dumps(replay, separators=(",", ":")).replace("</", "<\\/")
    return html.replace(PLACEHOLDER, data, 1)


def _replay(source, exact, first, names, turns, options) -> dict:
    return {
        "source": source,
        "exact": exact,
        "width": first.width,
        "height": first.height,
        "options": dataclasses.asdict(options),
        "tree": build_tree(options).describe(),
        "snakes": [{"id": sid, "name": name} for sid, name in names.items()],
        "turns": turns,
    }


def _frame(state: rules.GameState) -> dict:
    return {
        "turn": state.turn,
        "food": [list(p) for p in state.food],
        "snakes": [
            {"id": s.id, "health": s.health, "body": [list(p) for p in s.body]}
            for s in state.snakes
        ],
    }


def _explain_all(
    state: rules.GameState, rng: random.Random, options: Options, skip: str | None = None
) -> dict:
    return {
        s.id: explain(World.from_json(rules.to_api_json(state, s.id)), rng, options)
        for s in state.snakes
        if s.id != skip
    }


def _move_between(before, after) -> str | None:
    delta = (after[0] - before[0], after[1] - before[1])
    return next((move for move, d in MOVES.items() if d == delta), None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log", nargs="?", type=Path, help="a CLI game log (.jsonl)")
    parser.add_argument("--simulate", action="store_true", help="play a new game instead")
    parser.add_argument("--opponents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=1000)
    parser.add_argument(
        "--agent", type=Path, help="with --simulate: a trained model.zip plays as snake1"
    )
    parser.add_argument(
        "--opponent-options",
        choices=["default", "none"],
        default="default",
        help="with --simulate: behavior-tree options, all default or all off",
    )
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    if args.simulate == (args.log is not None):
        parser.error("give either a CLI log or --simulate, not both")
    if args.agent and not args.simulate:
        parser.error("--agent needs --simulate")
    if args.simulate:
        agent = None
        if args.agent:
            from rl.agent import RLPolicy  # needs requirements-rl.txt

            agent = RLPolicy(args.agent)
        options = DEFAULT_OPTIONS if args.opponent_options == "default" else Options()
        replay = replay_from_simulation(
            args.seed, args.opponents, options, max_turns=args.max_turns, agent=agent
        )
        output = args.output or Path(f"replay-seed{args.seed}.html")
    else:
        replay = replay_from_log(args.log)
        output = args.output or args.log.with_suffix(".html")

    output.write_text(render_html(replay), encoding="utf-8")
    size = output.stat().st_size / 1024
    print(f"{output}: {len(replay['turns'])} turns, {len(replay['snakes'])} snakes, {size:.0f} KB")


if __name__ == "__main__":
    main()
