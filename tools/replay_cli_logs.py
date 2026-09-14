"""Check gym_env.rules against games recorded by the Battlesnake CLI.

Record games with `battlesnake play ... -o game.jsonl`, then run:

    python tools/replay_cli_logs.py <log file or folder> [...]

For every pair of consecutive turns in a log, this works out each snake's
move from where its head went, runs rules.step, and compares the result with
the next recorded turn: the same snakes alive, with the same bodies and the
same health. Snakes that were eliminated leave no next head, so every move
is tried for them.

That makes eliminations the weak spot. The log records neither the move nor
the cause, and turning into one's own neck always kills, so almost any death
can be explained. Survivors, growth, health and food are checked strictly,
but *why* a snake died is not. A bug that let equal-length snakes survive a
head-to-head passed this check unnoticed. tests/test_rules.py covers the
elimination rules case by case.

New food lands on a random cell, and our random numbers aren't the engine's,
so food is checked loosely. All food the rules kept must still be there, the
right number of new pieces may appear, and only on cells no snake covers.

The first turn of each game is also checked against the standard map's
start layout.
"""

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from agent.world import MOVES  # noqa: E402
from gym_env import rules  # noqa: E402


def infer_move(before, after) -> str | None:
    delta = (after[0] - before[0], after[1] - before[1])
    return next((move for move, d in MOVES.items() if d == delta), None)


def transition_matches(current: dict, following: dict, settings: dict) -> bool:
    before = rules.from_api_json(current)
    after = rules.from_api_json(following)
    alive_after = {s.id: s for s in after.snakes}

    known, unknown = {}, []
    for snake in before.snakes:
        if snake.id in alive_after:
            move = infer_move(snake.head, alive_after[snake.id].head)
            if move is None:
                return False
            known[snake.id] = move
        else:
            unknown.append(snake.id)

    for guesses in itertools.product(MOVES, repeat=len(unknown)):
        moves = {**known, **dict(zip(unknown, guesses))}
        result, _ = rules.step(before, moves, food_spawn_chance=0, minimum_food=0)
        if same_snakes(result, after) and food_matches(result, after, settings):
            return True
    return False


def same_snakes(ours: rules.GameState, recorded: rules.GameState) -> bool:
    def summary(state):
        return {s.id: (s.body, s.health) for s in state.snakes}

    return ours.turn == recorded.turn and summary(ours) == summary(recorded)


def food_matches(ours: rules.GameState, recorded: rules.GameState, settings: dict) -> bool:
    kept, seen = set(ours.food), set(recorded.food)
    if not kept <= seen:
        return False
    new = seen - kept
    if new & {p for s in recorded.snakes for p in s.body}:
        return False
    minimum = settings.get("minimumFood", 0)
    if len(ours.food) < minimum:
        return len(new) == minimum - len(ours.food)
    return len(new) <= (1 if settings.get("foodSpawnChance", 0) > 0 else 0)


def start_problem(game_state: dict) -> str | None:
    """What's wrong with a game's first turn, or None if it's a standard start."""
    state = rules.from_api_json(game_state)
    low, mid, high = 1, (state.width - 1) // 2, state.width - 2
    corners = {(low, low), (low, high), (high, low), (high, high)}
    edges = {(low, mid), (mid, low), (mid, high), (high, mid)}
    heads = [s.head for s in state.snakes]
    if any(s.body != [s.head] * rules.START_LENGTH or s.health != rules.MAX_HEALTH for s in state.snakes):
        return "snakes should start coiled up with full health"
    if len(set(heads)) != len(heads) or not set(heads) <= corners | edges:
        return f"unexpected start cells {heads}"
    if len(heads) <= 4 and not (set(heads) <= corners or set(heads) <= edges):
        return "with up to 4 snakes, all should start on corners or all on edges"
    if (mid, mid) not in state.food or len(state.food) != len(heads) + 1:
        return f"expected centre food plus one per snake, got {state.food}"
    for x, y in heads:
        if not any(abs(fx - x) == 1 and abs(fy - y) == 1 for fx, fy in state.food):
            return f"no food diagonal to the snake at {(x, y)}"
    return None


def main(paths: list[str]) -> int:
    files = []
    for path in map(Path, paths):
        files += sorted(path.glob("*.jsonl")) if path.is_dir() else [path]

    games = transitions = failures = bad_starts = 0
    problems = []
    for path in files:
        lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        settings = lines[0].get("ruleset", {}).get("settings", {})
        states = [line for line in lines if "board" in line]
        if not states:
            continue
        games += 1
        problem = start_problem(states[0])
        if problem:
            bad_starts += 1
            problems.append(f"{path.name}: start: {problem}")
        for current, following in zip(states, states[1:]):
            transitions += 1
            if not transition_matches(current, following, settings):
                failures += 1
                problems.append(f"{path.name}: turn {current['turn']} -> {following['turn']}")

    print(f"{games} games, {transitions} turn transitions")
    print(f"  start layouts that differ: {bad_starts}")
    print(f"  transitions that differ:   {failures}")
    for problem in problems[:10]:
        print("   ", problem)
    return 1 if failures or bad_starts else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
