"""A/B test strategy options in the simulator.

    python tools/ab_test.py                                  # every variant below
    python tools/ab_test.py "territory:*" --games 1000       # glob patterns work
    python tools/ab_test.py lookahead hunt --baseline none   # against all-off snakes

A challenger snake using a variant's Options plays three baseline snakes on
an 11x11 board, under the rules in gym_env/rules.py. The baseline is the
current DEFAULT_OPTIONS unless --baseline names another variant. Four
equally strong snakes each win 25% of games, so a variant helps if its win
rate is clearly above 25%. We print a 95% Wilson confidence interval and mark
variants whose whole interval clears 25%. Seeds are the same for every
variant, so they all start from the same layouts. After screening many
variants, confirm the best ones with --first-seed set past the screening
seeds: a variant picked for doing well on some seeds tends to look a bit
better on those seeds than it really is.

Also printed, for context: draws, how the challenger died, how often it was
trapped, and its average survival alone on the board.
"""

import argparse
import collections
import dataclasses
import fnmatch
import itertools
import math
import multiprocessing
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from agent.strategy import DEFAULT_OPTIONS, Options, decide  # noqa: E402
from agent.world import World  # noqa: E402
from gym_env import rules  # noqa: E402

HEAD_TO_HEAD = Options(lookahead=True, length_race=True, hunt=True)
TRAP_OPTIONS = ["voronoi_pick", "voronoi_gate", "worst_case_filter", "timed_area"]

VARIANTS = {
    "default": DEFAULT_OPTIONS,
    "none": Options(),
    # Head-to-head experiments (originally run with --baseline none).
    "lookahead": Options(lookahead=True),
    "length_race": Options(length_race=True),
    "hunt": Options(hunt=True),
    "lookahead+length_race": Options(lookahead=True, length_race=True),
    "lookahead+hunt": Options(lookahead=True, hunt=True),
    "length_race+hunt": Options(length_race=True, hunt=True),
    "head_to_head": HEAD_TO_HEAD,
}
# Trap experiments: every combination of the trap options on top of head_to_head.
for size in range(1, len(TRAP_OPTIONS) + 1):
    for chosen in itertools.combinations(TRAP_OPTIONS, size):
        VARIANTS["territory:" + "+".join(chosen)] = dataclasses.replace(
            HEAD_TO_HEAD, **{name: True for name in chosen}
        )

CHALLENGER = "challenger"
TRAPPED = {rules.WALL_COLLISION, rules.SELF_COLLISION, rules.BODY_COLLISION}
MAX_TURNS = 5000
SOLO_GAMES = 200


def play(variant: str, seed: int, baseline: str, opponents: int = 3) -> dict:
    """One game. Returns how it ended for the challenger."""
    rng = random.Random(seed)
    ids = [CHALLENGER] + [f"baseline{i}" for i in range(1, opponents + 1)]
    state = rules.new_game(ids, rng)
    options = {sid: VARIANTS[variant if sid == CHALLENGER else baseline] for sid in ids}
    game_over = (lambda s: not s.snakes) if opponents == 0 else (lambda s: len(s.snakes) <= 1)
    cause = None
    while not game_over(state) and state.turn < MAX_TURNS:
        moves = {
            s.id: decide(World.from_json(rules.to_api_json(state, s.id)), rng, options[s.id])
            for s in state.snakes
        }
        state, eliminated = rules.step(state, moves, rng)
        cause = eliminated.get(CHALLENGER, cause)
    alive = [s.id for s in state.snakes]
    return {
        "won": alive == [CHALLENGER],
        "draw": not alive,
        "cause": cause,
        "turns": state.turn,
    }


def _play_duel(args):
    return play(*args)


def _play_solo(args):
    return play(*args, opponents=0)


def wilson(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a win rate (Wilson score interval)."""
    p = wins / games
    denom = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denom
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    return center - half, center + half


def pick_variants(patterns: list[str]) -> list[str]:
    """Variant names matching the given names or glob patterns, in table order."""
    if not patterns:
        return list(VARIANTS)
    picked = []
    for pattern in patterns:
        matches = [name for name in VARIANTS if fnmatch.fnmatchcase(name, pattern)]
        if not matches:
            raise ValueError(f"no variant matches {pattern!r}")
        picked += [name for name in matches if name not in picked]
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", help="variant names or glob patterns (default: all)")
    parser.add_argument("--baseline", default="default", help="variant the three opponents use")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument(
        "--first-seed", type=int, default=0,
        help="seeds run from here; confirm a screening winner on seeds it wasn't picked on",
    )
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()
    try:
        variants = pick_variants(args.variants)
    except ValueError as error:
        parser.error(str(error))
    if args.baseline not in VARIANTS:
        parser.error(f"unknown baseline {args.baseline!r}; choose from: {', '.join(VARIANTS)}")

    seeds = range(args.first_seed, args.first_seed + args.games)
    print(
        f"{args.games} games per variant (seeds {seeds.start}-{seeds.stop - 1}): "
        f"1 challenger vs 3 '{args.baseline}' snakes, {args.workers} workers"
    )
    summary = []
    with multiprocessing.Pool(args.workers) as pool:
        for variant in variants:
            start = time.perf_counter()
            jobs = [(variant, seed, args.baseline) for seed in seeds]
            results = pool.map(_play_duel, jobs, chunksize=8)
            solo_seeds = range(args.first_seed, args.first_seed + SOLO_GAMES)
            solo = pool.map(_play_solo, [(variant, seed, args.baseline) for seed in solo_seeds], chunksize=4)

            wins = sum(r["won"] for r in results)
            low, high = wilson(wins, args.games)
            verdict = "better than baseline" if low > 0.25 else ("worse" if high < 0.25 else "no clear difference")
            deaths = collections.Counter(r["cause"] for r in results if not r["won"] and r["cause"])
            died = sum(deaths.values())
            death_mix = ", ".join(f"{cause} {n / died:.0%}" for cause, n in deaths.most_common())
            trapped = sum(n for cause, n in deaths.items() if cause in TRAPPED)
            summary.append((wins / args.games, low, high, trapped / args.games, variant))
            print(f"\n{variant}  ({time.perf_counter() - start:.0f}s)")
            print(f"  win rate {wins / args.games:.1%}  95% CI [{low:.1%}, {high:.1%}]  -> {verdict}")
            print(f"  draws {sum(r['draw'] for r in results)}, mean game length {statistics.mean(r['turns'] for r in results):.0f} turns")
            print(f"  challenger deaths: {death_mix}")
            print(f"  trapped (wall, self or body) in {trapped / args.games:.1%} of games")
            print(f"  solo survival ({SOLO_GAMES} games): mean {statistics.mean(r['turns'] for r in solo):.0f} turns")

    if len(summary) > 1:
        print("\nRanking by win rate:")
        for rate, low, high, trapped, variant in sorted(summary, reverse=True):
            print(f"  {rate:6.1%}  [{low:5.1%}, {high:5.1%}]  trapped {trapped:5.1%}  {variant}")


if __name__ == "__main__":
    main()
