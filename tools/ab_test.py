"""A/B test strategy options in the simulator.

    python tools/ab_test.py                         # every variant below
    python tools/ab_test.py lookahead hunt          # just these
    python tools/ab_test.py lookahead --games 500 --workers 8

A challenger snake using a variant's Options plays three baseline snakes
(every option off) on an 11x11 board, under the rules in gym_env/rules.py.
Four equally strong snakes each win 25% of games, so a variant helps if
its win rate is clearly above 25%. We print a 95% Wilson confidence interval
and mark variants whose whole interval clears 25%. Seeds are the same for
every variant, so they all start from the same layouts.

Also printed, for context: draws, how the challenger died, and its average
survival alone on the board.
"""

import argparse
import collections
import math
import multiprocessing
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from agent.strategy import Options, decide  # noqa: E402
from agent.world import World  # noqa: E402
from gym_env import rules  # noqa: E402

BASELINE = Options()
VARIANTS = {
    "baseline": BASELINE,
    "lookahead": Options(lookahead=True),
    "length_race": Options(length_race=True),
    "hunt": Options(hunt=True),
    "lookahead+length_race": Options(lookahead=True, length_race=True),
    "lookahead+hunt": Options(lookahead=True, hunt=True),
    "length_race+hunt": Options(length_race=True, hunt=True),
    "lookahead+length_race+hunt": Options(lookahead=True, length_race=True, hunt=True),
}
CHALLENGER = "challenger"
MAX_TURNS = 5000
SOLO_GAMES = 200


def play(variant: str, seed: int, opponents: int = 3) -> dict:
    """One game. Returns how it ended for the challenger."""
    rng = random.Random(seed)
    ids = [CHALLENGER] + [f"baseline{i}" for i in range(1, opponents + 1)]
    state = rules.new_game(ids, rng)
    options = {sid: VARIANTS[variant] if sid == CHALLENGER else BASELINE for sid in ids}
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # No `choices` here: with nargs="*", argparse would also check the default
    # list against them and reject it, so names are validated below instead.
    parser.add_argument("variants", nargs="*", help=f"any of: {', '.join(VARIANTS)} (default: all)")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()
    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        parser.error(f"unknown variant(s): {', '.join(unknown)}")
    args.variants = args.variants or list(VARIANTS)

    print(f"{args.games} games per variant: 1 challenger vs 3 baseline snakes, {args.workers} workers")
    with multiprocessing.Pool(args.workers) as pool:
        for variant in args.variants:
            start = time.perf_counter()
            results = pool.map(_play_duel, [(variant, seed) for seed in range(args.games)], chunksize=8)
            solo = pool.map(_play_solo, [(variant, seed) for seed in range(SOLO_GAMES)], chunksize=4)

            wins = sum(r["won"] for r in results)
            low, high = wilson(wins, args.games)
            verdict = "better than baseline" if low > 0.25 else ("worse" if high < 0.25 else "no clear difference")
            deaths = collections.Counter(r["cause"] for r in results if not r["won"] and r["cause"])
            died = sum(deaths.values())
            death_mix = ", ".join(f"{cause} {n / died:.0%}" for cause, n in deaths.most_common())
            print(f"\n{variant}  ({time.perf_counter() - start:.0f}s)")
            print(f"  win rate {wins / args.games:.1%}  95% CI [{low:.1%}, {high:.1%}]  -> {verdict}")
            print(f"  draws {sum(r['draw'] for r in results)}, mean game length {statistics.mean(r['turns'] for r in results):.0f} turns")
            print(f"  challenger deaths: {death_mix}")
            print(f"  solo survival ({SOLO_GAMES} games): mean {statistics.mean(r['turns'] for r in solo):.0f} turns")


if __name__ == "__main__":
    main()
