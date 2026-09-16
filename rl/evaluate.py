"""Measure a trained snake on fresh seeds.

    python -m rl.evaluate runs/four_weak/model.zip --stage four_weak --games 1000 --first-seed 100000
    python -m rl.evaluate runs/solo/model.zip --stage solo --games 200

Against opponents it prints the win rate with a 95% Wilson confidence
interval. With three opponents an equally strong snake wins 25% of games,
with one 50%. Alone it prints how long the snake survives.

Each parallel environment plays its share of the games back to back. If we
simply took the first N games to finish, short games (early deaths) would be
over-represented.
"""

import argparse
import collections
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from gym_env import rules
from gym_env.env import CHANNELS
from rl.envs import STAGES
from rl.train import build_vec_env

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from ab_test import wilson  # noqa: E402

TRAPPED = {rules.WALL_COLLISION, rules.SELF_COLLISION, rules.BODY_COLLISION}


def evaluate(
    model_path: str | Path,
    stage: str,
    games: int = 1000,
    first_seed: int = 100_000,
    n_envs: int = 16,
    device: str = "auto",
    subprocess: bool = True,
) -> dict:
    model = MaskablePPO.load(model_path, device=device)
    n_envs = min(n_envs, games)
    quota = [games // n_envs + (1 if i < games % n_envs else 0) for i in range(n_envs)]
    # Match the observation the model was trained on, with or without the
    # space channels. Reward weights don't matter here; we count wins.
    spatial = model.observation_space.shape[0] > CHANNELS
    vec_env = build_vec_env(stage, n_envs, first_seed, subprocess, spatial=spatial)
    results: list[dict] = []
    finished = [0] * n_envs
    start = time.perf_counter()
    obs = vec_env.reset()
    while any(finished[i] < quota[i] for i in range(n_envs)):
        masks = np.stack(vec_env.env_method("action_masks"))
        actions, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, _, dones, infos = vec_env.step(actions)
        for i, (done, info) in enumerate(zip(dones, infos)):
            if done and finished[i] < quota[i]:
                finished[i] += 1
                results.append({
                    "won": bool(info.get("won", False)),
                    "cause": info.get("cause"),
                    "turns": int(info["turn"]),
                    "truncated": bool(info.get("TimeLimit.truncated", False)),
                })
    vec_env.close()

    wins = sum(r["won"] for r in results)
    low, high = wilson(wins, len(results))
    causes = collections.Counter(r["cause"] for r in results if r["cause"])
    return {
        "stage": stage,
        "games": len(results),
        "wins": wins,
        "win_rate": wins / len(results),
        "ci": (low, high),
        "fair_share": 1 / (STAGES[stage].opponents + 1),
        "causes": dict(causes),
        "trapped": sum(n for cause, n in causes.items() if cause in TRAPPED) / len(results),
        "mean_turns": statistics.mean(r["turns"] for r in results),
        "truncated": sum(r["truncated"] for r in results),
        "seconds": time.perf_counter() - start,
    }


def report(stats: dict) -> str:
    lines = [f"{stats['games']} games on stage {stats['stage']} ({stats['seconds']:.0f}s)"]
    if STAGES[stats["stage"]].opponents:
        low, high = stats["ci"]
        share = stats["fair_share"]
        verdict = "better than an equal snake" if low > share else ("worse" if high < share else "no clear difference")
        lines.append(
            f"  win rate {stats['win_rate']:.1%}  95% CI [{low:.1%}, {high:.1%}]"
            f"  (equal strength: {share:.0%})  -> {verdict}"
        )
    lines.append(f"  mean game length {stats['mean_turns']:.0f} turns, {stats['truncated']} hit the turn limit")
    died = sum(stats["causes"].values())
    if died:
        mix = ", ".join(f"{cause} {n / died:.0%}" for cause, n in sorted(stats["causes"].items(), key=lambda kv: -kv[1]))
        lines.append(f"  deaths: {mix}; trapped in {stats['trapped']:.1%} of games")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--stage", required=True, choices=list(STAGES))
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--first-seed", type=int, default=100_000)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    stats = evaluate(args.model, args.stage, args.games, args.first_seed, args.envs, args.device)
    print(report(stats))


if __name__ == "__main__":
    main()
