"""Play our behavior tree inside SnakeEnv and report survival and speed.

    python tools/sim_benchmark.py [episodes] [opponents]

With 0 opponents, compare the survival numbers against CLI games of the
same snake to check the simulator end to end. Milestone 4's CLI benchmark
averaged 482 turns.
"""

import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from gym_env import rules  # noqa: E402
from gym_env.env import ACTIONS, AGENT, SnakeEnv, behavior_tree_policy  # noqa: E402


def main() -> None:
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    opponents = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    env = SnakeEnv(opponents=opponents, max_turns=100_000)
    turns, wins, steps = [], 0, 0
    start = time.perf_counter()
    for episode in range(episodes):
        our_rng = random.Random(episode)
        _, info = env.reset(seed=episode)
        done = False
        while not done:
            move = behavior_tree_policy(rules.to_api_json(env.state, AGENT), our_rng)
            _, _, terminated, truncated, info = env.step(ACTIONS.index(move))
            steps += 1
            done = terminated or truncated
        turns.append(info["turn"])
        wins += info["won"]
    elapsed = time.perf_counter() - start

    print(f"Our behavior tree as the agent: {episodes} episodes, {opponents} opponents")
    print(
        f"  turns: mean {statistics.mean(turns):.1f}, median {statistics.median(turns):.0f},"
        f" sd {statistics.pstdev(turns):.1f}, min {min(turns)}, max {max(turns)}"
    )
    if opponents:
        print(f"  wins: {wins}/{episodes}")
    print(f"  {steps} steps in {elapsed:.1f}s = {steps / elapsed:.0f} steps/s")

    # The environment alone: random moves that don't hit anything.
    env = SnakeEnv(opponents=0, max_turns=100_000)
    rng = np.random.default_rng(0)
    _, info = env.reset(seed=0)
    steps, start = 0, time.perf_counter()
    while time.perf_counter() - start < 2.0:
        allowed = np.flatnonzero(info["action_mask"])
        action = int(rng.choice(allowed)) if len(allowed) else 0
        _, _, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            _, info = env.reset()
    print(f"Environment alone (random safe moves, solo): {steps / (time.perf_counter() - start):.0f} steps/s")


if __name__ == "__main__":
    main()
