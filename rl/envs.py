"""Training environments: one per curriculum stage, with action masks.

A curriculum raises the difficulty step by step. The agent first learns to
survive alone, then to beat one weak behavior-tree snake, then three. Each
stage continues from the model trained on the previous one.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor

from agent.strategy import DEFAULT_OPTIONS, Options
from gym_env.env import SnakeEnv, action_mask, bt_policy


@dataclass(frozen=True)
class Stage:
    opponents: int
    opponent_options: Options
    description: str


STAGES = {
    "solo": Stage(0, Options(), "alone on the board"),
    "duel_weak": Stage(1, Options(), "one behavior-tree snake with every option off"),
    "four_weak": Stage(3, Options(), "three behavior-tree snakes with every option off"),
    "four_default": Stage(3, DEFAULT_OPTIONS, "three behavior-tree snakes with the default options"),
}


class MaskedSnakeEnv(gym.Wrapper):
    """Adds action_masks(), which MaskablePPO asks for before every step.

    Masked-out actions get zero probability, so the agent never wastes time
    learning that walls and bodies kill. When every move is fatal the
    environment's mask is all zeros; a policy with nothing to choose from
    breaks (its probabilities become NaN), so we allow everything then. The
    snake dies either way.
    """

    def action_masks(self) -> np.ndarray:
        mask = action_mask(self.env.unwrapped.state).astype(bool)
        return mask if mask.any() else np.ones_like(mask)


def make_env(stage: str, max_turns: int = 1000, monitor: bool = True) -> gym.Env:
    """A fresh environment for `stage` (a key of STAGES)."""
    spec = STAGES[stage]
    env: gym.Env = SnakeEnv(
        opponents=spec.opponents,
        opponent_policy=bt_policy(spec.opponent_options),
        max_turns=max_turns,
    )
    if monitor:  # records episode length and reward for training logs
        env = Monitor(env)
    return MaskedSnakeEnv(env)
