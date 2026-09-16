"""Training environments: one per curriculum stage, with action masks.

A curriculum raises the difficulty step by step. The agent first learns to
survive alone, then to beat one weak behavior-tree snake, then three. Each
stage continues from the model trained on the previous one.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

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
        return usable_mask(action_mask(self.env.unwrapped.state))


def usable_mask(mask: np.ndarray) -> np.ndarray:
    """The mask as booleans, with every action allowed when none is."""
    mask = np.asarray(mask, dtype=bool)
    return mask if mask.any() else np.ones_like(mask)


class MaskCachingVecEnv(VecEnvWrapper):
    """Answers action_masks() from the masks that step() and reset() already return.

    MaskablePPO asks every environment for its mask before each step. With
    one process per environment that's a second round trip for every move,
    which on Windows roughly halved training speed. SnakeEnv puts the mask in
    info anyway, so we keep the latest one. When an episode ends the vector
    env resets it straight away, and the new episode's mask is in reset_infos.
    """

    def __init__(self, venv: VecEnv):
        super().__init__(venv)
        self._masks = np.ones((self.num_envs, 4), dtype=bool)

    def reset(self):
        obs = self.venv.reset()
        for i, info in enumerate(self.venv.reset_infos):
            self._masks[i] = usable_mask(info["action_mask"])
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for i, (done, info) in enumerate(zip(dones, infos)):
            fresh = self.venv.reset_infos[i] if done else info
            self._masks[i] = usable_mask(fresh["action_mask"])
        return obs, rewards, dones, infos

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if method_name == "action_masks":
            chosen = range(self.num_envs) if indices is None else np.atleast_1d(indices)
            return [self._masks[i].copy() for i in chosen]
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)


def make_env(
    stage: str,
    max_turns: int = 1000,
    monitor: bool = True,
    rewards: dict[str, float] | None = None,
) -> gym.Env:
    """A fresh environment for `stage` (a key of STAGES).

    `rewards` overrides single reward weights, e.g. {"territory": 0.02} to
    pay for holding space; see gym_env.env.DEFAULT_REWARDS.
    """
    spec = STAGES[stage]
    env: gym.Env = SnakeEnv(
        opponents=spec.opponents,
        opponent_policy=bt_policy(spec.opponent_options),
        max_turns=max_turns,
        rewards=rewards,
    )
    if monitor:  # records episode length and reward for training logs
        env = Monitor(env)
    return MaskedSnakeEnv(env)
