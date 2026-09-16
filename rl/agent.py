"""A trained model as a snake: engine-style game state in, move out.

RLPolicy has the same shape as gym_env.env.Policy, so a trained snake can be
an opponent in SnakeEnv, a player in replays, or later the brain behind the
HTTP server.
"""

import random
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from gym_env import rules
from gym_env.env import ACTIONS, CHANNELS, action_mask, encode_observation


class RLPolicy:
    def __init__(self, model_path: str | Path, device: str = "cpu"):
        # One board at a time is faster on the CPU than a round trip to the GPU.
        self.model = MaskablePPO.load(model_path, device=device)
        self.name = Path(model_path).as_posix()
        # A model trained with the space channels expects them at every step.
        self.spatial = self.model.observation_space.shape[0] > CHANNELS

    def __call__(self, game_state: dict, rng: random.Random | None = None) -> str:
        return self.explain(game_state)["move"]

    def explain(self, game_state: dict) -> dict:
        """The chosen move, with the policy's probabilities and value estimate.

        The move is the most likely allowed action, i.e. the same one
        model.predict(..., deterministic=True) would pick.
        """
        state = rules.from_api_json(game_state)
        you = game_state["you"]["id"]
        mask = action_mask(state, you).astype(bool)
        if not mask.any():  # every move is fatal; see rl.envs.MaskedSnakeEnv
            mask[:] = True
        obs = encode_observation(state, you, spatial=self.spatial)

        policy = self.model.policy
        obs_tensor, _ = policy.obs_to_tensor(obs)
        with torch.no_grad():
            distribution = policy.get_distribution(obs_tensor, action_masks=mask)
            probs = distribution.distribution.probs[0].cpu().numpy()
            value = float(policy.predict_values(obs_tensor)[0, 0])
        action = int(np.argmax(probs))
        return {
            "move": ACTIONS[action],
            "policy": {move: round(float(p), 4) for move, p in zip(ACTIONS, probs)},
            "mask": {move: bool(allowed) for move, allowed in zip(ACTIONS, mask)},
            "value": round(value, 4),
        }
