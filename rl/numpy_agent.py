"""A trained snake that plays with numpy alone, for deployment without PyTorch.

    from rl.numpy_agent import NumpyPolicy
    policy = NumpyPolicy("models/snake.npz")
    move = policy(game_state)          # "up" | "down" | "left" | "right"

Same shape as gym_env.env.Policy, so it drops into the server, the viewer or
an evaluation like any other snake. The weights come from rl/export.py, and
the arithmetic below repeats what the trained network does, layer by layer:

    3x3 convolutions, 9 -> 32 -> 64 -> 64 channels, each followed by ReLU
    flatten, then 7744 -> 256 with ReLU        (our SnakeCNN)
    256 -> 256 with tanh                       (Stable-Baselines3's default)
    256 -> 4                                   (one score per move)
    scores for moves that hit a wall or a body are dropped, then the best wins

That last step matches model.predict(deterministic=True). About five million
multiply-adds, well under a millisecond.
"""

import random
from pathlib import Path

import numpy as np

from gym_env import rules
from gym_env.env import ACTIONS, CHANNELS, action_mask, encode_observation


def convolve(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """A 3x3 convolution with padding 1, over a (channels, height, width) board.

    Each output cell mixes a 3x3 patch of every input channel. Gathering
    those patches into one matrix ("im2col") turns the whole layer into a
    single matrix multiply, which numpy does far faster than looping.
    """
    channels, height, width = x.shape
    padded = np.zeros((channels, height + 2, width + 2), dtype=np.float32)
    padded[:, 1:-1, 1:-1] = x
    patches = np.empty((channels * 9, height * width), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            window = padded[:, i : i + height, j : j + width]
            patches[(i * 3 + j) * channels : (i * 3 + j + 1) * channels] = window.reshape(channels, -1)
    # The weight is (out, in, 3, 3); reorder it to match how patches are stacked.
    flat = weight.transpose(2, 3, 1, 0).reshape(9 * channels, -1)
    out = flat.T @ patches + bias[:, None]
    return out.reshape(-1, height, width)


class NumpyPolicy:
    def __init__(self, weights_path: str | Path):
        self.name = Path(weights_path).as_posix()
        self.w = {k: v for k, v in np.load(weights_path).items()}
        self.channels = int(self.w["shape"][0])
        self.spatial = self.channels > CHANNELS

    def __call__(self, game_state: dict, rng: random.Random | None = None) -> str:
        return self.explain(game_state)["move"]

    def explain(self, game_state: dict) -> dict:
        """The chosen move, with each move's probability and whether it was allowed."""
        state = rules.from_api_json(game_state)
        me = game_state["you"]["id"]
        mask = action_mask(state, me).astype(bool)
        if not mask.any():  # every move is fatal; pick one and die facing forward
            mask[:] = True
        logits = self._scores(encode_observation(state, me, spatial=self.spatial))

        allowed = np.where(mask, logits, -np.inf)
        probs = np.exp(allowed - allowed.max())
        probs /= probs.sum()
        action = int(np.argmax(allowed))
        return {
            "move": ACTIONS[action],
            "policy": {move: round(float(p), 4) for move, p in zip(ACTIONS, probs)},
            "mask": {move: bool(ok) for move, ok in zip(ACTIONS, mask)},
        }

    def _scores(self, obs: np.ndarray) -> np.ndarray:
        w = self.w
        x = obs.astype(np.float32)
        for layer in ("conv1", "conv2", "conv3"):
            x = np.maximum(convolve(x, w[f"{layer}.weight"], w[f"{layer}.bias"]), 0.0)
        x = x.reshape(-1)
        x = np.maximum(w["features.weight"] @ x + w["features.bias"], 0.0)
        x = np.tanh(w["policy.weight"] @ x + w["policy.bias"])
        return w["action.weight"] @ x + w["action.bias"]
