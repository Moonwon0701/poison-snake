"""Give a trained model the three space channels without losing what it learned.

    python -m rl.expand runs/four_default_territory/model.zip runs/spatial/start.zip

Only the first convolution changes: its weights grow from (32, 6, 3, 3) to
(32, 9, 3, 3), and the new slots start at zero. Everything after it is the
same shape, so it transfers exactly. A zero weight means the new channels
contribute nothing at first, so the expanded model plays precisely like the
old one and then learns what the extra information is worth.
"""

import argparse
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from gym_env.env import CHANNELS, SPACE_CHANNELS
from rl.envs import make_env
from rl.model import new_model

CONV1 = "features_extractor.net.0.weight"


def expand(model_path: str | Path, out_path: str | Path, stage: str = "four_default") -> Path:
    """Write a copy of `model_path` that takes CHANNELS + SPACE_CHANNELS inputs."""
    old = MaskablePPO.load(model_path, device="cpu")
    channels = old.observation_space.shape[0]
    if channels != CHANNELS:
        raise ValueError(f"expected a {CHANNELS}-channel model, got {channels}")

    env = make_env(stage, monitor=False, spatial=True)
    new = new_model(env, device="cpu")
    state = dict(old.policy.state_dict())
    # The policy and value networks share one extractor, but the state dict
    # lists it under each name, so every copy needs the same padding.
    for key in [k for k in state if k.endswith(CONV1)]:
        weight = state[key]
        pad = torch.zeros(weight.shape[0], SPACE_CHANNELS, *weight.shape[2:])
        state[key] = torch.cat([weight, pad], dim=1)
    new.policy.load_state_dict(state)
    new.num_timesteps = old.num_timesteps

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new.save(out_path)
    env.close()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("out")
    args = parser.parse_args()
    path = expand(args.model, args.out)
    print(f"{path}: {CHANNELS} -> {CHANNELS + SPACE_CHANNELS} input channels")


if __name__ == "__main__":
    main()
