"""Save a trained model's weights as a .npz the server can use without PyTorch.

    python -m rl.export runs/selfplay_1/model.zip models/snake.npz

A model.zip carries the optimizer state and everything else training needs,
about 26 MB. Playing needs only the weights that turn a board into four move
scores: about 8 MB, and no torch to load them. See rl/numpy_agent.py.
"""

import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

# The layers a move goes through, in order. Names are state_dict keys.
LAYERS = {
    "conv1": "features_extractor.net.0",
    "conv2": "features_extractor.net.2",
    "conv3": "features_extractor.net.4",
    "features": "features_extractor.net.7",
    "policy": "mlp_extractor.policy_net.0",
    "action": "action_net",
}


def export(model_path: str | Path, out_path: str | Path) -> Path:
    model = MaskablePPO.load(model_path, device="cpu")
    state = model.policy.state_dict()
    arrays = {}
    for name, key in LAYERS.items():
        for part in ("weight", "bias"):
            arrays[f"{name}.{part}"] = state[f"{key}.{part}"].cpu().numpy().astype(np.float32)
    channels, height, width = model.observation_space.shape
    arrays["shape"] = np.array([channels, height, width], dtype=np.int32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("out")
    args = parser.parse_args()
    path = export(args.model, args.out)
    size = path.stat().st_size / 1024**2
    print(f"{path}: {size:.1f} MB")


if __name__ == "__main__":
    main()
