"""The policy network and PPO settings."""

import gymnasium as gym
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class SnakeCNN(BaseFeaturesExtractor):
    """Turns the (6, 11, 11) board into a feature vector.

    Stable-Baselines3's default image network expects pictures of at least
    36x36 and shrinks them fast. Our board is tiny, so we use padded 3x3
    convolutions that keep every cell. Three of them let each cell see up to
    3 cells away, then a linear layer looks at the whole board.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        channels, height, width = observation_space.shape
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * height * width, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


POLICY_KWARGS = dict(
    features_extractor_class=SnakeCNN,
    features_extractor_kwargs=dict(features_dim=256),
    normalize_images=False,  # observations are already between 0 and 1
    net_arch=dict(pi=[256], vf=[256]),
)

# Starting points, not tuned yet. Each update uses n_steps moves from every
# environment (4,096 with 16 environments), in minibatches of batch_size.
PPO_KWARGS = dict(
    n_steps=256,
    batch_size=1024,
    n_epochs=4,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
)


def new_model(env, device: str = "auto", tensorboard_log: str | None = None, seed: int | None = None) -> MaskablePPO:
    return MaskablePPO(
        "CnnPolicy",
        env,
        policy_kwargs=POLICY_KWARGS,
        device=device,
        tensorboard_log=tensorboard_log,
        seed=seed,
        verbose=0,
        **PPO_KWARGS,
    )
