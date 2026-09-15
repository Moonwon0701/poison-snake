"""Reinforcement learning on top of gym_env: MaskablePPO with a curriculum.

- rl.envs: training environments (one per curriculum stage) with action masks
- rl.model: the policy network and PPO settings
- rl.train: train on one stage, optionally continuing from an earlier stage
- rl.evaluate: win rate and survival of a trained model on fresh seeds
- rl.agent: a trained model as a snake policy (game state in, move out)

Needs the extras in requirements-rl.txt.
"""
