"""Battlesnake as a Gymnasium environment (Milestone 6).

gym_env.rules is the game itself; gym_env.env.SnakeEnv wraps it for
Gymnasium. Importing this package also registers the environment, so
`gymnasium.make("PoisonSnake-v0", opponents=2)` works.
"""

from gymnasium.envs.registration import register

register(id="PoisonSnake-v0", entry_point="gym_env.env:SnakeEnv")
