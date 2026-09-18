"""Opponents for self-play: earlier versions of the agent, mixed with our behavior tree.

Training against one kind of opponent teaches the agent that opponent's
habits. Ours always gives way at a head-to-head it can't win, so the agent
learned to walk into contested cells and take the food; against a snake that
doesn't give way, both would die. Playing earlier copies of itself removes
that shortcut, because the copies exploit it too.

The behavior tree stays in the mix so the agent doesn't forget how to beat
it, and so the win rate against it remains a meaningful yardstick.
"""

import random
from pathlib import Path

from agent.strategy import DEFAULT_OPTIONS, Options
from gym_env.env import Policy, bt_policy


class MixedOpponents:
    """An opponent policy that picks a brain per snake, per game.

    Each snake keeps its brain for a whole game (it's re-drawn on turn 0), so
    a game is played against consistent opponents rather than a committee
    that changes its mind every turn.
    """

    def __init__(
        self,
        model_paths: tuple[str, ...] | list[str] = (),
        bt_share: float = 0.3,
        options: Options = DEFAULT_OPTIONS,
    ):
        if not model_paths:
            bt_share = 1.0
        self.model_paths = [str(p) for p in model_paths]
        self.bt_share = bt_share
        self.behavior_tree = bt_policy(options)
        self._models: dict[str, Policy] = {}
        self._playing: dict[str, Policy] = {}

    def _model(self, path: str) -> Policy:
        if path not in self._models:
            import torch

            from rl.agent import RLPolicy

            # One thread per worker process: 24 environments each grabbing
            # every core would fight over them and run slower.
            torch.set_num_threads(1)
            self._models[path] = RLPolicy(path)
        return self._models[path]

    def _draw(self, rng: random.Random) -> Policy:
        if not self.model_paths or rng.random() < self.bt_share:
            return self.behavior_tree
        return self._model(rng.choice(self.model_paths))

    def __call__(self, game_state: dict, rng: random.Random) -> str:
        snake_id = game_state["you"]["id"]
        if game_state["turn"] == 0 or snake_id not in self._playing:
            self._playing[snake_id] = self._draw(rng)
        return self._playing[snake_id](game_state, rng)


def describe(model_paths, bt_share: float) -> str:
    names = ", ".join(Path(p).parent.name for p in model_paths) or "none"
    return f"{int((1 - bt_share) * 100)}% earlier models ({names}), {int(bt_share * 100)}% behavior tree"
