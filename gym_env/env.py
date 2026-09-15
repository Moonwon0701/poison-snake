"""A Gymnasium environment around gym_env.rules.

We control one snake, with id "agent", through step(action). Up to three
opponents move by a policy, our behavior tree unless you pass another, so
this looks like an ordinary single-agent Gymnasium environment:

    env = SnakeEnv(opponents=2)
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

import random
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from agent.strategy import DEFAULT_OPTIONS, Options, decide
from agent.world import MOVES, World
from gym_env import rules

AGENT = "agent"
ACTIONS = list(MOVES)  # action index -> move: 0 up, 1 down, 2 left, 3 right

# Observation channels, indexed obs[channel, y, x].
MY_HEAD, MY_BODY, ENEMY_HEADS, ENEMY_BODIES, FOOD, MY_HEALTH = range(6)
CHANNELS = 6

DEFAULT_REWARDS = {
    "survive": 0.01,  # each turn our snake is still alive
    "eat": 0.1,  # on a turn we ate
    "death": -1.0,  # we were eliminated; a draw counts as this too
    "win": 1.0,  # the last opponent went out and we didn't
}

# An opponent policy gets the engine-style game state (as that snake sees it)
# and the environment's random generator, and returns a move.
Policy = Callable[[dict, random.Random], str]


def behavior_tree_policy(game_state: dict, rng: random.Random) -> str:
    """Our behavior-tree snake with its default options."""
    return decide(World.from_json(game_state), rng)


def bt_policy(options: Options = DEFAULT_OPTIONS) -> Policy:
    """Our behavior-tree snake with the given options, e.g. Options() for a weak one."""

    def policy(game_state: dict, rng: random.Random) -> str:
        return decide(World.from_json(game_state), rng, options)

    return policy


def encode_observation(state: rules.GameState, agent_id: str = AGENT) -> np.ndarray:
    """The board as a (6, height, width) float32 array, indexed [channel, y, x].

    y grows upward like on the Battlesnake board, so row 0 is the bottom.

    Body cells say how soon they free up. Segment i of a snake of length L
    leaves its cell after L - i more moves, stored as (L - i) / L: close to 1
    at the neck, small at the tail. A doubled tail cell keeps the larger
    value, since it stays one turn longer.
    """
    obs = np.zeros((CHANNELS, state.height, state.width), dtype=np.float32)
    me = state.snake(agent_id)
    my_length = len(me.body) if me else 0
    for snake in state.snakes:
        mine = snake.id == agent_id
        length = len(snake.body)
        body_channel = MY_BODY if mine else ENEMY_BODIES
        for i, (x, y) in enumerate(snake.body[1:], start=1):
            obs[body_channel, y, x] = max(obs[body_channel, y, x], (length - i) / length)
        x, y = snake.head
        if mine:
            obs[MY_HEAD, y, x] = 1.0
        else:
            # 1 marks a head that would beat or tie us head-to-head; 0.5 one we'd beat.
            obs[ENEMY_HEADS, y, x] = 1.0 if length >= my_length else 0.5
    for x, y in state.food:
        obs[FOOD, y, x] = 1.0
    if me:
        obs[MY_HEALTH] = me.health / rules.MAX_HEALTH
    return obs


def action_mask(state: rules.GameState, agent_id: str = AGENT) -> np.ndarray:
    """1 for each action that doesn't hit a wall or a body this turn."""
    allowed = rules.non_colliding_moves(state, agent_id)
    return np.array([move in allowed for move in ACTIONS], dtype=np.int8)


def render_text(state: rules.GameState, agent_id: str = AGENT) -> str:
    """The board as text, top row first. A/a is us, O/o opponents, * food."""
    grid = [["."] * state.width for _ in range(state.height)]
    for x, y in state.food:
        grid[y][x] = "*"
    for snake in state.snakes:
        mine = snake.id == agent_id
        for x, y in snake.body[1:]:
            grid[y][x] = "a" if mine else "o"
        x, y = snake.head
        grid[y][x] = "A" if mine else "O"
    me = state.snake(agent_id)
    header = f"turn {state.turn}, health {me.health if me else 'eliminated'}"
    return "\n".join([header, *(" ".join(row) for row in reversed(grid))])


class SnakeEnv(gym.Env):
    """Battlesnake from one snake's point of view.

    - Actions: Discrete(4), see ACTIONS.
    - Observations: see encode_observation. info["action_mask"] marks the
      actions that don't hit a wall or a body.
    - Rewards: DEFAULT_REWARDS, overridable per key through `rewards`.
    - terminated: our snake was eliminated, or it outlived every opponent.
      truncated: `max_turns` steps since reset.
    - reset(options={"state": GameState}) starts from any position that has a
      snake with id "agent".
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        opponents: int = 0,
        size: int = 11,
        max_turns: int = 1000,
        rewards: dict[str, float] | None = None,
        opponent_policy: Policy = behavior_tree_policy,
        food_spawn_chance: int = 15,
        minimum_food: int = 1,
        render_mode: str | None = None,
    ):
        if not 0 <= opponents <= 3:
            raise ValueError("opponents must be between 0 and 3")
        self.opponents = opponents
        self.size = size
        self.max_turns = max_turns
        self.rewards = {**DEFAULT_REWARDS, **(rewards or {})}
        self.opponent_policy = opponent_policy
        self.food_spawn_chance = food_spawn_chance
        self.minimum_food = minimum_food
        self.render_mode = render_mode
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(CHANNELS, size, size), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.state: rules.GameState | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # One Python generator drives the rules and the opponents. It's seeded
        # from Gymnasium's, so reset(seed=...) makes the whole episode repeat.
        self.rng = random.Random(int(self.np_random.integers(2**32)))
        if options and "state" in options:
            self.state = options["state"].copy()
            if self.state.snake(AGENT) is None:
                raise ValueError(f'the starting state needs a snake with id "{AGENT}"')
        else:
            ids = [AGENT] + [f"opponent{i}" for i in range(1, self.opponents + 1)]
            self.state = rules.new_game(ids, self.rng, self.size, self.size)
        self._had_opponents = len(self.state.snakes) > 1
        self._steps = 0
        return encode_observation(self.state), self._info()

    def step(self, action):
        if self.state is None or self.state.snake(AGENT) is None:
            raise RuntimeError("the episode is over; call reset() first")
        moves = {AGENT: ACTIONS[int(action)]}
        for snake in self.state.snakes:
            if snake.id != AGENT:
                seen_by_them = rules.to_api_json(self.state, snake.id)
                moves[snake.id] = self.opponent_policy(seen_by_them, self.rng)
        self.state, eliminated = rules.step(
            self.state, moves, self.rng, self.food_spawn_chance, self.minimum_food
        )
        self._steps += 1

        me = self.state.snake(AGENT)
        won = me is not None and self._had_opponents and len(self.state.snakes) == 1
        if me is None:
            reward = self.rewards["death"]
        else:
            reward = self.rewards["survive"]
            if me.health == rules.MAX_HEALTH:  # only eating resets health
                reward += self.rewards["eat"]
            if won:
                reward += self.rewards["win"]

        terminated = me is None or won
        truncated = not terminated and self._steps >= self.max_turns
        info = self._info()
        info["won"] = won
        if AGENT in eliminated:
            info["cause"] = eliminated[AGENT]
        return encode_observation(self.state), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return render_text(self.state)
        return None

    def _info(self) -> dict:
        return {"turn": self.state.turn, "action_mask": action_mask(self.state)}
