"""Parse the Battlesnake game-state JSON into a World object.

This module only describes the board: where things are, and what the rules
say will happen when everyone moves. It never decides what to do -- that
lives in agent/strategy.py.
"""

from dataclasses import dataclass

# A board cell as (x, y). Battlesnake puts (0, 0) at the BOTTOM-left corner.
Point = tuple[int, int]

# How each move changes (x, y). "up" is y + 1 -- the opposite of most
# screen coordinates.
MOVES: dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}


def step(point: Point, move: str) -> Point:
    """The cell you reach by making `move` from `point`."""
    dx, dy = MOVES[move]
    return (point[0] + dx, point[1] + dy)


@dataclass
class Snake:
    id: str
    health: int
    body: list[Point]  # head first, tail last

    @property
    def head(self) -> Point:
        return self.body[0]

    @property
    def length(self) -> int:
        return len(self.body)


@dataclass
class World:
    width: int
    height: int
    turn: int
    me: Snake
    enemies: list[Snake]
    food: set[Point]

    @classmethod
    def from_json(cls, game_state: dict) -> "World":
        board = game_state["board"]
        my_id = game_state["you"]["id"]
        return cls(
            width=board["width"],
            height=board["height"],
            turn=game_state["turn"],
            me=_parse_snake(game_state["you"]),
            enemies=[_parse_snake(s) for s in board["snakes"] if s["id"] != my_id],
            food={_parse_point(f) for f in board["food"]},
        )

    @property
    def snakes(self) -> list[Snake]:
        return [self.me, *self.enemies]

    def in_bounds(self, point: Point) -> bool:
        x, y = point
        return 0 <= x < self.width and 0 <= y < self.height

    def blocked_next_turn(self) -> set[Point]:
        """Cells that will still be covered by a snake body after this move.

        All snakes move at once: each head advances one cell and each tail
        cell is dropped. So a tail is safe to step into -- except right after
        that snake ate. Eating adds a segment on top of the tail (the last two
        body entries share a cell), so that cell stays covered for a turn.
        Dropping just the last body entry handles both cases.
        """
        blocked = set()
        for snake in self.snakes:
            blocked.update(snake.body[:-1])
        return blocked


def _parse_point(data: dict) -> Point:
    return (data["x"], data["y"])


def _parse_snake(data: dict) -> Snake:
    return Snake(
        id=data["id"],
        health=data["health"],
        body=[_parse_point(p) for p in data["body"]],
    )
