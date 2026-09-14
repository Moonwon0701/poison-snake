"""Simplified Battlesnake rules in pure Python, for simulation and RL.

Follows the order of the official engine (github.com/BattlesnakeOfficial/
rules: standard.go and maps/standard.go). Every turn:

1. Move: every snake adds a head in its chosen direction and drops its tail.
2. Starve: every snake loses 1 health.
3. Feed: a head on food eats it. Health goes back to 100, and the tail cell
   is doubled, so the snake is one longer from next turn on.
4. Eliminate: first snakes out of health or off the board. Then, all at
   once: heads that hit a body (their own or another snake's), and heads
   that met a head at least as long.
5. Spawn: top the food back up to the minimum, or sometimes add one.

Left out: hazards, other game modes, and the engine's random numbers. The
same seed gives a different game than the CLI would, under the same rules.

Like the JSON the engine sends to snakes, a GameState holds only the snakes
still alive. Nothing here knows about Gymnasium.
"""

import random
from dataclasses import dataclass, field

from agent.world import MOVES, Point
from agent.world import step as next_cell

MAX_HEALTH = 100
START_LENGTH = 3

# Why a snake was eliminated, using the official engine's names.
OUT_OF_HEALTH = "out-of-health"
WALL_COLLISION = "wall-collision"
SELF_COLLISION = "snake-self-collision"
BODY_COLLISION = "snake-collision"
HEAD_COLLISION = "head-collision"


@dataclass
class Snake:
    id: str
    body: list[Point]  # head first; a snake that just ate repeats its tail cell
    health: int = MAX_HEALTH

    @property
    def head(self) -> Point:
        return self.body[0]


@dataclass
class GameState:
    """A board mid-game. Holds only the snakes still alive."""

    width: int
    height: int
    snakes: list[Snake]
    food: list[Point] = field(default_factory=list)
    turn: int = 0

    def copy(self) -> "GameState":
        snakes = [Snake(s.id, list(s.body), s.health) for s in self.snakes]
        return GameState(self.width, self.height, snakes, list(self.food), self.turn)

    def snake(self, snake_id: str) -> Snake | None:
        return next((s for s in self.snakes if s.id == snake_id), None)

    def in_bounds(self, point: Point) -> bool:
        x, y = point
        return 0 <= x < self.width and 0 <= y < self.height


# --- a new game -------------------------------------------------------------


def new_game(
    snake_ids: list[str], rng: random.Random, width: int = 11, height: int = 11
) -> GameState:
    """Lay out a fresh game like the official "standard" map.

    Supports square boards of 7x7 or larger with up to 8 snakes, which covers
    the usual 11x11 games. Snakes start coiled up (all 3 segments on one cell)
    on a corner or edge-midpoint spot. Each gets a food 2 moves away, and one
    more goes in the centre.
    """
    if width != height or width < 7:
        raise ValueError("only square boards of 7x7 or larger are supported")
    if len(snake_ids) > 8:
        raise ValueError("at most 8 snakes fit the fixed start positions")
    state = GameState(width, height, _place_snakes(snake_ids, width, rng))
    state.food = _place_food(state, rng)
    return state


def _place_snakes(snake_ids: list[str], size: int, rng: random.Random) -> list[Snake]:
    low, mid, high = 1, (size - 1) // 2, size - 2
    corners = [(low, low), (low, high), (high, low), (high, high)]
    edges = [(low, mid), (mid, low), (mid, high), (high, mid)]
    rng.shuffle(corners)
    rng.shuffle(edges)
    # A coin flip decides whether the first snakes get corners or edges.
    starts = corners + edges if rng.randrange(2) == 0 else edges + corners
    return [Snake(sid, [start] * START_LENGTH) for sid, start in zip(snake_ids, starts)]


def _place_food(state: GameState, rng: random.Random) -> list[Point]:
    cx, cy = center = ((state.width - 1) // 2, (state.height - 1) // 2)
    food: list[Point] = []
    small_board = state.width * state.height < 11 * 11
    if len(state.snakes) <= 4 or not small_board:
        for snake in state.snakes:
            # One food on a diagonal next to the head: not the centre, not a
            # corner, and farther from the centre than the head on some axis.
            hx, hy = snake.head
            options = []
            for x, y in [(hx - 1, hy - 1), (hx - 1, hy + 1), (hx + 1, hy - 1), (hx + 1, hy + 1)]:
                away_from_center = x < hx < cx or cx < hx < x or y < hy < cy or cy < hy < y
                in_corner = x in (0, state.width - 1) and y in (0, state.height - 1)
                if (x, y) != center and (x, y) not in food and away_from_center and not in_corner:
                    options.append((x, y))
            food.append(rng.choice(options))
    food.append(center)
    return food


# --- one turn ---------------------------------------------------------------


def step(
    state: GameState,
    moves: dict[str, str],
    rng: random.Random | None = None,
    food_spawn_chance: int = 15,
    minimum_food: int = 1,
) -> tuple[GameState, dict[str, str]]:
    """Play one turn. Returns the next state and the snakes eliminated in it.

    `moves` maps snake id to "up", "down", "left" or "right". A snake without
    a move keeps going the way it faces, as in the official engine. The
    eliminations come back as {snake id: cause}. `state` itself is left
    unchanged. `rng` is only needed when food may spawn.
    """
    nxt = state.copy()
    for snake in nxt.snakes:
        move = moves.get(snake.id) or default_move(snake)
        snake.body = [next_cell(snake.head, move)] + snake.body[:-1]
    for snake in nxt.snakes:
        snake.health -= 1
    _feed(nxt)
    eliminated = _eliminations(nxt)
    nxt.snakes = [s for s in nxt.snakes if s.id not in eliminated]
    nxt.turn += 1
    _spawn_food(nxt, rng, food_spawn_chance, minimum_food)
    return nxt, eliminated


def default_move(snake: Snake) -> str:
    """The engine's fallback: keep going away from the neck, or up if coiled."""
    if len(snake.body) >= 2:
        dx = snake.body[0][0] - snake.body[1][0]
        dy = snake.body[0][1] - snake.body[1][1]
        for move, delta in MOVES.items():
            if delta == (dx, dy):
                return move
    return "up"


def _feed(state: GameState) -> None:
    remaining = []
    for food in state.food:
        eaters = [s for s in state.snakes if s.head == food]
        for snake in eaters:  # several heads on one food all eat
            snake.health = MAX_HEALTH
            snake.body.append(snake.body[-1])
        if not eaters:
            remaining.append(food)
    state.food = remaining


def _eliminations(state: GameState) -> dict[str, str]:
    eliminated = {}
    for snake in state.snakes:
        if snake.health <= 0:
            eliminated[snake.id] = OUT_OF_HEALTH
        elif not state.in_bounds(snake.head):
            eliminated[snake.id] = WALL_COLLISION

    # Collisions are judged against everyone who survived the checks above,
    # and applied together afterwards, so two snakes can take each other out.
    # Head-to-head uses lengths after feeding, but a food under two heads
    # feeds both, so eating never changes who wins.
    survivors = [s for s in state.snakes if s.id not in eliminated]
    collisions = {}
    for snake in survivors:
        others = [o for o in survivors if o is not snake]
        if snake.head in snake.body[1:]:
            collisions[snake.id] = SELF_COLLISION
        elif any(snake.head in o.body[1:] for o in others):
            collisions[snake.id] = BODY_COLLISION
        elif any(snake.head == o.head and len(snake.body) <= len(o.body) for o in others):
            collisions[snake.id] = HEAD_COLLISION
    eliminated.update(collisions)
    return eliminated


def _spawn_food(
    state: GameState, rng: random.Random | None, chance: int, minimum: int
) -> None:
    if len(state.food) < minimum:
        needed = minimum - len(state.food)
    elif chance > 0 and 100 - rng.randrange(100) < chance:
        # The engine's own formula. It works out to (chance - 1)% a turn.
        needed = 1
    else:
        return
    occupied = set(state.food) | {p for s in state.snakes for p in s.body}
    free = [
        (x, y)
        for x in range(state.width)
        for y in range(state.height)
        if (x, y) not in occupied
    ]
    state.food += rng.sample(free, min(needed, len(free)))


# --- helpers ----------------------------------------------------------------


def non_colliding_moves(state: GameState, snake_id: str) -> list[str]:
    """Moves that don't run into a wall or a body this turn.

    Every tail moves out of the way, unless its snake just ate (a doubled
    tail cell), so each snake's last body entry doesn't block. Head-to-head
    collisions depend on what the other snake does, so they aren't ruled out.
    """
    snake = state.snake(snake_id)
    if snake is None:
        return []
    blocked = {p for s in state.snakes for p in s.body[:-1]}
    return [
        move
        for move in MOVES
        if state.in_bounds(next_cell(snake.head, move))
        and next_cell(snake.head, move) not in blocked
    ]


def to_api_json(state: GameState, you_id: str) -> dict:
    """The state as the engine's /move request body, seen by snake `you_id`."""

    def point(p: Point) -> dict:
        return {"x": p[0], "y": p[1]}

    snakes = [
        {
            "id": s.id,
            "name": s.id,
            "health": s.health,
            "body": [point(p) for p in s.body],
            "head": point(s.head),
            "length": len(s.body),
        }
        for s in state.snakes
    ]
    return {
        "game": {"id": "simulation"},
        "turn": state.turn,
        "board": {
            "width": state.width,
            "height": state.height,
            "food": [point(p) for p in state.food],
            "hazards": [],
            "snakes": snakes,
        },
        "you": next(s for s in snakes if s["id"] == you_id),
    }


def from_api_json(game_state: dict) -> GameState:
    """Read a game state in the engine's JSON format, e.g. from a CLI log."""
    board = game_state["board"]
    return GameState(
        width=board["width"],
        height=board["height"],
        snakes=[
            Snake(s["id"], [(p["x"], p["y"]) for p in s["body"]], s["health"])
            for s in board["snakes"]
        ],
        food=[(f["x"], f["y"]) for f in board["food"]],
        turn=game_state["turn"],
    )
