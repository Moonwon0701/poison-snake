"""Decision logic: given a World, choose a move.

The decision is a behavior tree (see agent/bt.py). With every option off it
is the Milestone 5 tree:

    Selector "choose a move"
    ├── Sequence "no way out"
    │   ├── Condition "no safe moves"
    │   └── Action "go up anyway"
    └── Sequence "normal turn"
        ├── Action "keep safe moves"      (walls, bodies, head-to-head risk)
        ├── Action "keep roomy moves"     (flood fill: area >= our length)
        └── Selector "pick one"
            ├── Sequence "eat"
            │   ├── Condition "hungry"    (health <= HUNGRY_HEALTH)
            │   └── Action "step toward food"
            └── Action "roomiest side"

Options switch on three head-to-head improvements. Each changes one spot:

- lookahead: a third filter, "keep escapable moves", after "keep roomy moves".
- length_race: "hungry" becomes "needs food", which is also true while
  we're not longer than every enemy.
- hunt: a "hunt" branch between "eat" and "roomiest side" that closes in
  on a nearby shorter snake.

The "keep" actions only narrow down the candidate moves on the blackboard.
The leaves under "pick one" choose among what's left, so food and hunting
can never outrank safety, space or escape routes.
"""

import functools
import random
from dataclasses import dataclass, field
from typing import Any

from agent.bt import Action, Condition, Node, Selector, Sequence
from agent.pathfind import flood_fill, neighbors, shortest_path
from agent.world import MOVES, Point, World, step

# Go for food at this health or below. Health drops by 1 each turn and
# resets to 100 when we eat. The farthest cell on an 11x11 board is 20 moves
# away, so 50 leaves room for detours around bodies, and for a rival taking
# the food first.
HUNGRY_HEALTH = 50

# Hunt a shorter snake whose head is at most this many moves from ours.
HUNT_DISTANCE = 2


@dataclass(frozen=True)
class Options:
    """Which head-to-head improvements the tree uses. All off: Milestone 5."""

    lookahead: bool = False  # prefer moves that leave an escape route next turn
    length_race: bool = False  # also look for food while not the longest snake
    hunt: bool = False  # go after nearby shorter snakes


DEFAULT_OPTIONS = Options()


def distance(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# --- game logic -------------------------------------------------------------


def survivable_moves(world: World) -> list[str]:
    """Moves that don't run into a wall or a snake body."""
    blocked = world.blocked_next_turn()
    moves = []
    for move in MOVES:
        target = step(world.me.head, move)
        if world.in_bounds(target) and target not in blocked:
            moves.append(move)
    return moves


def head_to_head_danger(world: World) -> set[Point]:
    """Cells where an enemy at least as long as us could put its head.

    When two heads land on the same cell, the shorter snake dies, and equal
    lengths kill both. We can't know which way the enemy will turn, so every
    cell next to its head counts as dangerous.
    """
    danger = set()
    for enemy in world.enemies:
        if enemy.length >= world.me.length:
            for move in MOVES:
                danger.add(step(enemy.head, move))
    return danger


def safe_moves(world: World) -> list[str]:
    """Survivable moves, minus head-to-head risks if anything else is left.

    A risky move might still work out (the enemy may turn away); a wall never
    does. So if every survivable move is risky, keep them all.
    """
    survivable = survivable_moves(world)
    danger = head_to_head_danger(world)
    calm = [m for m in survivable if step(world.me.head, m) not in danger]
    return calm or survivable


def reachable_area(world: World, move: str) -> int:
    """How many cells we could still reach after making `move`.

    Bodies are treated as frozen where they'll be after this turn. That's
    cautious: every turn each tail frees another cell, so the real space
    only grows.
    """
    target = step(world.me.head, move)
    return flood_fill(target, world.blocked_next_turn(), world.width, world.height)


def roomy_moves(areas: dict[str, int], length: int) -> list[str]:
    """Moves whose area can hold our whole body; if none can, the roomiest.

    Every turn our head fills one more cell of the area it's in. If the area
    has fewer cells than we are long, we run out of room before our own tail
    has left its current spot, so it's a trap. If every move is a trap, the
    biggest one at least buys the most turns for something to open up.
    """
    roomy = [m for m, area in areas.items() if area >= length]
    if roomy:
        return roomy
    most = max(areas.values())
    return [m for m, area in areas.items() if area == most]


def escape_cells(world: World, move: str) -> list[Point]:
    """After `move`, the cells we could safely move on to the turn after.

    This looks two turns ahead, roughly:

    - Bodies: by then every snake has dropped two tail cells, and our current
      head and the cell we move to have become our body.
    - Enemy heads: an enemy at least as long as us can reach any cell 1 or 2
      moves from its head in that time. At distance 2 our heads could meet.
      At distance 1 its head might arrive first and leave its neck there, so
      we'd hit its body.
    """
    head = world.me.head
    target = step(head, move)
    blocked = {head, target}
    for snake in world.snakes:
        blocked.update(snake.body[:-2])
    threats = [e.head for e in world.enemies if e.length >= world.me.length]
    return [
        cell
        for cell in neighbors(target, world.width, world.height)
        if cell not in blocked and not any(1 <= distance(cell, t) <= 2 for t in threats)
    ]


def move_toward_food(world: World, moves: list[str]) -> str | None:
    """First step of the shortest path to the nearest food, or None.

    The path must start with one of `moves`, so food never outranks safety
    or space: cells next to our head that `moves` ruled out count as blocked.
    Bodies are treated as staying put for the whole path. That's cautious,
    since tails move away while we travel, but it keeps the search simple.
    """
    head = world.me.head
    blocked = world.blocked_next_turn()
    blocked |= {step(head, m) for m in MOVES if m not in moves}
    path = shortest_path(head, world.food, blocked, world.width, world.height)
    if path is None:
        return None
    return next(m for m in moves if step(head, m) == path[0])


def prey(world: World) -> list[Point]:
    """Heads of shorter snakes within HUNT_DISTANCE of our head."""
    return [
        e.head
        for e in world.enemies
        if e.length < world.me.length and distance(e.head, world.me.head) <= HUNT_DISTANCE
    ]


# --- behavior tree ----------------------------------------------------------


@dataclass
class Blackboard:
    """What the tree's nodes share during one turn."""

    world: World
    moves: list[str] = field(default_factory=list)  # candidates still allowed
    areas: dict[str, int] = field(default_factory=dict)  # flood-fill size per move
    move: str | None = None  # the final choice
    rng: Any = random  # anything with .choice(); a seeded one makes games repeatable


def _no_safe_moves(bb: Blackboard) -> bool:
    # A Condition must not write to the blackboard, so "keep safe moves"
    # computes this again. It's a handful of cell checks, so that's cheap.
    return not safe_moves(bb.world)


def _go_up_anyway(bb: Blackboard) -> bool:
    # Every move is fatal. Still answer quickly rather than crash --
    # a missing reply also counts as a move, so we gain nothing by failing.
    bb.move = "up"
    return True


def _keep_safe_moves(bb: Blackboard) -> bool:
    bb.moves = safe_moves(bb.world)
    return bool(bb.moves)


def _keep_roomy_moves(bb: Blackboard) -> bool:
    bb.areas = {m: reachable_area(bb.world, m) for m in bb.moves}
    bb.moves = roomy_moves(bb.areas, bb.world.me.length)
    return True


def _keep_escapable_moves(bb: Blackboard) -> bool:
    # Like the other filters, never narrow down to nothing: if no move keeps
    # an escape route, a risky one is still better than none.
    escapable = [m for m in bb.moves if escape_cells(bb.world, m)]
    if escapable:
        bb.moves = escapable
    return True


def _hungry(bb: Blackboard) -> bool:
    return bb.world.me.health <= HUNGRY_HEALTH


def _needs_food(bb: Blackboard) -> bool:
    # Head-to-head goes to the longer snake, so keep eating until we're the
    # longest on the board.
    world = bb.world
    longest_enemy = max((e.length for e in world.enemies), default=0)
    return _hungry(bb) or world.me.length <= longest_enemy


def _step_toward_food(bb: Blackboard) -> bool:
    bb.move = move_toward_food(bb.world, bb.moves)
    return bb.move is not None


def _shorter_snake_nearby(bb: Blackboard) -> bool:
    return bool(prey(bb.world))


def _close_in(bb: Blackboard) -> bool:
    # Aim for the cells a prey's head could move into next: if it goes there
    # too, the collision kills it, not us.
    world = bb.world
    goals = [
        cell
        for head in prey(world)
        for cell in neighbors(head, world.width, world.height)
    ]
    gaps = {m: min(distance(step(world.me.head, m), g) for g in goals) for m in bb.moves}
    closest = min(gaps.values())
    bb.move = bb.rng.choice([m for m in bb.moves if gaps[m] == closest])
    return True


def _roomiest_side(bb: Blackboard) -> bool:
    most_room = max(bb.areas[m] for m in bb.moves)
    bb.move = bb.rng.choice([m for m in bb.moves if bb.areas[m] == most_room])
    return True


@functools.cache
def build_tree(options: Options) -> Node:
    """Assemble the tree for `options`. Built once per combination."""
    filters: list[Node] = [
        Action("keep safe moves", _keep_safe_moves),
        Action("keep roomy moves", _keep_roomy_moves),
    ]
    if options.lookahead:
        filters.append(Action("keep escapable moves", _keep_escapable_moves))

    wants_food = (
        Condition("needs food", _needs_food)
        if options.length_race
        else Condition("hungry", _hungry)
    )
    pickers: list[Node] = [
        Sequence("eat", [wants_food, Action("step toward food", _step_toward_food)])
    ]
    if options.hunt:
        pickers.append(
            Sequence(
                "hunt",
                [
                    Condition("shorter snake nearby", _shorter_snake_nearby),
                    Action("close in", _close_in),
                ],
            )
        )
    pickers.append(Action("roomiest side", _roomiest_side))

    return Selector(
        "choose a move",
        [
            Sequence(
                "no way out",
                [
                    Condition("no safe moves", _no_safe_moves),
                    Action("go up anyway", _go_up_anyway),
                ],
            ),
            Sequence("normal turn", [*filters, Selector("pick one", pickers)]),
        ],
    )


def decide_with_trace(
    world: World, rng: Any = random, options: Options = DEFAULT_OPTIONS
) -> tuple[str, list[str]]:
    """Run the tree once. Returns the move and the path of nodes that chose it.

    Ties are broken with `rng` (the random module unless you pass a seeded
    random.Random).
    """
    blackboard = Blackboard(world, rng=rng)
    trace: list[str] = []
    build_tree(options).tick(blackboard, trace)
    return blackboard.move, trace


def decide(world: World, rng: Any = random, options: Options = DEFAULT_OPTIONS) -> str:
    return decide_with_trace(world, rng, options)[0]
