"""Decision logic: given a World, choose a move.

Milestone 5: the same decisions as Milestone 4, now laid out as an explicit
behavior tree (see agent/bt.py). The tree is built once, at import:

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

The two "keep" actions only narrow down the candidate moves on the
blackboard; the leaves under "pick one" choose among what's left. So food
can never outrank safety or space.

The helper functions below hold the actual game logic. The tree's leaves
just call them and read or write the blackboard.
"""

import random
from dataclasses import dataclass, field

from agent.bt import Action, Condition, Selector, Sequence
from agent.pathfind import flood_fill, shortest_path
from agent.world import MOVES, Point, World, step

# Go for food at this health or below. Health drops by 1 each turn and
# resets to 100 when we eat. The farthest cell on an 11x11 board is 20 moves
# away, so 50 leaves room for detours around bodies, and for a rival taking
# the food first.
HUNGRY_HEALTH = 50


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


# --- behavior tree ----------------------------------------------------------


@dataclass
class Blackboard:
    """What the tree's nodes share during one turn."""

    world: World
    moves: list[str] = field(default_factory=list)  # candidates still allowed
    areas: dict[str, int] = field(default_factory=dict)  # flood-fill size per move
    move: str | None = None  # the final choice


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


def _hungry(bb: Blackboard) -> bool:
    return bb.world.me.health <= HUNGRY_HEALTH


def _step_toward_food(bb: Blackboard) -> bool:
    bb.move = move_toward_food(bb.world, bb.moves)
    return bb.move is not None


def _roomiest_side(bb: Blackboard) -> bool:
    most_room = max(bb.areas[m] for m in bb.moves)
    bb.move = random.choice([m for m in bb.moves if bb.areas[m] == most_room])
    return True


TREE = Selector(
    "choose a move",
    [
        Sequence(
            "no way out",
            [
                Condition("no safe moves", _no_safe_moves),
                Action("go up anyway", _go_up_anyway),
            ],
        ),
        Sequence(
            "normal turn",
            [
                Action("keep safe moves", _keep_safe_moves),
                Action("keep roomy moves", _keep_roomy_moves),
                Selector(
                    "pick one",
                    [
                        Sequence(
                            "eat",
                            [
                                Condition("hungry", _hungry),
                                Action("step toward food", _step_toward_food),
                            ],
                        ),
                        Action("roomiest side", _roomiest_side),
                    ],
                ),
            ],
        ),
    ],
)


def decide_with_trace(world: World) -> tuple[str, list[str]]:
    """Run the tree once. Returns the move and the path of nodes that chose it."""
    blackboard = Blackboard(world)
    trace: list[str] = []
    TREE.tick(blackboard, trace)
    return blackboard.move, trace


def decide(world: World) -> str:
    return decide_with_trace(world)[0]
