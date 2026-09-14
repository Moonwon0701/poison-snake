"""Decision logic: given a World, choose a move.

Milestone 3: survival first, then food. Drop moves that die this turn
(walls, bodies) and, when there's a choice, moves that risk losing a
head-to-head collision. If we're hungry, take the first step of the
shortest path to the nearest food. Otherwise pick randomly among the safe
moves. Milestone 4 adds space control; Milestone 5 turns this into a
behavior tree.
"""

import random

from agent.pathfind import shortest_path
from agent.world import MOVES, Point, World, step

# Go for food at this health or below. Health drops by 1 each turn and
# resets to 100 when we eat. The farthest cell on an 11x11 board is 20 moves
# away, so 50 leaves room for detours around bodies, and for a rival taking
# the food first.
HUNGRY_HEALTH = 50


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


def move_toward_food(world: World, moves: list[str]) -> str | None:
    """First step of the shortest path to the nearest food, or None.

    The path must start with one of `moves`, so food never outranks safety:
    cells next to our head that `moves` ruled out count as blocked. Bodies
    are treated as staying put for the whole path. That's cautious, since
    tails move away while we travel, but it keeps the search simple.
    """
    head = world.me.head
    blocked = world.blocked_next_turn()
    blocked |= {step(head, m) for m in MOVES if m not in moves}
    path = shortest_path(head, world.food, blocked, world.width, world.height)
    if path is None:
        return None
    return next(m for m in moves if step(head, m) == path[0])


def decide(world: World) -> str:
    moves = safe_moves(world)
    if not moves:
        # Every move is fatal. Still answer quickly rather than crash --
        # a missing reply also counts as a move, so we gain nothing by failing.
        return "up"
    if world.me.health <= HUNGRY_HEALTH:
        move = move_toward_food(world, moves)
        if move is not None:
            return move
    return random.choice(moves)
