"""Decision logic: given a World, choose a move.

Milestone 2: survival only. Drop moves that die this turn (walls, bodies)
and, when there's a choice, moves that risk losing a head-to-head
collision. Then pick randomly among what's left. Milestone 3 adds food
seeking; Milestone 5 turns this into a behavior tree.
"""

import random

from agent.world import MOVES, Point, World, step


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


def decide(world: World) -> str:
    moves = safe_moves(world)
    if not moves:
        # Every move is fatal. Still answer quickly rather than crash --
        # a missing reply also counts as a move, so we gain nothing by failing.
        return "up"
    return random.choice(moves)
