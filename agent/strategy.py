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

Options switch on improvements. Each changes one spot in the tree.

Head-to-head:

- lookahead: a filter, "keep escapable moves", after "keep roomy moves".
- length_race: "hungry" becomes "needs food", which is also true while
  we're not longer than every enemy.
- hunt: a "hunt" branch between "eat" and "roomiest side" that closes in
  on a nearby shorter snake.

Traps:

- timed_area: "keep roomy moves" also counts cells that bodies will have
  left by the time we get there.
- worst_case_filter: a filter, "keep moves that survive enemy moves", that
  wants our body to fit however nearby enemy heads move next.
- voronoi_gate and voronoi_pick: a "measure territory" step counts, for each
  move, the cells we'd reach before any enemy. The gate lets food and hunting
  use only moves whose territory fits our body. The pick replaces
  "roomiest side" with "most territory".

The "keep" actions only narrow down the candidate moves on the blackboard.
The leaves under "pick one" choose among what's left, so food and hunting
can never outrank safety, space or escape routes.
"""

import functools
import itertools
import random
from dataclasses import dataclass, field
from typing import Any

from agent.bt import Action, Condition, Node, Selector, Sequence
from agent.pathfind import distance_map, flood_fill, neighbors, shortest_path, timed_flood_fill
from agent.world import MOVES, Point, World, step

# Go for food at this health or below. Health drops by 1 each turn and
# resets to 100 when we eat. The farthest cell on an 11x11 board is 20 moves
# away, so 50 leaves room for detours around bodies, and for a rival taking
# the food first.
HUNGRY_HEALTH = 50

# Hunt a shorter snake whose head is at most this many moves from ours.
HUNT_DISTANCE = 2

# worst_case_area() only considers enemy heads this close to where we move.
# Farther heads can't cut us off in one move, and every extra enemy
# multiplies the combinations to try by up to 4.
WORST_CASE_RADIUS = 4


@dataclass(frozen=True)
class Options:
    """Which improvements the tree uses. All off: the Milestone 5 tree."""

    lookahead: bool = False  # prefer moves that leave an escape route next turn
    length_race: bool = False  # also look for food while not the longest snake
    hunt: bool = False  # go after nearby shorter snakes
    timed_area: bool = False  # count cells that bodies leave before we arrive
    worst_case_filter: bool = False  # our body must fit whatever nearby enemies do
    voronoi_gate: bool = False  # food and hunting only through moves whose territory fits
    voronoi_pick: bool = False  # otherwise go where our territory is biggest


# Picked with tools/ab_test.py in two rounds, one challenger against three
# other snakes.
# - Head-to-head, against snakes with every option off (2,000 games each):
#   lookahead + length_race + hunt won 57.6%; every option off won 22.4%.
# - Traps, against snakes with those three (every combination screened on
#   1,000 games, the best confirmed on 4,000 fresh seeds): adding
#   voronoi_pick + voronoi_gate + timed_area won 58.1% (95% CI 56.6-59.6%)
#   and was trapped in 26.7% of games, against 24.4% and 50.3% without.
#   worst_case_filter on top did worse (55.4%), so it stays off.
DEFAULT_OPTIONS = Options(
    lookahead=True,
    length_race=True,
    hunt=True,
    timed_area=True,
    voronoi_gate=True,
    voronoi_pick=True,
)


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


def timed_reachable_area(world: World, move: str) -> int:
    """Like reachable_area(), but a body cell opens once it's gone when we arrive.

    Segment i of a snake of length L leaves its cell after L - i moves. A cell
    we'd reach in d moves is open if that has happened by then, so a region
    closed off by our own body can still be big enough if the body unwinds
    ahead of us. Growth from eating isn't predicted.
    """
    target = step(world.me.head, move)
    if not world.in_bounds(target):
        return 0
    free_at: dict[Point, int] = {}
    for snake in world.snakes:
        for i, cell in enumerate(snake.body):
            free_at[cell] = max(free_at.get(cell, 0), len(snake.body) - i)
    return timed_flood_fill(target, free_at, world.width, world.height)


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


def worst_case_area(world: World, move: str, radius: int = WORST_CASE_RADIUS) -> int:
    """The area left after `move` if nearby enemy heads move as badly as possible.

    Each enemy whose head is within `radius` of the cell we move to could
    step onto any free cell next to its head, and that cell is then blocked
    for us. We try every combination and keep the smallest area. The cell we
    move to is left out: meeting there is a head-to-head, which the safety
    filter already handles.
    """
    target = step(world.me.head, move)
    blocked = world.blocked_next_turn()
    choices = []
    for enemy in world.enemies:
        if distance(enemy.head, target) <= radius:
            cells = [
                c
                for c in neighbors(enemy.head, world.width, world.height)
                if c not in blocked and c != target
            ]
            if cells:
                choices.append(cells)
    return min(
        flood_fill(target, blocked | set(combo), world.width, world.height)
        for combo in itertools.product(*choices)  # one empty combo if no one is near
    )


def rival_distances(world: World) -> dict[Point, int]:
    """For each cell, the fewest moves any enemy head needs to get there.

    An enemy's next cell counts as 1 move, the same way territory() counts
    ours. Bodies are treated as staying where they'll be after this turn.
    """
    starts = [
        cell
        for enemy in world.enemies
        for cell in neighbors(enemy.head, world.width, world.height)
    ]
    return distance_map(starts, world.blocked_next_turn(), world.width, world.height)


def territory(world: World, move: str, rivals: dict[Point, int]) -> int:
    """After `move`, how many cells we'd reach strictly before every enemy.

    This is our Voronoi area: the board divided up by whichever head can get
    to each cell first. Cells an enemy reaches at the same time don't count,
    since arriving together means a head-to-head. `rivals` comes from
    rival_distances(), computed once per turn.
    """
    target = step(world.me.head, move)
    mine = distance_map([target], world.blocked_next_turn(), world.width, world.height)
    return sum(1 for cell, d in mine.items() if cell not in rivals or d < rivals[cell])


def food_path(world: World, moves: list[str]) -> list[Point] | None:
    """Shortest path from our head to the nearest food, or None.

    The path must start with one of `moves`, so food never outranks safety
    or space: cells next to our head that `moves` ruled out count as blocked.
    Bodies are treated as staying put for the whole path. That's cautious,
    since tails move away while we travel, but it keeps the search simple.
    """
    head = world.me.head
    blocked = world.blocked_next_turn()
    blocked |= {step(head, m) for m in MOVES if m not in moves}
    return shortest_path(head, world.food, blocked, world.width, world.height)


def move_toward_food(world: World, moves: list[str]) -> str | None:
    """First step of food_path(), or None if no food can be reached."""
    path = food_path(world, moves)
    if path is None:
        return None
    return next(m for m in moves if step(world.me.head, m) == path[0])


def prey(world: World) -> list[Point]:
    """Heads of shorter snakes within HUNT_DISTANCE of our head."""
    return [
        e.head
        for e in world.enemies
        if e.length < world.me.length and distance(e.head, world.me.head) <= HUNT_DISTANCE
    ]


def hunt_goals(world: World) -> list[Point]:
    """The cells our prey's heads could move into next.

    If a prey moves onto one of these cells together with us, the collision
    kills it, not us.
    """
    return [
        cell
        for head in prey(world)
        for cell in neighbors(head, world.width, world.height)
    ]


# --- behavior tree ----------------------------------------------------------


# The filters, as named in Blackboard.stages and explain().
SAFETY, SPACE, ESCAPE, WORST_CASE = "safety", "space", "escape", "worst case"


@dataclass
class Blackboard:
    """What the tree's nodes share during one turn."""

    world: World
    moves: list[str] = field(default_factory=list)  # candidates still allowed
    areas: dict[str, int] = field(default_factory=dict)  # flood-fill size per move
    worst: dict[str, int] = field(default_factory=dict)  # worst-case area per move
    territory: dict[str, int] = field(default_factory=dict)  # Voronoi area per move
    food_path: list[Point] | None = None  # the path "step toward food" followed
    move: str | None = None  # the final choice
    rng: Any = random  # anything with .choice(); a seeded one makes games repeatable
    # The moves each filter kept, in order, as (stage, moves). Only explain()
    # reads this; it's how the viewer knows which filter dropped which move.
    stages: list[tuple[str, list[str]]] = field(default_factory=list)


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
    bb.stages.append((SAFETY, bb.moves))
    return bool(bb.moves)


def _keep_roomy(bb: Blackboard, area_of) -> bool:
    bb.areas = {m: area_of(bb.world, m) for m in bb.moves}
    bb.moves = roomy_moves(bb.areas, bb.world.me.length)
    bb.stages.append((SPACE, bb.moves))
    return True


def _keep_roomy_moves(bb: Blackboard) -> bool:
    return _keep_roomy(bb, reachable_area)


def _keep_roomy_moves_timed(bb: Blackboard) -> bool:
    return _keep_roomy(bb, timed_reachable_area)


def _keep_escapable_moves(bb: Blackboard) -> bool:
    # Like the other filters, never narrow down to nothing: if no move keeps
    # an escape route, a risky one is still better than none.
    escapable = [m for m in bb.moves if escape_cells(bb.world, m)]
    if escapable:
        bb.moves = escapable
    bb.stages.append((ESCAPE, bb.moves))
    return True


def _keep_moves_that_survive_enemy_moves(bb: Blackboard) -> bool:
    bb.worst = {m: worst_case_area(bb.world, m) for m in bb.moves}
    fits = [m for m in bb.moves if bb.worst[m] >= bb.world.me.length]
    if fits:  # never narrow down to nothing
        bb.moves = fits
    bb.stages.append((WORST_CASE, bb.moves))
    return True


def _measure_territory(bb: Blackboard) -> bool:
    rivals = rival_distances(bb.world)  # the enemies' side, once for all moves
    bb.territory = {m: territory(bb.world, m, rivals) for m in bb.moves}
    return True


def _moves_with_territory_to_spare(bb: Blackboard) -> list[str]:
    return [m for m in bb.moves if bb.territory[m] >= bb.world.me.length]


def _hungry(bb: Blackboard) -> bool:
    return bb.world.me.health <= HUNGRY_HEALTH


def _needs_food(bb: Blackboard) -> bool:
    # Head-to-head goes to the longer snake, so keep eating until we're the
    # longest on the board.
    world = bb.world
    longest_enemy = max((e.length for e in world.enemies), default=0)
    return _hungry(bb) or world.me.length <= longest_enemy


def _step_toward_food_among(bb: Blackboard, moves: list[str]) -> bool:
    if not moves:
        return False
    path = food_path(bb.world, moves)
    if path is None:
        return False
    bb.food_path = path
    bb.move = next(m for m in moves if step(bb.world.me.head, m) == path[0])
    return True


def _step_toward_food(bb: Blackboard) -> bool:
    return _step_toward_food_among(bb, bb.moves)


def _step_toward_food_gated(bb: Blackboard) -> bool:
    # Food inside space an enemy controls is where traps close, so only
    # start down a path whose first move leaves us territory to spare.
    return _step_toward_food_among(bb, _moves_with_territory_to_spare(bb))


def _shorter_snake_nearby(bb: Blackboard) -> bool:
    return bool(prey(bb.world))


def _close_in_among(bb: Blackboard, moves: list[str]) -> bool:
    if not moves:
        return False
    world = bb.world
    goals = hunt_goals(world)
    gaps = {m: min(distance(step(world.me.head, m), g) for g in goals) for m in moves}
    closest = min(gaps.values())
    bb.move = bb.rng.choice([m for m in moves if gaps[m] == closest])
    return True


def _close_in(bb: Blackboard) -> bool:
    return _close_in_among(bb, bb.moves)


def _close_in_gated(bb: Blackboard) -> bool:
    return _close_in_among(bb, _moves_with_territory_to_spare(bb))


def _roomiest_side(bb: Blackboard) -> bool:
    most_room = max(bb.areas[m] for m in bb.moves)
    bb.move = bb.rng.choice([m for m in bb.moves if bb.areas[m] == most_room])
    return True


def _most_territory(bb: Blackboard) -> bool:
    # Most territory first; among equals, the most open area; then chance.
    best = max((bb.territory[m], bb.areas[m]) for m in bb.moves)
    bb.move = bb.rng.choice([m for m in bb.moves if (bb.territory[m], bb.areas[m]) == best])
    return True


@functools.cache
def build_tree(options: Options) -> Node:
    """Assemble the tree for `options`. Built once per combination."""
    gated = options.voronoi_gate
    prepare: list[Node] = [
        Action("keep safe moves", _keep_safe_moves),
        Action(
            "keep roomy moves",
            _keep_roomy_moves_timed if options.timed_area else _keep_roomy_moves,
        ),
    ]
    if options.lookahead:
        prepare.append(Action("keep escapable moves", _keep_escapable_moves))
    if options.worst_case_filter:
        prepare.append(
            Action("keep moves that survive enemy moves", _keep_moves_that_survive_enemy_moves)
        )
    if options.voronoi_gate or options.voronoi_pick:
        prepare.append(Action("measure territory", _measure_territory))

    wants_food = (
        Condition("needs food", _needs_food)
        if options.length_race
        else Condition("hungry", _hungry)
    )
    pickers: list[Node] = [
        Sequence(
            "eat",
            [
                wants_food,
                Action("step toward food", _step_toward_food_gated if gated else _step_toward_food),
            ],
        )
    ]
    if options.hunt:
        pickers.append(
            Sequence(
                "hunt",
                [
                    Condition("shorter snake nearby", _shorter_snake_nearby),
                    Action("close in", _close_in_gated if gated else _close_in),
                ],
            )
        )
    if options.voronoi_pick:
        pickers.append(Action("most territory", _most_territory))
    else:
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
            Sequence("normal turn", [*prepare, Selector("pick one", pickers)]),
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


# --- explaining a decision --------------------------------------------------


def explain(world: World, rng: Any = random, options: Options = DEFAULT_OPTIONS) -> dict:
    """Run the tree like decide(), and report why it chose what it did.

    Returns plain JSON-ready data for tools/explain_game.py and the viewer:

    - move, trace: as from decide_with_trace().
    - visits: {node id: "success" or "failure"} for every node that ran. Ids
      match build_tree(options).describe(); nodes that didn't run are absent.
    - moves: for each of the four moves, a verdict ("chosen", "candidate" for
      kept but not picked, or "eliminated"), the filter that dropped it
      (SAFETY, SPACE, ESCAPE, WORST_CASE or None) and a short reason.
    - areas: flood-fill size for each move that passed the safety filter.
    - worst_case: worst-case area for each move the worst-case filter checked.
    - territory: Voronoi area for each move, if the tree measured it.
    - danger: cells next to an enemy head at least as long as us.
    - escape: exit cells for each move the lookahead filter looked at.
    - food_path: the path to food, if the tree stepped toward food.
    - hunt_goals: the cells the hunt branch aimed for, if it ran.
    """
    tree = build_tree(options)
    bb = Blackboard(world, rng=rng)
    trace: list[str] = []
    visits: list = []
    tree.tick(bb, trace, visits)
    ids = {node: i for i, node in enumerate(tree.walk())}
    ran = {node.name for node, _ in visits}

    moves: dict[str, dict] = {}
    remaining = list(MOVES)
    for stage, kept in bb.stages:
        for move in remaining:
            if move not in kept:
                reason = _elimination_reason(bb, stage, move)
                moves[move] = {"verdict": "eliminated", "stage": stage, "reason": reason}
        remaining = kept
    if not bb.stages:  # "no way out": every move failed the safety check
        for move in MOVES:
            reason = _elimination_reason(bb, SAFETY, move)
            moves[move] = {"verdict": "eliminated", "stage": SAFETY, "reason": reason}
        remaining = []
    for move in remaining:
        moves[move] = {"verdict": "candidate", "stage": None, "reason": "passed every filter"}
    chosen = moves[bb.move]
    if chosen["verdict"] == "eliminated":
        chosen["reason"] = f"no safe move ({chosen['reason']}), {trace[-1]}"
    else:
        chosen["reason"] = trace[-1]
    chosen["verdict"] = "chosen"

    escape = {}
    if ESCAPE in (stage for stage, _ in bb.stages):
        considered = next(kept for stage, kept in bb.stages if stage == SPACE)
        escape = {m: _cells(escape_cells(world, m)) for m in considered}

    return {
        "move": bb.move,
        "trace": trace,
        "visits": {str(ids[node]): status.value for node, status in visits},
        "moves": {m: moves[m] for m in MOVES},
        "areas": dict(bb.areas),
        "worst_case": dict(bb.worst),
        "territory": dict(bb.territory),
        "danger": _cells(sorted(head_to_head_danger(world))),
        "escape": escape,
        "food_path": _cells(bb.food_path) if bb.food_path is not None else None,
        "hunt_goals": _cells(hunt_goals(world)) if "close in" in ran else [],
    }


def _elimination_reason(bb: Blackboard, stage: str, move: str) -> str:
    world = bb.world
    target = step(world.me.head, move)
    if stage == SAFETY:
        if not world.in_bounds(target):
            return "wall"
        if target in world.blocked_next_turn():
            return "body"
        return "next to an enemy head at least as long as us"
    if stage == SPACE:
        area, most = bb.areas[move], max(bb.areas.values())
        if most >= world.me.length:
            return f"only {area} cells to move in, body is {world.me.length} long"
        return f"{area} cells, fewer than the {most} another move has"
    if stage == WORST_CASE:
        return (
            f"only {bb.worst[move]} cells if nearby enemies move badly,"
            f" body is {world.me.length} long"
        )
    return "no safe exit the turn after"


def _cells(points) -> list[list[int]]:
    return [list(p) for p in points]
