"""Hand-crafted board situations for the decision logic.

Coordinates are (x, y) with (0, 0) at the bottom-left; the board is 11x11
unless a test says otherwise. Bodies are listed head first.
"""

import random

from agent.strategy import (
    DEFAULT_OPTIONS,
    HUNGRY_HEALTH,
    Options,
    decide,
    decide_with_trace,
    escape_cells,
    move_toward_food,
    reachable_area,
    roomy_moves,
    safe_moves,
)
from agent.world import MOVES, World
from boards import make_state


def world(my_body, enemies=(), food=(), health=100):
    return World.from_json(
        make_state(my_body, enemies=enemies, food=food, health=health)
    )


# --- walls and bodies -------------------------------------------------------


def test_corner_leaves_only_right():
    # Left and down are walls; up is our own neck.
    w = world([(0, 0), (0, 1), (0, 2)])
    assert safe_moves(w) == ["right"]
    assert decide(w) == "right"


def test_avoids_own_body():
    # Coiled around (5, 5): up, left and right are all our own body.
    w = world([(5, 5), (4, 5), (4, 6), (5, 6), (6, 6), (6, 5), (6, 4)])
    assert safe_moves(w) == ["down"]


def test_avoids_enemy_body():
    # Enemy bodies on both sides; their heads are far from (5, 6).
    w = world(
        [(5, 5), (5, 4), (5, 3)],
        enemies=[[(4, 4), (4, 5), (4, 6)], [(6, 4), (6, 5), (6, 6)]],
    )
    assert safe_moves(w) == ["up"]


# --- tails --------------------------------------------------------------------


def test_can_follow_own_tail():
    # The exact position our Milestone 1 snake died in: top-left corner,
    # boxed in by its own body. The tail at (0, 9) moves away, so down is safe.
    w = world([(0, 10), (1, 10), (1, 9), (0, 9)])
    assert safe_moves(w) == ["down"]


def test_stacked_tail_is_not_safe():
    # Same position, but the snake just ate: the tail stays put this turn.
    w = world([(0, 10), (1, 10), (1, 9), (0, 9), (0, 9)])
    assert safe_moves(w) == []


def test_can_move_into_enemy_tail():
    w = world([(5, 5), (5, 4), (5, 3)], enemies=[[(8, 5), (7, 5), (6, 5)]])
    assert "right" in safe_moves(w)


# --- head-to-head -----------------------------------------------------------
# We are heading right at (5, 5). An enemy head at (7, 5) could also move
# to (6, 5), so "right" risks a head-on collision.


def test_avoids_head_to_head_with_longer_enemy():
    w = world([(5, 5), (4, 5), (3, 5)], enemies=[[(7, 5), (8, 5), (9, 5), (10, 5)]])
    assert set(safe_moves(w)) == {"up", "down"}


def test_avoids_head_to_head_with_equal_enemy():
    # Equal lengths: both snakes die, which is still a loss for us.
    w = world([(5, 5), (4, 5), (3, 5)], enemies=[[(7, 5), (8, 5), (9, 5)]])
    assert set(safe_moves(w)) == {"up", "down"}


def test_allows_head_to_head_with_shorter_enemy():
    # We'd win that collision, so "right" stays on the table.
    w = world([(5, 5), (4, 5), (3, 5)], enemies=[[(7, 5), (8, 5)]])
    assert set(safe_moves(w)) == {"up", "down", "right"}


def test_diagonal_enemy_threatens_two_cells():
    # Enemy head at (6, 6) can reach both (5, 6) and (6, 5).
    w = world([(5, 5), (5, 4), (5, 3)], enemies=[[(6, 6), (7, 6), (8, 6), (9, 6)]])
    assert safe_moves(w) == ["left"]


def test_takes_risky_move_when_it_is_the_only_one():
    # Cornered: "right" is the only way out, and a longer enemy could go
    # there too. A maybe-collision beats a certain wall.
    w = world([(0, 0), (0, 1), (0, 2)], enemies=[[(2, 0), (3, 0), (4, 0), (5, 0)]])
    assert safe_moves(w) == ["right"]
    assert decide(w) == "right"


# --- no way out -------------------------------------------------------------


def test_trapped_still_returns_a_move():
    # Boxed into the corner by our own body, and the tail is stacked.
    w = world([(0, 0), (0, 1), (1, 1), (1, 0), (1, 0)])
    assert safe_moves(w) == []
    assert decide(w) in MOVES


# --- food -------------------------------------------------------------------


def test_hungry_snake_heads_for_food():
    w = world([(5, 5), (5, 4), (5, 3)], food=[(5, 8)], health=HUNGRY_HEALTH)
    assert decide(w) == "up"


def test_hungry_snake_picks_the_nearest_food():
    # (2, 5) is 3 moves left; (5, 10) is 5 moves up.
    w = world([(5, 5), (5, 4), (5, 3)], food=[(2, 5), (5, 10)], health=10)
    assert decide(w) == "left"


def test_hungry_snake_routes_around_its_own_body():
    # Food at (3, 5) is just past our neck at (4, 5). Going around the top
    # takes 4 moves; going around the bottom takes 6.
    w = world([(5, 5), (4, 5), (4, 4), (4, 3)], food=[(3, 5)], health=10)
    assert decide(w) == "up"


def test_well_fed_snake_does_not_chase_food():
    # Food straight up, but above the threshold we still wander randomly.
    w = world([(5, 5), (5, 4), (5, 3)], food=[(5, 8)], health=HUNGRY_HEALTH + 1)
    assert len({decide(w) for _ in range(100)}) > 1


def test_food_never_outranks_safety():
    # Food right next to us, but a longer enemy could move onto it too.
    w = world(
        [(5, 5), (4, 5), (3, 5)],
        enemies=[[(7, 5), (8, 5), (9, 5), (10, 5)]],
        food=[(6, 5)],
        health=10,
    )
    assert decide(w) in {"up", "down"}


def test_unreachable_food_falls_back_to_a_safe_move():
    # Food in the corner is sealed off by an enemy body.
    w = world(
        [(5, 5), (5, 4), (5, 3)],
        enemies=[[(1, 0), (1, 1), (0, 1), (0, 2)]],
        food=[(0, 0)],
        health=10,
    )
    assert move_toward_food(w, safe_moves(w)) is None
    assert decide(w) in safe_moves(w)


# --- space ------------------------------------------------------------------

# A pocket in the bottom-left corner. Our head is at (1, 1) with the body
# going up and bending left, so (0, 2) is sealed. An enemy body ends in a
# stacked tail at (2, 0), sealing the bottom. Left and down both lead into
# the 3-cell pocket {(0, 1), (0, 0), (1, 0)}; right leads to the open board.
POCKET_ME = [(1, 1), (1, 2), (0, 2), (0, 3)]
POCKET_ENEMY = [(5, 1), (4, 1), (3, 1), (3, 0), (2, 0), (2, 0)]


def test_reachable_area_sees_the_pocket():
    w = world(POCKET_ME, enemies=[POCKET_ENEMY])
    assert reachable_area(w, "left") == 3
    assert reachable_area(w, "down") == 3
    assert reachable_area(w, "right") > 50


def test_avoids_a_pocket_smaller_than_our_body():
    # Left, down and right are all safe this turn, but we're 4 long and the
    # pocket only has 3 cells.
    w = world(POCKET_ME, enemies=[POCKET_ENEMY])
    assert set(safe_moves(w)) == {"left", "down", "right"}
    assert decide(w) == "right"


def test_hungry_snake_does_not_follow_food_into_a_pocket():
    w = world(POCKET_ME, enemies=[POCKET_ENEMY], food=[(0, 0)], health=10)
    assert decide(w) == "right"


# A wall made of our own body at x=3, head at the top. The tail is stacked
# so the wall stays closed at the bottom. Left of it: 3 columns x 11 rows =
# 33 cells. Right of it: 7 x 11 = 77 cells. Both can hold our 12-long body.
WALL_ME = [(3, y) for y in range(10, -1, -1)] + [(3, 0)]


def test_well_fed_snake_heads_for_the_bigger_side():
    w = world(WALL_ME)
    assert reachable_area(w, "left") == 33
    assert reachable_area(w, "right") == 77
    assert decide(w) == "right"


def test_hungry_snake_may_eat_on_the_smaller_side_if_it_fits():
    # 33 cells is plenty for 12 segments, so food on the left is fair game.
    w = world(WALL_ME, food=[(1, 5)], health=10)
    assert decide(w) == "left"


def test_roomy_moves_keeps_every_move_that_fits():
    assert roomy_moves({"up": 10, "left": 20}, length=8) == ["up", "left"]


def test_roomy_moves_drops_moves_that_do_not_fit():
    assert roomy_moves({"up": 3, "left": 20}, length=8) == ["left"]


def test_roomy_moves_falls_back_to_the_biggest_trap():
    # Nothing fits; the bigger trap buys more turns.
    assert roomy_moves({"up": 3, "down": 5, "left": 5}, length=8) == ["down", "left"]


# --- which branch of the behavior tree decided --------------------------------


def test_trace_when_there_is_no_way_out():
    w = world([(0, 0), (0, 1), (1, 1), (1, 0), (1, 0)])
    assert decide_with_trace(w) == ("up", ["choose a move", "no way out", "go up anyway"])


def test_trace_when_hungry_and_food_is_reachable():
    w = world([(5, 5), (5, 4), (5, 3)], food=[(5, 8)], health=10)
    move, trace = decide_with_trace(w)
    assert move == "up"
    assert trace == ["choose a move", "normal turn", "pick one", "eat", "step toward food"]


def test_trace_when_hungry_but_food_is_unreachable():
    w = world(
        [(5, 5), (5, 4), (5, 3)],
        enemies=[[(1, 0), (1, 1), (0, 1), (0, 2)]],
        food=[(0, 0)],
        health=10,
    )
    move, trace = decide_with_trace(w)
    assert trace == ["choose a move", "normal turn", "pick one", "roomiest side"]


def test_trace_when_well_fed():
    w = world([(5, 5), (5, 4), (5, 3)], food=[(5, 8)])
    move, trace = decide_with_trace(w)
    assert trace == ["choose a move", "normal turn", "pick one", "roomiest side"]


def test_a_seeded_generator_makes_the_choice_repeatable():
    # Three equally roomy moves; the same seed must pick the same one.
    # The Gymnasium environment relies on this to replay seeded games.
    w = world([(5, 5), (5, 4), (5, 3)])
    assert len({decide(w, random.Random(7)) for _ in range(20)}) == 1


# --- option: lookahead ------------------------------------------------------

# We're at (5, 5), 5 long, with the body bending right under us. An enemy of
# equal length has its head at (7, 6). Up, left and right are all calm this
# turn, but right leads to (6, 5): next turn its exits (7, 5) and (6, 6) sit
# next to the enemy head, and (6, 4) is still our body.
CORNER_ME = [(5, 5), (5, 4), (6, 4), (6, 3), (6, 2)]
CORNER_ENEMY = [(7, 6), (8, 6), (9, 6), (10, 6), (10, 7)]


def test_escape_cells_finds_no_way_out_after_the_trap_move():
    w = world(CORNER_ME, enemies=[CORNER_ENEMY])
    assert set(safe_moves(w)) == {"up", "left", "right"}
    assert escape_cells(w, "right") == []
    assert escape_cells(w, "up") != []
    assert escape_cells(w, "left") != []


def test_lookahead_avoids_the_move_that_gets_cornered():
    w = world(CORNER_ME, enemies=[CORNER_ENEMY])
    moves = {decide(w, random.Random(seed), Options(lookahead=True)) for seed in range(50)}
    assert "right" not in moves


def test_lookahead_keeps_every_move_when_none_can_escape():
    # Boxed into the corner with a single safe move and no exit after it.
    w = world([(0, 0), (0, 1), (0, 2)], enemies=[[(2, 1), (3, 1), (4, 1), (5, 1)]])
    assert escape_cells(w, "right") == []
    assert decide(w, random.Random(0), Options(lookahead=True)) == "right"


# --- option: length race ----------------------------------------------------

FAR_LONG_ENEMY = [(0, 10), (1, 10), (2, 10), (3, 10), (4, 10)]


def test_length_race_eats_while_an_enemy_is_longer():
    w = world([(5, 5), (5, 4), (5, 3)], enemies=[FAR_LONG_ENEMY], food=[(5, 8)])
    move, trace = decide_with_trace(w, random.Random(0), Options(length_race=True))
    assert move == "up"
    assert trace[-2:] == ["eat", "step toward food"]


def test_without_length_race_a_well_fed_snake_ignores_food():
    w = world([(5, 5), (5, 4), (5, 3)], enemies=[FAR_LONG_ENEMY], food=[(5, 8)])
    _, trace = decide_with_trace(w, random.Random(0), Options())
    assert trace[-1] == "roomiest side"


def test_length_race_stops_once_we_are_the_longest():
    w = world([(5, 5), (5, 4), (5, 3)], enemies=[[(0, 10), (1, 10)]], food=[(5, 8)])
    _, trace = decide_with_trace(w, random.Random(0), Options(length_race=True))
    assert trace[-1] == "roomiest side"


# --- option: hunt -----------------------------------------------------------

HUNTER = [(5, 5), (5, 4), (5, 3), (5, 2), (5, 1)]  # 5 long


def test_hunt_moves_onto_a_cell_the_shorter_snake_could_enter():
    # Prey (3 long) has its head at (7, 5); (6, 5) is one of its next cells.
    w = world(HUNTER, enemies=[[(7, 5), (8, 5), (9, 5)]])
    move, trace = decide_with_trace(w, random.Random(0), Options(hunt=True))
    assert move == "right"
    assert trace[-2:] == ["hunt", "close in"]


def test_hunt_ignores_snakes_that_are_too_far_or_not_shorter():
    far = world(HUNTER, enemies=[[(9, 9), (10, 9), (10, 8)]])
    longer = world(HUNTER, enemies=[[(7, 5), (8, 5), (9, 5), (10, 5), (10, 6), (10, 7)]])
    for w in (far, longer):
        _, trace = decide_with_trace(w, random.Random(0), Options(hunt=True))
        assert trace[-1] == "roomiest side"


def test_decide_uses_the_default_options():
    assert DEFAULT_OPTIONS == Options(lookahead=True, length_race=True, hunt=True)
    w = world(HUNTER, enemies=[[(7, 5), (8, 5), (9, 5)]])
    default = decide_with_trace(w, random.Random(1))
    assert default == decide_with_trace(w, random.Random(1), DEFAULT_OPTIONS)
    assert default[1][-2:] == ["hunt", "close in"]
