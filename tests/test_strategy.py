"""Hand-crafted board situations for the survival layer.

Coordinates are (x, y) with (0, 0) at the bottom-left; the board is 11x11
unless a test says otherwise. Bodies are listed head first.
"""

from agent.strategy import decide, safe_moves
from agent.world import MOVES, World
from boards import make_state


def world(my_body, enemies=()):
    return World.from_json(make_state(my_body, enemies=enemies))


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
