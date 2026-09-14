"""BFS shortest path and flood fill on small hand-made grids."""

from agent.pathfind import flood_fill, neighbors, shortest_path


def assert_walkable(start, path, blocked=frozenset(), width=11, height=11):
    """Each step moves exactly one cell, stays on the board, avoids blocked."""
    previous = start
    for x, y in path:
        assert abs(x - previous[0]) + abs(y - previous[1]) == 1
        assert 0 <= x < width and 0 <= y < height
        assert (x, y) not in blocked
        previous = (x, y)


def test_neighbors_in_the_middle():
    assert neighbors((5, 5), 11, 11) == [(5, 6), (5, 4), (4, 5), (6, 5)]


def test_neighbors_in_a_corner():
    assert neighbors((0, 0), 11, 11) == [(0, 1), (1, 0)]


def test_straight_line():
    assert shortest_path((0, 0), {(3, 0)}, set(), 11, 11) == [(1, 0), (2, 0), (3, 0)]


def test_picks_the_nearest_goal():
    path = shortest_path((0, 0), {(5, 0), (0, 2)}, set(), 11, 11)
    assert path == [(0, 1), (0, 2)]


def test_routes_around_a_wall():
    # A wall at x=1 from y=0 to y=3. The goal is right behind it, so the
    # path goes up 4, across 2, and down 4.
    wall = {(1, 0), (1, 1), (1, 2), (1, 3)}
    path = shortest_path((0, 0), {(2, 0)}, wall, 11, 11)
    assert len(path) == 10
    assert path[-1] == (2, 0)
    assert_walkable((0, 0), path, wall)


def test_stays_on_a_small_board():
    # 3x3 board with the centre blocked: 4 moves around the edge.
    path = shortest_path((0, 0), {(2, 2)}, {(1, 1)}, 3, 3)
    assert len(path) == 4
    assert_walkable((0, 0), path, {(1, 1)}, width=3, height=3)


def test_unreachable_goal_returns_none():
    # The goal is in the corner, sealed off by two blocked cells.
    assert shortest_path((5, 5), {(0, 0)}, {(1, 0), (0, 1)}, 11, 11) is None


def test_no_goals_returns_none():
    assert shortest_path((5, 5), set(), set(), 11, 11) is None


def test_flood_fill_counts_the_whole_empty_board():
    assert flood_fill((0, 0), set(), 3, 3) == 9


def test_flood_fill_stops_at_a_wall():
    # A full wall at x=1 cuts a 3x3 board in two: only x=0 is reachable.
    assert flood_fill((0, 0), {(1, 0), (1, 1), (1, 2)}, 3, 3) == 3


def test_flood_fill_goes_around_a_partial_wall():
    # The wall has a gap at the top, so everything but the wall is reachable.
    assert flood_fill((0, 0), {(1, 0), (1, 1)}, 3, 3) == 7


def test_flood_fill_counts_the_start_cell():
    # Boxed in on all sides: just the start cell itself.
    assert flood_fill((1, 1), {(1, 0), (1, 2), (0, 1), (2, 1)}, 3, 3) == 1
