"""Grid search on the Battlesnake board.

Pure functions: they take plain points, sets and the board size rather than
a World, so they're easy to test on their own. Milestone 4 adds flood fill.
"""

from collections import deque

from agent.world import MOVES, Point, step


def neighbors(point: Point, width: int, height: int) -> list[Point]:
    """On-board cells one move away, in MOVES order (up, down, left, right)."""
    result = []
    for move in MOVES:
        x, y = step(point, move)
        if 0 <= x < width and 0 <= y < height:
            result.append((x, y))
    return result


def shortest_path(
    start: Point, goals: set[Point], blocked: set[Point], width: int, height: int
) -> list[Point] | None:
    """Breadth-first search from `start` to the nearest cell in `goals`.

    Returns the cells to walk through, from the first step up to and
    including the goal, so `path[0]` is where to move next. Returns None if
    no goal can be reached without entering a `blocked` cell.

    BFS explores in rings of growing distance: every cell 1 move away, then
    every cell 2 moves away, and so on. So the first goal it reaches is a
    nearest one, and the route it took to get there is a shortest one.
    """
    came_from: dict[Point, Point | None] = {start: None}  # cell -> previous cell
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for nxt in neighbors(current, width, height):
            if nxt in came_from or nxt in blocked:
                continue
            came_from[nxt] = current
            if nxt in goals:
                return _walk_back(came_from, start, nxt)
            queue.append(nxt)
    return None


def _walk_back(came_from: dict, start: Point, goal: Point) -> list[Point]:
    """Follow came_from links from the goal back to start, then reverse."""
    path = [goal]
    while came_from[path[-1]] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path
