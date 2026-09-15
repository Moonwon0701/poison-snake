"""Grid search on the Battlesnake board.

Pure functions: they take plain points, sets and the board size rather than
a World, so they're easy to test on their own.
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


def flood_fill(start: Point, blocked: set[Point], width: int, height: int) -> int:
    """Count the cells reachable from `start` without entering `blocked`.

    `start` itself counts. It's the same breadth-first walk as shortest_path,
    just without a goal: keep spreading until there's nowhere new to go,
    then count everything we visited.
    """
    seen = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for nxt in neighbors(current, width, height):
            if nxt not in seen and nxt not in blocked:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen)


def distance_map(
    starts: list[Point], blocked: set[Point], width: int, height: int
) -> dict[Point, int]:
    """How many moves it takes to reach each cell from the nearest start.

    Every start counts as 1 move away (it's where a head moves next), its
    neighbors as 2, and so on. Starts that are blocked or off the board are
    skipped. With several starts this is one BFS spreading from all of them
    at once, so each cell gets its distance to whichever start is closest.
    """
    dist: dict[Point, int] = {}
    queue = deque()
    for start in starts:
        x, y = start
        if 0 <= x < width and 0 <= y < height and start not in blocked and start not in dist:
            dist[start] = 1
            queue.append(start)
    while queue:
        current = queue.popleft()
        for nxt in neighbors(current, width, height):
            if nxt not in dist and nxt not in blocked:
                dist[nxt] = dist[current] + 1
                queue.append(nxt)
    return dist


def timed_flood_fill(
    start: Point, free_at: dict[Point, int], width: int, height: int
) -> int:
    """Count the cells reachable from `start` when some cells open up later.

    `free_at[cell]` is the move on which that cell becomes free; cells not in
    it are free already. We step onto `start` on move 1 and onto a cell d
    moves away on move d, which is allowed once free_at.get(cell, 0) <= d.
    For snakes that's a body segment moving out of the way before we arrive.
    """
    if free_at.get(start, 0) > 1:
        return 0
    arrival = {start: 1}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for nxt in neighbors(current, width, height):
            move = arrival[current] + 1
            if nxt not in arrival and free_at.get(nxt, 0) <= move:
                arrival[nxt] = move
                queue.append(nxt)
    return len(arrival)


def _walk_back(came_from: dict, start: Point, goal: Point) -> list[Point]:
    """Follow came_from links from the goal back to start, then reverse."""
    path = [goal]
    while came_from[path[-1]] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path
