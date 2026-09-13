"""Decision logic: given a game state, choose a move.

Milestone 1 placeholder: pick a random move that doesn't kill us on the
very next turn (off the board, or into any snake's body). Later milestones
replace the raw JSON dict with a World object (agent/world.py) and grow
this into a behavior tree.
"""

import random

# How each move changes (x, y). Battlesnake puts (0, 0) at the BOTTOM-left
# corner, so "up" is y + 1 -- the opposite of most screen coordinates.
MOVES = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}


def decide(game_state: dict) -> str:
    board = game_state["board"]
    head = game_state["you"]["head"]

    # Every square covered by any snake (including us). Tails are treated
    # as blocked too, which is slightly too cautious -- a tail usually moves
    # away next turn. Milestone 2 handles that properly.
    occupied = set()
    for snake in board["snakes"]:
        for segment in snake["body"]:
            occupied.add((segment["x"], segment["y"]))

    safe_moves = []
    for move, (dx, dy) in MOVES.items():
        x, y = head["x"] + dx, head["y"] + dy
        on_board = 0 <= x < board["width"] and 0 <= y < board["height"]
        if on_board and (x, y) not in occupied:
            safe_moves.append(move)

    if not safe_moves:
        # Every move is fatal. Still answer quickly rather than crash --
        # a missing reply also counts as a move, so we gain nothing by failing.
        return "up"
    return random.choice(safe_moves)
