"""Helpers for building hand-crafted board situations in tests."""


def make_state(my_body, enemies=(), food=(), width=11, height=11):
    """Build a minimal game-state dict.

    Bodies are lists of (x, y), head first. Repeat the last cell to model a
    snake that just ate (its tail is stacked).
    """

    def snake(snake_id, body):
        return {
            "id": snake_id,
            "head": {"x": body[0][0], "y": body[0][1]},
            "body": [{"x": x, "y": y} for x, y in body],
            "health": 100,
            "length": len(body),
        }

    me = snake("me", my_body)
    others = [snake(f"enemy{i}", body) for i, body in enumerate(enemies)]
    return {
        "game": {"id": "test-game"},
        "turn": 1,
        "board": {
            "width": width,
            "height": height,
            "food": [{"x": x, "y": y} for x, y in food],
            "hazards": [],
            "snakes": [me, *others],
        },
        "you": me,
    }
