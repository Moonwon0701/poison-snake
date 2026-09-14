"""The pure-Python Battlesnake rules, one situation at a time.

Coordinates are (x, y) with (0, 0) at the bottom-left; bodies are head first.
Food spawning is switched off (minimum_food=0) unless a test is about it.
"""

import random

import pytest

from gym_env import rules


def state(snakes, food=(), health=None, size=11):
    """snakes: {id: body}. health: {id: value}, 100 if not given."""
    health = health or {}
    return rules.GameState(
        size,
        size,
        [rules.Snake(sid, list(body), health.get(sid, 100)) for sid, body in snakes.items()],
        list(food),
    )


def play(s, moves):
    return rules.step(s, moves, minimum_food=0, food_spawn_chance=0)


# --- moving, health, food -----------------------------------------------------


def test_move_adds_a_head_and_drops_the_tail():
    nxt, eliminated = play(state({"a": [(5, 5), (5, 4), (5, 3)]}), {"a": "up"})
    assert nxt.snake("a").body == [(5, 6), (5, 5), (5, 4)]
    assert eliminated == {}
    assert nxt.turn == 1


def test_step_leaves_the_input_state_alone():
    s = state({"a": [(5, 5), (5, 4), (5, 3)]})
    play(s, {"a": "up"})
    assert s.snake("a").body == [(5, 5), (5, 4), (5, 3)]
    assert s.turn == 0


def test_health_drops_by_one_each_turn():
    nxt, _ = play(state({"a": [(5, 5), (5, 4), (5, 3)]}, health={"a": 40}), {"a": "up"})
    assert nxt.snake("a").health == 39


def test_eating_restores_health_and_grows_from_the_next_turn():
    s = state({"a": [(5, 5), (5, 4), (5, 3)]}, food=[(5, 6)], health={"a": 40})
    nxt, _ = play(s, {"a": "up"})
    a = nxt.snake("a")
    assert a.health == 100
    assert a.body == [(5, 6), (5, 5), (5, 4), (5, 4)]  # tail cell doubled
    assert nxt.food == []
    after, _ = play(nxt, {"a": "up"})
    assert after.snake("a").body == [(5, 7), (5, 6), (5, 5), (5, 4)]


def test_missing_move_keeps_going_straight():
    nxt, _ = play(state({"a": [(5, 5), (5, 4), (5, 3)]}), {})
    assert nxt.snake("a").head == (5, 6)
    nxt, _ = play(state({"a": [(5, 5), (6, 5), (7, 5)]}), {})
    assert nxt.snake("a").head == (4, 5)


def test_missing_move_from_a_coiled_start_goes_up():
    nxt, _ = play(state({"a": [(1, 1), (1, 1), (1, 1)]}), {})
    assert nxt.snake("a").head == (1, 2)


# --- eliminations -----------------------------------------------------------


def test_snake_starves_at_zero_health():
    nxt, eliminated = play(state({"a": [(5, 5), (5, 4), (5, 3)]}, health={"a": 1}), {"a": "up"})
    assert eliminated == {"a": rules.OUT_OF_HEALTH}
    assert nxt.snakes == []


def test_eating_on_the_last_health_point_saves_the_snake():
    # Feeding comes before eliminations, so health 1 -> 0 -> 100 survives.
    s = state({"a": [(5, 5), (5, 4), (5, 3)]}, food=[(5, 6)], health={"a": 1})
    nxt, eliminated = play(s, {"a": "up"})
    assert eliminated == {}
    assert nxt.snake("a").health == 100


def test_moving_off_the_board():
    _, eliminated = play(state({"a": [(0, 5), (1, 5), (2, 5)]}), {"a": "left"})
    assert eliminated == {"a": rules.WALL_COLLISION}


def test_running_into_own_body():
    # Coiled so that moving left lands on (4, 5), which is still body.
    _, eliminated = play(state({"a": [(5, 5), (5, 6), (4, 6), (4, 5), (4, 4)]}), {"a": "left"})
    assert eliminated == {"a": rules.SELF_COLLISION}


def test_following_own_tail_is_safe():
    # Same move, but (4, 5) is the tail, which moves away this turn.
    _, eliminated = play(state({"a": [(5, 5), (5, 6), (4, 6), (4, 5)]}), {"a": "left"})
    assert eliminated == {}


def test_a_doubled_tail_is_not_safe():
    _, eliminated = play(state({"a": [(5, 5), (5, 6), (4, 6), (4, 5), (4, 5)]}), {"a": "left"})
    assert eliminated == {"a": rules.SELF_COLLISION}


def test_running_into_another_snake():
    s = state({"a": [(5, 5), (5, 4), (5, 3)], "b": [(6, 6), (6, 5), (6, 4)]})
    nxt, eliminated = play(s, {"a": "right", "b": "up"})
    assert eliminated == {"a": rules.BODY_COLLISION}
    assert [snake.id for snake in nxt.snakes] == ["b"]


def test_head_to_head_the_longer_snake_wins():
    s = state({"a": [(4, 5), (3, 5), (2, 5), (1, 5)], "b": [(6, 5), (7, 5), (8, 5)]})
    nxt, eliminated = play(s, {"a": "right", "b": "left"})
    assert eliminated == {"b": rules.HEAD_COLLISION}
    assert nxt.snake("a").head == (5, 5)


def test_head_to_head_equal_lengths_both_die():
    s = state({"a": [(4, 5), (3, 5), (2, 5)], "b": [(6, 5), (7, 5), (8, 5)]})
    nxt, eliminated = play(s, {"a": "right", "b": "left"})
    assert eliminated == {"a": rules.HEAD_COLLISION, "b": rules.HEAD_COLLISION}
    assert nxt.snakes == []


def test_head_to_head_on_food_both_eat_and_the_longer_still_wins():
    s = state(
        {"a": [(4, 5), (3, 5), (2, 5), (1, 5)], "b": [(6, 5), (7, 5), (8, 5)]},
        food=[(5, 5)],
    )
    nxt, eliminated = play(s, {"a": "right", "b": "left"})
    assert eliminated == {"b": rules.HEAD_COLLISION}
    assert len(nxt.snake("a").body) == 5
    assert nxt.food == []


def test_a_snake_that_hit_the_wall_does_not_block_others():
    # "a" leaves the board, so its body no longer counts when "b" moves
    # onto the cell a's neck now covers.
    s = state({"a": [(0, 5), (1, 5), (2, 5)], "b": [(1, 6), (1, 7), (1, 8)]})
    nxt, eliminated = play(s, {"a": "left", "b": "down"})
    assert eliminated == {"a": rules.WALL_COLLISION}
    assert nxt.snake("b").head == (1, 5)


# --- food spawning ----------------------------------------------------------


def test_food_is_topped_up_to_the_minimum_on_a_free_cell():
    s = state({"a": [(5, 5), (5, 4), (5, 3)]})
    nxt, _ = rules.step(s, {"a": "up"}, random.Random(0), food_spawn_chance=0, minimum_food=1)
    assert len(nxt.food) == 1
    assert nxt.food[0] not in nxt.snake("a").body


def test_extra_food_spawns_about_as_often_as_the_engine():
    # With minimum_food already met, a 15 setting spawns food 14% of turns.
    s = state({"a": [(5, 5), (5, 4), (5, 3)]}, food=[(0, 0)])
    rng = random.Random(0)
    spawned = sum(
        len(rules.step(s, {"a": "up"}, rng, food_spawn_chance=15, minimum_food=1)[0].food) == 2
        for _ in range(2000)
    )
    assert 200 < spawned < 360


# --- a new game -------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_new_game_uses_the_standard_start_layout(seed):
    g = rules.new_game(["a", "b", "c", "d"], random.Random(seed))
    corners = {(1, 1), (1, 9), (9, 1), (9, 9)}
    edges = {(1, 5), (5, 1), (5, 9), (9, 5)}
    heads = {snake.head for snake in g.snakes}
    assert len(heads) == 4
    assert heads == corners or heads == edges
    for snake in g.snakes:
        assert snake.body == [snake.head] * 3
        assert snake.health == 100
        x, y = snake.head
        assert any(abs(fx - x) == 1 and abs(fy - y) == 1 for fx, fy in g.food)
    assert (5, 5) in g.food
    assert len(g.food) == 5


def test_new_game_rejects_boards_it_cannot_lay_out():
    with pytest.raises(ValueError):
        rules.new_game(["a"], random.Random(0), width=11, height=9)
    with pytest.raises(ValueError):
        rules.new_game([str(i) for i in range(9)], random.Random(0))


# --- helpers ----------------------------------------------------------------


def test_non_colliding_moves_in_a_corner():
    assert rules.non_colliding_moves(state({"a": [(0, 0), (0, 1), (0, 2)]}), "a") == ["right"]


def test_non_colliding_moves_can_follow_the_tail():
    s = state({"a": [(0, 10), (1, 10), (1, 9), (0, 9)]})
    assert rules.non_colliding_moves(s, "a") == ["down"]


def test_api_json_round_trip():
    s = state({"a": [(5, 5), (5, 4), (5, 3)], "b": [(1, 1), (1, 2), (1, 2)]}, food=[(3, 3)])
    as_json = rules.to_api_json(s, "b")
    assert as_json["you"]["id"] == "b"
    assert as_json["you"]["length"] == 3
    assert rules.from_api_json(as_json) == s
