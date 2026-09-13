"""World parsing and the "what's blocked next turn" rules."""

from agent.world import World, step
from boards import make_state


def test_parses_board_and_food():
    world = World.from_json(make_state([(5, 5), (5, 4), (5, 3)], food=[(3, 4)]))
    assert (world.width, world.height, world.turn) == (11, 11, 1)
    assert world.food == {(3, 4)}


def test_separates_me_from_enemies():
    world = World.from_json(
        make_state([(5, 5), (5, 4), (5, 3)], enemies=[[(1, 1), (1, 2)]])
    )
    assert world.me.id == "me"
    assert world.me.head == (5, 5)
    assert world.me.length == 3
    assert [enemy.id for enemy in world.enemies] == ["enemy0"]


def test_step_up_increases_y():
    # (0, 0) is the bottom-left corner, so "up" means y + 1.
    assert step((5, 5), "up") == (5, 6)
    assert step((5, 5), "down") == (5, 4)


def test_in_bounds():
    world = World.from_json(make_state([(5, 5), (5, 4), (5, 3)]))
    assert world.in_bounds((0, 0))
    assert world.in_bounds((10, 10))
    assert not world.in_bounds((11, 0))
    assert not world.in_bounds((0, -1))


def test_tail_is_free_next_turn():
    world = World.from_json(make_state([(5, 5), (5, 4), (5, 3)]))
    blocked = world.blocked_next_turn()
    assert (5, 5) in blocked
    assert (5, 4) in blocked
    assert (5, 3) not in blocked


def test_stacked_tail_stays_blocked():
    # The snake just ate, so its last two segments share (5, 4).
    world = World.from_json(make_state([(5, 5), (5, 4), (5, 4)]))
    assert (5, 4) in world.blocked_next_turn()


def test_enemy_tail_is_free_too():
    world = World.from_json(
        make_state([(5, 5), (5, 4), (5, 3)], enemies=[[(1, 1), (1, 2), (1, 3)]])
    )
    blocked = world.blocked_next_turn()
    assert (1, 2) in blocked
    assert (1, 3) not in blocked
