"""The behavior tree building blocks, tested without any Battlesnake logic."""

from agent.bt import Action, Condition, Selector, Sequence, Status


def recording_action(name, succeeds, ran):
    """An Action that notes it ran, then reports `succeeds`."""

    def act(blackboard):
        ran.append(name)
        return succeeds

    return Action(name, act)


def tick(node, blackboard=None):
    trace = []
    return node.tick(blackboard, trace), trace


# --- Sequence ---------------------------------------------------------------


def test_sequence_succeeds_when_every_child_succeeds():
    ran = []
    seq = Sequence("seq", [recording_action("a", True, ran), recording_action("b", True, ran)])
    assert tick(seq) == (Status.SUCCESS, ["seq", "b"])
    assert ran == ["a", "b"]


def test_sequence_stops_at_the_first_failure():
    ran = []
    seq = Sequence(
        "seq",
        [
            recording_action("a", True, ran),
            recording_action("b", False, ran),
            recording_action("c", True, ran),
        ],
    )
    assert tick(seq) == (Status.FAILURE, ["seq", "b"])
    assert ran == ["a", "b"]  # "c" never ran


def test_empty_sequence_succeeds():
    assert tick(Sequence("seq", [])) == (Status.SUCCESS, ["seq"])


# --- Selector ---------------------------------------------------------------


def test_selector_stops_at_the_first_success():
    ran = []
    sel = Selector(
        "sel",
        [
            recording_action("a", False, ran),
            recording_action("b", True, ran),
            recording_action("c", True, ran),
        ],
    )
    assert tick(sel) == (Status.SUCCESS, ["sel", "b"])
    assert ran == ["a", "b"]  # "c" never ran


def test_selector_fails_when_every_child_fails():
    ran = []
    sel = Selector("sel", [recording_action("a", False, ran), recording_action("b", False, ran)])
    assert tick(sel) == (Status.FAILURE, ["sel", "b"])
    assert ran == ["a", "b"]


def test_empty_selector_fails():
    assert tick(Selector("sel", [])) == (Status.FAILURE, ["sel"])


# --- leaves and the blackboard ----------------------------------------------


def test_condition_turns_true_and_false_into_statuses():
    is_ready = Condition("ready?", lambda bb: bb["ready"])
    assert tick(is_ready, {"ready": True}) == (Status.SUCCESS, ["ready?"])
    assert tick(is_ready, {"ready": False}) == (Status.FAILURE, ["ready?"])


def test_nodes_share_information_through_the_blackboard():
    def write(bb):
        bb["answer"] = 42
        return True

    seq = Sequence(
        "seq",
        [Action("write", write), Condition("answer is 42", lambda bb: bb["answer"] == 42)],
    )
    blackboard = {}
    assert tick(seq, blackboard)[0] is Status.SUCCESS
    assert blackboard == {"answer": 42}


def test_trace_follows_the_deciding_branch_through_nested_nodes():
    # The first branch fails at its condition; the second one succeeds.
    tree = Selector(
        "root",
        [
            Sequence("first", [Condition("never", lambda bb: False), Action("x", lambda bb: True)]),
            Sequence("second", [Condition("always", lambda bb: True), Action("y", lambda bb: True)]),
        ],
    )
    assert tick(tree) == (Status.SUCCESS, ["root", "second", "y"])


def test_visits_record_every_node_that_ran_with_its_status():
    root = Selector(
        "root",
        [Condition("a", lambda bb: False), Action("b", lambda bb: True), Action("c", lambda bb: True)],
    )
    visits = []
    root.tick(None, [], visits)
    # Children come before their parent; "c" never ran.
    assert [(node.name, status) for node, status in visits] == [
        ("a", Status.FAILURE),
        ("b", Status.SUCCESS),
        ("root", Status.SUCCESS),
    ]


def test_describe_gives_the_shape_with_ids_in_walk_order():
    tree = Selector(
        "root",
        [Sequence("seq", [Condition("c", lambda bb: True)]), Action("act", lambda bb: True)],
    )
    assert [node.name for node in tree.walk()] == ["root", "seq", "c", "act"]
    assert tree.describe() == {
        "id": 0,
        "name": "root",
        "kind": "Selector",
        "children": [
            {
                "id": 1,
                "name": "seq",
                "kind": "Sequence",
                "children": [{"id": 2, "name": "c", "kind": "Condition", "children": []}],
            },
            {"id": 3, "name": "act", "kind": "Action", "children": []},
        ],
    }


def test_one_tree_can_be_ticked_again_with_fresh_results():
    # Nodes keep no state between ticks.
    tree = Selector("root", [Condition("flag", lambda bb: bb["flag"]), Action("fallback", lambda bb: True)])
    assert tick(tree, {"flag": True}) == (Status.SUCCESS, ["root", "flag"])
    assert tick(tree, {"flag": False}) == (Status.SUCCESS, ["root", "fallback"])
