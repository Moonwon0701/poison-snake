"""tools/explain_game.py: replay data from simulations and CLI logs, and the HTML."""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import explain_game  # noqa: E402

from gym_env import rules  # noqa: E402

MOVES = {"up", "down", "left", "right"}


def embedded_json(html: str) -> dict:
    opening = '<script id="replay-data" type="application/json">'
    start = html.index(opening) + len(opening)
    return json.loads(html[start : html.index("</script>", start)])


def test_simulated_replay_explains_every_snake_every_turn():
    replay = explain_game.replay_from_simulation(seed=3, opponents=1, max_turns=60)
    assert replay["exact"] is True
    assert replay["tree"]["name"] == "choose a move"
    assert [s["id"] for s in replay["snakes"]] == ["snake1", "snake2"]
    for frame in replay["turns"]:
        assert set(frame["decisions"]) == {s["id"] for s in frame["snakes"]}
        for decision in frame["decisions"].values():
            assert decision["actual"] == decision["move"]
            assert set(decision["moves"]) == MOVES
            chosen = [m for m, info in decision["moves"].items() if info["verdict"] == "chosen"]
            assert chosen == [decision["move"]]


def test_simulated_turns_follow_each_other():
    replay = explain_game.replay_from_simulation(seed=5, opponents=2, max_turns=40)
    turns = replay["turns"]
    assert [frame["turn"] for frame in turns] == list(range(len(turns)))
    for frame, following in zip(turns, turns[1:]):
        gone = {s["id"] for s in frame["snakes"]} - {s["id"] for s in following["snakes"]}
        assert gone == set(frame["eliminated"])


def test_log_replay_recovers_the_recorded_moves(tmp_path):
    # Write a short game in the CLI's log format, then explain it.
    rng = random.Random(1)
    state = rules.new_game(["a", "b"], rng)
    lines = [json.dumps({"id": "game", "ruleset": {"settings": {}}})]
    played = []
    for _ in range(6):
        lines.append(json.dumps(rules.to_api_json(state, "a")))
        moves = {s.id: rules.non_colliding_moves(state, s.id)[0] for s in state.snakes}
        played.append(moves)
        state, eliminated = rules.step(state, moves, rng)
        assert eliminated == {}
    lines.append(json.dumps(rules.to_api_json(state, "a")))
    lines.append(json.dumps({"winnerId": "", "winnerName": "", "isDraw": False}))
    log = tmp_path / "game.jsonl"
    log.write_text("\n".join(lines))

    replay = explain_game.replay_from_log(log)
    assert replay["exact"] is False
    assert [s["name"] for s in replay["snakes"]] == ["a", "b"]
    assert len(replay["turns"]) == 7
    for frame, moves in zip(replay["turns"], played):
        assert {sid: d["actual"] for sid, d in frame["decisions"].items()} == moves
    assert all(d["actual"] is None for d in replay["turns"][-1]["decisions"].values())


def test_html_embeds_the_replay_as_json():
    replay = explain_game.replay_from_simulation(seed=0, opponents=0, max_turns=5)
    assert embedded_json(explain_game.render_html(replay)) == replay


def test_a_snake_name_cannot_break_out_of_the_script_tag():
    replay = explain_game.replay_from_simulation(seed=0, opponents=0, max_turns=3)
    replay["snakes"][0]["name"] = "</script><b>boom</b>"
    html = explain_game.render_html(replay)
    assert embedded_json(html) == replay
    assert "<b>boom</b>" not in html
