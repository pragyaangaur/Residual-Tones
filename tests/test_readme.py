"""The numbers in the README come from the result files. These tests keep them in step."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()


def load(p):
    return json.load(open(ROOT / p))


def judged(run):
    s = load(f"{run}/state.json")
    return s, [h for v in s.values() for h in v["rounds"] if "probs" in h]


def test_no_piece_passed_in_either_loop():
    for run in ("loop", "loop_sf"):
        s, rows = judged(run)
        assert len(rows) == 80
        assert not any(h["verified"] for h in rows)
        assert not any("confirm" in v for v in s.values())
    assert "None of its 80 pieces passed" in README and "None of the 80 pieces passed again" in README


def test_first_loop_description_matches():
    _, rows = judged("loop")
    assert sum(h["desc_rank"] == 1 for h in rows) == 7
    assert "7 of 80 pieces" in README


def test_second_loop_iteration_did_not_help():
    _, rows = judged("loop_sf")
    early = np.mean([h["q_target"] for h in rows if h["round"] <= 4])
    late = np.mean([h["q_target"] for h in rows if h["round"] >= 5])
    assert (round(early, 3), round(late, 3)) == (0.119, 0.109)
    assert "0.119 in rounds 1 to 4 and 0.109 in rounds 5 to 8" in README


def test_arousal_is_heard_as_excitement():
    s, _ = judged("loop_sf")
    rs = s["arousal"]["rounds"]
    assert sum(h["pick"] == "excitement" for h in rs) == 7
    assert sum(h["desc_rank"] == 1 for h in rs) == 6
    assert round(np.mean([h["p_target"] for h in rs]), 2) == 0.48
    assert round(max(h["q_target"] for h in rs), 2) == 0.40
    assert "excitement in 7 of 8 rounds, at 0.48" in README and "6 of 8 rounds" in README


def test_listener_bias():
    p = dict(zip(load("loop_sf/prior.json")["keys"], load("loop_sf/prior.json")["prior"]))
    assert round(p["excitement"], 2) == 0.28 and round(p["sadness"], 2) == 0.22
    assert round(p["neutral"], 3) == 0.035 and round(p["fear"], 3) == 0.048


def test_text_only_listener():
    l = load("out/listener.json")
    assert sum(t["pick"] == t["truth"] for t in l["tones"]) == 2
    assert sum(t["pick"] == t["truth"] for t in l["voices"]) == 3 and len(l["voices"]) == 33
    assert sum(t["pick"] == "sadness" for t in l["voices"]) == 13
    assert "2 of 10 tones" in README and "3 of 33 passages" in README


def test_tone_geometry():
    m = load("out/sound_meta.json")
    C, n = np.array(m["cos"]), m["names"]
    i = n.index("s2_pain")
    assert round(max(C[i][j] for j in range(len(n)) if j != i), 2) == 0.33
    assert round(m["sounds"]["sadness"]["norm"], 1) == 24.6 and round(m["sounds"]["s2_pain"]["norm"], 1) == 4.9
