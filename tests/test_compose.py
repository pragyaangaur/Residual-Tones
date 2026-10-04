"""The producer's output is messy, so the parser and the synthesisers must cope with it."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compose  # noqa: E402
from paths import SF2  # noqa: E402

MESSY = '''{"tempo_bpm": 60, "key": "C", "mode" : "minor", "articulation": legato, "instrument":"strings",
"melody" : [ [0, 2], [1, 2], [-3,0.5], [5, 1],
"chords" = [1, 4, 5, 2], "percussion" = "heartbeat", "reverb" = 0.7, "intent" = "sorrow"'''


def test_parser_reads_bare_words_equals_signs_and_an_unclosed_melody():
    s = compose.parse_json(MESSY)
    assert s["articulation"] == "legato"
    assert s["chords"] == [1, 4, 5, 2]
    assert s["percussion"] == "heartbeat"
    assert s["melody"] == [[0, 2], [1, 2], [-3, 0.5], [5, 1]]


def test_parser_gives_up_on_text_with_no_score():
    assert compose.parse_json("I would write something slow and sad.") is None


def test_clean_clamps_and_fills_defaults():
    s = compose.clean({"tempo_bpm": 999, "key": "H", "dissonance": 7, "melody": [[40, 99]]})
    assert s["tempo_bpm"] == 200 and s["key"] == "C" and s["dissonance"] == 1
    assert s["melody"] == [[21, 8.0]] and s["chords"] == [1]


@pytest.mark.parametrize("which", ["sine", "soundfont"])
def test_render_writes_twelve_seconds_of_stereo_sound(tmp_path, which):
    if which == "soundfont" and not SF2.exists():
        pytest.skip("run scripts/setup.sh to fetch the soundfont")
    fn = compose.render if which == "sine" else compose.render_sf
    out = tmp_path / "x.wav"
    fn(compose.parse_json(MESSY), out)
    sr, x = wavfile.read(out)
    assert x.shape == (12 * sr, 2)
    assert np.abs(x).max() > 1000
