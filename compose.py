"""Producer side of the compose and verify loop.

The producer is Qwen 2.5 7B Instruct (4-bit MLX), steered at layer 8 with the target
emotion's Pain Axis vector and told the target. It writes a score as JSON, and render()
turns the score into a real stereo WAV with a small synthesiser. After each round it gets
the blind verifier's description and revises.

  compose.py test fear arousal     write and render one score per emotion, no verifier
"""
import json, re, sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

HERE = Path(__file__).parent
from paths import ROOT, VECTORS, QWEN, SF2
SR = 22050
NOTES = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
         "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
MODES = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10], "dorian": [0, 2, 3, 5, 7, 9, 10],
         "phrygian": [0, 1, 3, 5, 7, 8, 10], "lydian": [0, 2, 4, 6, 7, 9, 11], "mixolydian": [0, 2, 4, 5, 7, 9, 10],
         "locrian": [0, 1, 3, 5, 6, 8, 10], "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
         "chromatic": list(range(12)), "whole_tone": [0, 2, 4, 6, 8, 10]}
INSTRUMENTS = ["piano", "strings", "organ", "choir", "flute", "bell", "synth_pad", "plucked", "brass", "low_drone"]
DYN = {"pp": 0.15, "p": 0.3, "mp": 0.45, "mf": 0.6, "f": 0.8, "ff": 1.0}
REG = {"very_low": 36, "low": 48, "mid": 60, "high": 72, "very_high": 84}

SCHEMA = """{
  "tempo_bpm": number from 30 to 200,
  "key": one of C C# D Eb E F F# G Ab A Bb B,
  "mode": one of major minor dorian phrygian lydian mixolydian locrian harmonic_minor chromatic whole_tone,
  "instrument": one of piano strings organ choir flute bell synth_pad plucked brass low_drone,
  "register": one of very_low low mid high very_high,
  "dynamics": one of pp p mp mf f ff,
  "crescendo": number from -1 (fade away) to 1 (swell),
  "articulation": one of legato staccato tremolo accented,
  "dissonance": number from 0 (pure) to 1 (harsh clusters and detuning),
  "melody": list of 8 to 24 notes (never more than 24), each [scale_degree, beats]; degree 1 to 15 counts up the scale from the root, 0 is a rest, negative goes below the root,
  "chords": list of 2 to 8 chords, each a scale degree from 1 to 7 for the triad root, one per bar,
  "percussion": one of none heartbeat steady_pulse march irregular_hits rumble,
  "reverb": number from 0 (dry) to 1 (cavernous),
  "intent": one sentence on what you are going for
}"""


# ---------------------------------------------------------------- synthesis

def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tone(instr, f, dur, amp, art, rough, rng):
    n = max(int(dur * SR), 1)
    t = np.arange(n) / SR
    det = 1 + rough * 0.012 * rng.uniform(-1, 1)
    if instr in ("piano", "plucked", "bell"):
        partials = {"piano": [(1, 1), (2, .5), (3, .25), (4, .12), (5, .06)],
                    "plucked": [(1, 1), (2, .6), (3, .4), (4, .3), (5, .2), (6, .1)],
                    "bell": [(1, 1), (2.76, .6), (5.4, .35), (8.93, .2), (2.0, .3)]}[instr]
        decay = {"piano": 1.6, "plucked": 4.0, "bell": 0.9}[instr]
        y = sum(a * np.sin(2 * np.pi * f * k * det * t) * np.exp(-t * decay * (1 + 0.4 * k)) for k, a in partials)
        env = np.minimum(1, t / 0.004)
    else:
        partials = {"strings": [(k, 1 / k) for k in range(1, 9)], "organ": [(1, 1), (2, .8), (3, .5), (4, .6), (8, .3)],
                    "choir": [(1, 1), (2, .5), (3, .35), (4, .1), (5, .25)], "flute": [(1, 1), (2, .15), (3, .05)],
                    "synth_pad": [(1, 1), (1.005, .9), (2, .4), (2.01, .3), (3, .2)],
                    "brass": [(k, 1 / k ** 0.7) for k in range(1, 10)], "low_drone": [(0.5, 1), (1, .8), (1.5, .3), (2, .4)]}[instr]
        vib = 1 + (0.004 * np.sin(2 * np.pi * 5.2 * t) if instr in ("strings", "choir", "flute") else 0)
        y = sum(a * np.sin(2 * np.pi * f * k * det * vib * t + rng.uniform(0, 6.28)) for k, a in partials)
        if instr == "choir" or instr == "flute":
            y += 0.04 * rng.normal(size=n) * (1 if instr == "flute" else 0.5)
        att = {"strings": 0.12, "organ": 0.02, "choir": 0.2, "flute": 0.05, "synth_pad": 0.4, "brass": 0.06, "low_drone": 0.5}[instr]
        rel = min(0.15, dur / 3)
        env = np.minimum(1, t / att) * np.minimum(1, (dur - t) / rel)
    if rough > 0:
        y = y + rough * 0.5 * np.sin(2 * np.pi * f * 2 ** (rng.choice([1, 6, 11]) / 12) * t) * (y != 0)
    if art == "staccato":
        env = env * (t < min(dur * 0.35, 0.18))
    elif art == "tremolo":
        env = env * (0.55 + 0.45 * np.sin(2 * np.pi * 11 * t))
    elif art == "accented":
        env = env * (1 + 1.2 * np.exp(-t * 12))
    return amp * y * np.clip(env, 0, None)


def drum(kind, n_total, beat, rng):
    out = np.zeros(n_total)
    def hit(pos, f0, dec, amp, noise=0.0):
        n = int(0.5 * SR); t = np.arange(n) / SR
        y = np.sin(2 * np.pi * f0 * t * (1 + np.exp(-t * 30))) * np.exp(-t * dec) + noise * rng.normal(size=n) * np.exp(-t * dec * 2)
        p = int(pos * SR); e = min(n_total, p + n)
        if p < n_total:
            out[p:e] += amp * y[: e - p]
    T = n_total / SR
    if kind == "heartbeat":
        for s in np.arange(0, T, beat * 1.0):
            hit(s, 50, 18, 0.9); hit(s + 0.22, 45, 22, 0.6)
    elif kind == "steady_pulse":
        for s in np.arange(0, T, beat):
            hit(s, 70, 14, 0.6)
    elif kind == "march":
        for i, s in enumerate(np.arange(0, T, beat)):
            hit(s, 60 if i % 2 == 0 else 180, 16, 0.7, noise=0.0 if i % 2 == 0 else 0.6)
    elif kind == "irregular_hits":
        s = 0.0
        while s < T:
            hit(s, rng.uniform(40, 140), 10, rng.uniform(0.5, 1.0), noise=0.3)
            s += beat * rng.choice([0.5, 0.75, 1.5, 2.5, 0.25])
    elif kind == "rumble":
        r = np.convolve(rng.normal(size=n_total), np.ones(400) / 400, "same")
        out += 4 * r * (0.6 + 0.4 * np.sin(2 * np.pi * 0.3 * np.arange(n_total) / SR))
    return out


def clean(score):
    s = dict(score)
    s["tempo_bpm"] = float(np.clip(float(s.get("tempo_bpm", 90)), 30, 200))
    s["key"] = s.get("key") if s.get("key") in NOTES else "C"
    s["mode"] = s.get("mode") if s.get("mode") in MODES else "major"
    s["instrument"] = s.get("instrument") if s.get("instrument") in INSTRUMENTS else "piano"
    s["register"] = s.get("register") if s.get("register") in REG else "mid"
    s["dynamics"] = s.get("dynamics") if s.get("dynamics") in DYN else "mf"
    s["articulation"] = s.get("articulation") if s.get("articulation") in ("legato", "staccato", "tremolo", "accented") else "legato"
    s["percussion"] = s.get("percussion") if s.get("percussion") in ("none", "heartbeat", "steady_pulse", "march", "irregular_hits", "rumble") else "none"
    for k, d in (("crescendo", 0.0), ("dissonance", 0.0), ("reverb", 0.3)):
        try:
            s[k] = float(np.clip(float(s.get(k, d)), -1 if k == "crescendo" else 0, 1))
        except (TypeError, ValueError):
            s[k] = d
    mel = []
    for x in s.get("melody", []) or []:
        try:
            deg, b = int(round(float(x[0]))), float(x[1])
            mel.append([int(np.clip(deg, -14, 21)), float(np.clip(b, 0.125, 8))])
        except (TypeError, ValueError, IndexError):
            pass
    s["melody"] = mel[:32] or [[1, 1], [3, 1], [5, 1], [3, 1]]
    ch = []
    for c in s.get("chords", []) or []:
        try:
            ch.append(int(np.clip(int(round(float(c if not isinstance(c, list) else c[0]))), 1, 7)))
        except (TypeError, ValueError):
            pass
    s["chords"] = ch[:8] or [1]
    return s


def render(score, path, seconds=12.0, seed=0):
    s = clean(score)
    rng = np.random.default_rng(seed)
    beat = 60.0 / s["tempo_bpm"]
    scale = MODES[s["mode"]]
    root = REG[s["register"]] + NOTES[s["key"]]
    amp = DYN[s["dynamics"]]
    rough = s["dissonance"]
    N = int(seconds * SR)
    mel = np.zeros(N); harm = np.zeros(N)

    def deg_midi(d):
        if d == 0:
            return None
        k = d - 1 if d > 0 else d
        o, i = divmod(k, len(scale))
        return root + 12 * o + scale[i]

    pos, i = 0.0, 0
    while pos < seconds:
        d, b = s["melody"][i % len(s["melody"])]
        dur = b * beat
        m = deg_midi(d)
        if m is not None:
            y = tone(s["instrument"], midi_hz(m), min(dur * (1.05 if s["articulation"] == "legato" else 0.95), seconds - pos), 1.0, s["articulation"], rough, rng)
            p = int(pos * SR); e = min(N, p + len(y)); mel[p:e] += y[: e - p]
        pos += dur; i += 1

    bar = 4 * beat
    pad = "synth_pad" if s["instrument"] in ("piano", "plucked", "bell") else s["instrument"]
    pos, i = 0.0, 0
    while pos < seconds:
        c = s["chords"][i % len(s["chords"])]
        for step in (0, 2, 4):
            m = deg_midi(c + step)
            m = m - 12
            if rough > 0.6 and step == 2:
                m += rng.choice([-1, 1])
            y = tone(pad, midi_hz(m), min(bar * 1.02, seconds - pos), 0.33, "legato", rough * 0.6, rng)
            p = int(pos * SR); e = min(N, p + len(y)); harm[p:e] += y[: e - p]
        if rough > 0.75:
            y = tone(pad, midi_hz(deg_midi(c) - 11), min(bar, seconds - pos), 0.25, "legato", rough, rng)
            p = int(pos * SR); e = min(N, p + len(y)); harm[p:e] += y[: e - p]
        pos += bar; i += 1

    perc = drum(s["percussion"], N, beat, rng) if s["percussion"] != "none" else np.zeros(N)
    t = np.arange(N) / SR
    shape = 1 + s["crescendo"] * (t / seconds - 0.5) * 1.6
    norm = lambda x: x / (np.abs(x).max() + 1e-9)
    dry = 0.7 * norm(mel) + 0.45 * norm(harm) + (0.5 * norm(perc) if perc.any() else 0)
    if s["reverb"] > 0:
        L = int(SR * (0.4 + 2.6 * s["reverb"]))
        ir = rng.normal(size=(2, L)) * np.exp(-np.arange(L) / SR * (6.9 / (0.4 + 2.6 * s["reverb"])))
        wet = np.stack([fftconvolve(dry, ir[c])[:N] for c in range(2)], 1)
        wet /= np.abs(wet).max() + 1e-9
        st = (1 - 0.6 * s["reverb"]) * np.stack([dry, dry], 1) / (np.abs(dry).max() + 1e-9) + 0.6 * s["reverb"] * wet
    else:
        st = np.stack([dry, dry], 1)
    st *= np.clip(shape, 0.1, 2)[:, None]
    st *= np.minimum(1, t / 0.05)[:, None] * np.minimum(1, (seconds - t) / 1.0)[:, None]
    st = st / (np.abs(st).max() + 1e-9) * (0.45 + 0.5 * amp)
    wavfile.write(path, SR, (st * 32767).astype(np.int16))
    return s


# ---------------------------------------------------------------- producer

def parse_json(text):
    """Pull each field out on its own, so bare words, '=' for ':' and a melody that never
    closes still give a usable score. Returns None if nothing usable is found."""
    out = {}
    for k in ("tempo_bpm", "crescendo", "dissonance", "reverb"):
        m = re.search(rf'"?{k}"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', text)
        if m:
            out[k] = float(m.group(1))
    for k in ("key", "mode", "instrument", "register", "dynamics", "articulation", "percussion"):
        m = re.search(rf'"?{k}"?\s*[:=]\s*"?([A-Za-z_#]+)"?', text)
        if m:
            out[k] = m.group(1)
    m = re.search(r'"?melody"?\s*[:=]\s*\[(.*?)(?:"?chords"?\s*[:=]|$)', text, re.S)
    if m:
        out["melody"] = [[float(a), float(b)] for a, b in
                         re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", m.group(1))][:32]
    m = re.search(r'"?chords"?\s*[:=]\s*\[([^\]]*)\]', text)
    if m:
        out["chords"] = [int(x) for x in re.findall(r"-?\d+", m.group(1))][:8]
    m = re.search(r'"?intent"?\s*[:=]\s*"([^"]*)', text)
    if m:
        out["intent"] = m.group(1)
    return out if len(out) >= 6 and out.get("melody") else None


def producer_prompt(emotion, history):
    p = (f"You are a composer. Write a short piece of instrumental music that a listener will clearly hear as {emotion}. "
         f"Use everything music offers: tempo, mode, register, instrument, dynamics, rhythm, dissonance and space.\n\n"
         f"Reply with one JSON object only, with every string in double quotes, in this format:\n{SCHEMA}")
    if history:
        best = max(history, key=lambda h: h.get("q_target", h.get("p_target", 0)))
        shown = [best] + [h for h in history[-2:] if h is not best]
        p += "\n\nYour earlier attempts and what a listener said. The listener heard the audio but was not told the target.\n"
        for h in shown:
            tag = " (your best so far)" if h is best else ""
            p += (f"\nAttempt {h['round']}{tag}: {json.dumps(h['score'], separators=(',', ':'))}\n"
                  f"Listener's description: \"{h['description']}\"\n"
                  f"Listener's best guess from a list of emotions: {h.get('pick_corrected', h['pick'])} (target was {emotion}).\n")
        p += f"\nWrite a new score that moves the listener toward hearing {emotion}. Keep what worked in your best attempt and change what is not working."
    return p


class Producer:
    def __init__(self, steer=True, k=0.6):
        import torch
        from steering import Steerer
        d = torch.load(VECTORS,
                       map_location="cpu", weights_only=False)
        self.vec = {k_.removesuffix("_vector"): v.float().numpy() for k_, v in d.items() if k_.endswith("_vector")}
        self.st = Steerer(QWEN, 8, 8, self.vec["s2_pain"])
        self.resid = float(np.mean([np.linalg.norm(self.st.hidden(t)) for t in
                                    ["The weather today is", "Here is a list of fruits:", "The meeting is scheduled for"]]))
        self.steer, self.k = steer, k

    def compose(self, jobs, seed=0):
        """jobs: list of (vector_name, emotion_words, history). Returns list of (score, raw_text)."""
        rows, prompts = [], []
        for vname, words, hist in jobs:
            u = self.vec[vname] / np.linalg.norm(self.vec[vname]) if (self.steer and vname in self.vec) else None
            rows.append(np.zeros(self.st.d, np.float32) if u is None else (self.k * self.resid * u).astype(np.float32))
            prompts.append(self.st.encode(f"<|im_start|>user\n{producer_prompt(words, hist)}<|im_end|>\n<|im_start|>assistant\n"))
        b = self.st.batch(len(jobs), seed=seed, temp=0.8, top_p=0.9)
        b.feed(prompts)
        b.set_directions(np.stack(rows))
        b.remaining[:] = 450
        out, _, _ = b.generate(450)
        res = []
        for o in out:
            txt = self.st.decode(o)
            res.append((parse_json(txt), txt))
        return res


if __name__ == "__main__" and sys.argv[1] == "test":
    out = HERE / "loop" / "test"; out.mkdir(parents=True, exist_ok=True)
    pr = Producer()
    names = sys.argv[2:]
    words = {"fear": "fear", "arousal": "intense excitement or joy", "sadness": "sadness", "s2_pain": "pain"}
    for (score, txt), n in zip(pr.compose([(n, words.get(n, n), []) for n in names]), names):
        print(f"=== {n}\n{txt[:900]}\n")
        if score:
            render(score, out / f"{n}.wav")
            print("rendered", clean(score)["intent"] if "intent" in score else "")


# ---------------------------------------------------------------- soundfont synthesis

GM = {"piano": 0, "strings": 48, "organ": 19, "choir": 52, "flute": 73, "bell": 14, "synth_pad": 89,
      "plucked": 46, "brass": 61, "low_drone": 42}
GM_HARM = {"piano": 0, "plucked": 46, "bell": 14, "flute": 49, "strings": 49, "organ": 19, "choir": 52,
           "synth_pad": 89, "brass": 60, "low_drone": 49}
VEL = {"pp": 30, "p": 45, "mp": 62, "mf": 78, "f": 98, "ff": 118}


def render_sf(score, path, seconds=12.0, seed=0):
    """Render a score with the GeneralUser GS General MIDI soundfont."""
    import tinysoundfont
    s = clean(score)
    rng = np.random.default_rng(seed)
    beat = 60.0 / s["tempo_bpm"]
    scale = MODES[s["mode"]]
    root = REG[s["register"]] + NOTES[s["key"]]
    rough = s["dissonance"]
    v0 = VEL[s["dynamics"]]

    def deg_midi(d):
        k = d - 1 if d > 0 else d
        o, i = divmod(k, len(scale))
        return int(np.clip(root + 12 * o + scale[i], 21, 108))

    def vel(t, extra=0):
        ramp = 1 + s["crescendo"] * (t / seconds - 0.5) * 1.2
        return int(np.clip(v0 * ramp + extra + rng.integers(-4, 5), 8, 127))

    ev = []  # (time, order, kind, args)
    def note(ch, t, dur, key, v):
        if t >= seconds:
            return
        ev.append((t, 1, "on", (ch, key, v)))
        ev.append((min(t + dur, seconds), 0, "off", (ch, key)))

    tremolo_strings = s["articulation"] == "tremolo" and s["instrument"] in ("strings", "low_drone")
    pos, i = 0.0, 0
    while pos < seconds:
        d, b = s["melody"][i % len(s["melody"])]
        dur = b * beat
        if d != 0:
            m = deg_midi(d)
            accent = 25 if (s["articulation"] == "accented" and abs((pos / beat) % 4) < 1e-6) else 0
            if s["articulation"] == "staccato":
                note(0, pos, min(dur * 0.35, 0.2), m, vel(pos, 8))
            elif s["articulation"] == "tremolo" and not tremolo_strings:
                tt = pos
                while tt < pos + dur:
                    note(0, tt, 1 / 14, m, vel(tt)); tt += 1 / 12
            else:
                note(0, pos, dur * (1.02 if s["articulation"] == "legato" else 0.8), m, vel(pos, accent))
            if rough > 0.3:
                note(2, pos, dur, m, max(8, vel(pos) - 20))
            if rough > 0.8:
                note(2, pos, dur, m + 6, max(8, vel(pos) - 30))
        pos += dur; i += 1

    bar = 4 * beat
    pos, i = 0.0, 0
    while pos < seconds:
        c = s["chords"][i % len(s["chords"])]
        keys = [deg_midi(c + st) - 12 for st in (0, 2, 4)]
        if rough > 0.6:
            keys.append(keys[0] + 1)
        for k in keys:
            note(1, pos, bar * 1.0, k, max(8, vel(pos) - 28))
        pos += bar; i += 1

    if s["percussion"] != "none":
        T = seconds
        if s["percussion"] == "heartbeat":
            for t in np.arange(0, T, max(beat, 0.6)):
                note(9, t, 0.1, 36, 95); note(9, t + 0.22, 0.1, 36, 70)
        elif s["percussion"] == "steady_pulse":
            for t in np.arange(0, T, beat):
                note(9, t, 0.1, 36, 85)
        elif s["percussion"] == "march":
            for j, t in enumerate(np.arange(0, T, beat)):
                note(9, t, 0.1, 36 if j % 2 == 0 else 38, 95)
                note(9, t + beat / 2, 0.05, 42, 50)
        elif s["percussion"] == "irregular_hits":
            t = 0.0
            while t < T:
                note(9, t, 0.2, int(rng.choice([41, 43, 45, 47, 49])), int(rng.integers(70, 120)))
                t += beat * float(rng.choice([0.5, 0.75, 1.5, 2.5, 0.25]))
        elif s["percussion"] == "rumble":
            for t in np.arange(0, T, 1 / 16):
                note(3, t, 1 / 16, root % 12 + 36, int(40 + 25 * np.sin(2 * np.pi * 0.3 * t) ** 2))

    sy = tinysoundfont.Synth(samplerate=SR)
    sf = sy.sfload(str(SF2))
    mel_prog = 44 if tremolo_strings else GM[s["instrument"]]
    sy.program_select(0, sf, 0, mel_prog)
    sy.program_select(1, sf, 0, GM_HARM[s["instrument"]])
    sy.program_select(2, sf, 0, mel_prog)
    sy.program_select(3, sf, 0, 47)
    sy.program_select(9, sf, 128, 0, is_drums=True)
    sy.pitchbend(2, int(8192 + 8192 * (rough * 35) / 200))     # detune the doubling layer
    ev.sort(key=lambda e: (e[0], e[1]))
    N = int(seconds * SR)
    out = np.zeros((N + SR, 2), np.float32)
    cur = 0
    for t, _, kind, a in ev + [(seconds + 1.0, 0, "end", ())]:
        n = int(t * SR)
        if n > cur:
            buf = np.frombuffer(sy.generate(n - cur), dtype=np.float32).reshape(-1, 2)
            out[cur:cur + len(buf)] = buf[: len(out) - cur]
            cur = n
        if kind == "on":
            sy.noteon(*a)
        elif kind == "off":
            sy.noteoff(*a)
    st = out[:N].astype(np.float64)
    if s["reverb"] > 0:
        L = int(SR * (0.4 + 2.6 * s["reverb"]))
        ir = rng.normal(size=(2, L)) * np.exp(-np.arange(L) / SR * (6.9 / (0.4 + 2.6 * s["reverb"])))
        wet = np.stack([fftconvolve(st[:, c], ir[c])[:N] for c in range(2)], 1)
        wet *= np.abs(st).max() / (np.abs(wet).max() + 1e-9)
        st = (1 - 0.5 * s["reverb"]) * st + 0.5 * s["reverb"] * wet
    t = np.arange(N) / SR
    st *= np.minimum(1, (seconds - t) / 1.0)[:, None]
    st = st / (np.abs(st).max() + 1e-9) * (0.45 + 0.5 * DYN[s["dynamics"]])
    wavfile.write(path, SR, (st * 32767).astype(np.int16))
    return s
