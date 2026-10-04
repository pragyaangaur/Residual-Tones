"""Compose and verify loop. Each round runs two phases in separate processes, so only one
model is in memory at a time:

  loop.py produce R    the steered producer writes and renders a score for every open target
  loop.py verify R     the blind verifier hears each new WAV and judges it
  loop.py confirm      verified pieces are heard again with different wording and option orders
  loop.py run          rounds 1 to MAX_ROUNDS, then confirm

State lives in loop/state.json. A target closes when it is verified, or after MAX_ROUNDS."""
import json, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
import os
RUN = os.environ.get("LOOP_RUN", "loop_sf")          # run 1 (sine synth) is kept in loop/
LOOP = HERE / RUN
STATE = LOOP / "state.json"
PRIOR = LOOP / "prior.json"
RENDER = os.environ.get("LOOP_RENDER", "sf")
MAX_ROUNDS = 8
TARGETS = {"s2_pain": "pain", "s1_pain": "pain", "fear": "fear", "negemotion": "negative emotion",
           "negworld": "bad world", "sadness": "sadness", "numb": "numbness", "bodysens": "body",
           "arousal": "excitement", "random": "neutral"}
WORDS = {"pain": "pain", "fear": "fear", "negative emotion": "anger, shame or disgust",
         "bad world": "distress about bad things happening in the world", "sadness": "sadness",
         "numbness": "numbness, feeling nothing", "body": "a calm physical sensation in the body",
         "excitement": "intense excitement or joy", "neutral": "a neutral state with no particular emotion"}


def load():
    return json.load(open(STATE)) if STATE.exists() else {n: {"label": TARGETS[n], "rounds": [], "done": False} for n in TARGETS}


def save(s):
    json.dump(s, open(STATE, "w"), indent=1)


def open_targets(s):
    return [n for n in s if not s[n]["done"]]


def produce(r):
    sys.path.insert(0, str(HERE))
    from compose import Producer, render, render_sf, clean
    if RENDER == "sf":
        render = render_sf
    s = load()
    todo = [n for n in open_targets(s) if len(s[n]["rounds"]) < r]
    if not todo:
        return
    pr = Producer(steer=True)
    pending, tries = list(todo), 0
    while pending and tries < 4:
        jobs = [(n, WORDS[s[n]["label"]], [h for h in s[n]["rounds"] if "description" in h]) for n in pending]
        res = pr.compose(jobs, seed=1000 * r + tries)
        still = []
        for (score, txt), n in zip(res, pending):
            if score is None:
                still.append(n); continue
            wav = LOOP / "audio" / f"{n}_r{r}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            sc = render(score, wav, seed=r)
            s[n]["rounds"].append({"round": r, "score": sc, "wav": str(wav.relative_to(HERE)), "parse_tries": tries + 1})
            print(f"produced {n} r{r}: {sc.get('intent', '')[:100]}", flush=True)
        pending, tries = still, tries + 1
    for n in pending:
        print(f"no usable score for {n} in round {r}", flush=True)
    save(s)


def verify(r):
    sys.path.insert(0, str(HERE))
    from verify import Verifier
    s = load()
    v = Verifier()
    rng = np.random.default_rng(r)
    for n in open_targets(s):
        h = s[n]["rounds"][-1] if s[n]["rounds"] else None
        if not h or h["round"] != r or "description" in h:
            continue
        h.update(v.judge(HERE / h["wav"], s[n]["label"], rng, prior=prior()))
        print(f"r{r} {n:11s} pick={h['pick']:16s} p={h['p_target']:.2f} corr={h['pick_corrected']:16s} q={h['q_target']:.2f} desc_top={h['desc_top']:16s} "
              f"rank={h['desc_rank']} {'VERIFIED' if h['verified'] else ''}\n    \"{h['description'][:160]}\"", flush=True)
        if h["verified"] or r >= MAX_ROUNDS:
            s[n]["done"] = True
    save(s)


def confirm():
    sys.path.insert(0, str(HERE))
    from verify import Verifier
    s = load()
    v = Verifier()
    rng = np.random.default_rng(99)
    for n in s:
        good = [h for h in s[n]["rounds"] if h.get("verified")]
        if not good:
            continue
        h = good[-1]
        c = v.judge(HERE / h["wav"], s[n]["label"], rng, which="alt", prior=prior())
        s[n]["confirm"] = c
        print(f"confirm {n:11s} pick={c['pick_corrected']:16s} q={c['q_target']:.2f} desc_top={c['desc_top']} "
              f"{'HELD' if c['verified'] else 'DID NOT HOLD'}\n    \"{c['description'][:160]}\"", flush=True)
    save(s)


def prior():
    return json.load(open(PRIOR))["prior"] if PRIOR.exists() else None


def calibrate():
    """Measure the verifier's forced-choice bias on this renderer: re-render the round 1 to 4
    scores from run 1 (four attempts at each of the ten targets) and average its probabilities."""
    sys.path.insert(0, str(HERE))
    from compose import render_sf, render
    from verify import Verifier, KEYS
    old = json.load(open(HERE / "loop" / "state.json"))
    v = Verifier()
    rng = np.random.default_rng(5)
    (LOOP / "calib").mkdir(parents=True, exist_ok=True)
    P = []
    for n, t in old.items():
        for h in t["rounds"][:4]:
            w = LOOP / "calib" / f"{n}_r{h['round']}.wav"
            (render_sf if RENDER == "sf" else render)(h["score"], w, seed=h["round"])
            P.append(v.choose(w, rng))
    P = np.array(P)
    json.dump({"keys": KEYS, "prior": P.mean(0).round(5).tolist(), "n": len(P)}, open(PRIOR, "w"), indent=1)
    print("prior", dict(zip(KEYS, P.mean(0).round(3).tolist())), flush=True)


def run():
    py = sys.executable
    if not PRIOR.exists():
        subprocess.run([py, __file__, "calibrate"], check=True)
    for r in range(1, MAX_ROUNDS + 1):
        if not open_targets(load()):
            break
        print(f"\n######## round {r}: {len(open_targets(load()))} open", flush=True)
        for phase in ("produce", "verify"):
            subprocess.run([py, __file__, phase, str(r)], check=True)
    subprocess.run([py, __file__, "confirm"], check=True)


if __name__ == "__main__":
    LOOP.mkdir(exist_ok=True)
    cmd = sys.argv[1]
    {"produce": lambda: produce(int(sys.argv[2])), "verify": lambda: verify(int(sys.argv[2])),
     "confirm": confirm, "calibrate": calibrate, "run": run}[cmd]()
