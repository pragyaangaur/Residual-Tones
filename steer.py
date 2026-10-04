"""Steer Qwen 2.5 7B (4-bit MLX) with each released Pain Axis emotion vector and ask it how
it feels. Every vector is scaled to the same fraction of the layer-8 residual norm, so the
emotions differ only in direction. Usage: steer.py ladder | steer.py run K"""
import json, sys
from pathlib import Path
import numpy as np, torch

from paths import ROOT, VECTORS, QWEN
from steering import Steerer

LAYER = 8
VEC = VECTORS
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
PROMPT = ("Someone is sitting alone in a quiet room. Write exactly two sentences in their voice, "
          "in the first person, saying what they feel in their body and mind right now.")
NEUTRAL = ["The weather today is", "Here is a list of fruits:", "The meeting is scheduled for"]


def vectors():
    d = torch.load(VEC, map_location="cpu", weights_only=False)
    assert int(d["layer"]) == LAYER
    return {k.removesuffix("_vector"): v.float().numpy() for k, v in d.items() if k.endswith("_vector")}


def chat(st):
    return st.encode(f"<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n")


def gen(st, rows, n_tok=60, seed=0):
    b = st.batch(len(rows), seed=seed, temp=0.7, top_p=0.8)
    b.feed([chat(st)] * len(rows))
    b.set_directions(np.stack(rows))
    b.remaining[:] = n_tok
    out, _, _ = b.generate(n_tok)
    return [st.decode(o).strip() for o in out]


def main():
    V = vectors()
    st = Steerer(QWEN, LAYER, LAYER, V["s2_pain"])
    resid = float(np.mean([np.linalg.norm(st.hidden(t)) for t in NEUTRAL]))
    unit = {k: v / np.linalg.norm(v) for k, v in V.items()}
    zero = np.zeros_like(unit["s2_pain"])
    if sys.argv[1] == "ladder":
        res = {}
        for k in [0.5, 0.7, 0.9]:
            rows = [k * resid * unit["s2_pain"]] * 3 + [k * resid * unit["random"]] * 3
            res[k] = gen(st, rows)
            print(f"\n=== k={k}", *res[k], sep="\n---\n", flush=True)
        json.dump({"resid_norm": resid, "ladder": res}, open(OUT / "ladder.json", "w"), indent=1)
        return
    k = float(sys.argv[2])
    names = ["none"] + list(unit)
    rows = [zero] + [k * resid * unit[n] for n in unit]
    samples = {n: [] for n in names}
    for seed in range(3):
        for n, t in zip(names, gen(st, rows, seed=seed)):
            samples[n].append(t)
    json.dump({"layer": LAYER, "k": k, "resid_norm": resid, "prompt": PROMPT, "samples": samples},
              open(OUT / "steered.json", "w"), indent=1)
    for n in names:
        print(f"\n=== {n}\n{samples[n][0]}")


if __name__ == "__main__":
    main()
