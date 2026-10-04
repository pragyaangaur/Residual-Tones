"""Summarise loop/state.json: verifier bias, best attempt per target, and whether
the verifier's judgements carry any information about the target."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
s = json.load(open(HERE / (__import__("sys").argv[1] if len(__import__("sys").argv) > 1 else "loop") / "state.json"))
KEYS = list(next(iter(s.values()))["rounds"][0]["probs"])
rows = [(n, v["label"], h) for n, v in s.items() for h in v["rounds"] if "probs" in h]
P = np.array([[h["probs"][k] for k in KEYS] for _, _, h in rows])
prior = P.mean(0)
print("verifier's mean forced-choice probability over all", len(rows), "pieces:")
for k, p in sorted(zip(KEYS, prior), key=lambda x: -x[1]):
    print(f"  {k:17s} {p:.3f}")
lift = np.array([h["probs"][lab] / prior[KEYS.index(lab)] for _, lab, h in rows])
print(f"\nmean p(target)/prior = {lift.mean():.2f} (1.0 = no information)")
rng = np.random.default_rng(0)
labs = [lab for _, lab, _ in rows]
null = [np.mean([P[i, KEYS.index(l)] / prior[KEYS.index(l)] for i, l in enumerate(rng.permutation(labs))]) for _ in range(5000)]
print(f"permutation p = {np.mean(np.array(null) >= lift.mean()):.3f}")
dr = np.array([h["desc_rank"] for _, _, h in rows])
print(f"description rank of target: mean {dr.mean():.2f} (chance 5.0), rank 1 in {int((dr==1).sum())}/{len(dr)} (chance {len(dr)/9:.1f})")
print("\nbest attempt per target (by p_target):")
for n, v in s.items():
    rs = [h for h in v["rounds"] if "probs" in h]
    b = max(rs, key=lambda h: h["p_target"])
    print(f"  {n:11s} r{b['round']} p={b['p_target']:.2f} pick={b['pick']:16s} desc_rank={b['desc_rank']}  trend={[round(h['p_target'],2) for h in rs]}")
