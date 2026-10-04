"""Turn each released emotion vector into a sound. Every parameter comes from the vector's
own geometry, with one fixed mapping for all of them, so emotions the model holds close
together sound alike.

  timbre     24 harmonics. Harmonic n gets |u . r_n|, where r_n is a fixed random unit
             vector. Random projection roughly keeps angles, so similar directions get
             similar harmonic profiles.
  roughness  harmonics whose projection is negative are detuned by 9 cents and panned
             right, so the sign pattern is heard as beating.
  pitch      the vector's first principal coordinate across the ten vectors, 98 to 196 Hz.
  brightness the second principal coordinate sets a one-pole low-pass cutoff.
  pulse      excess kurtosis of the components (few dimensions carrying the direction)
             sets a tremolo rate.
  loudness   log of the raw norm, within a 9 dB range.
"""
import json
from pathlib import Path
import numpy as np, torch
from scipy.io import wavfile
from scipy.stats import kurtosis

HERE = Path(__file__).parent
OUT = HERE / "out"
from paths import VECTORS as VEC
SR, DUR, NH = 22050, 6.0, 24

d = torch.load(VEC, map_location="cpu", weights_only=False)
V = {k.removesuffix("_vector"): v.float().numpy().astype(np.float64) for k, v in d.items() if k.endswith("_vector")}
names = list(V)
U = np.stack([V[n] / np.linalg.norm(V[n]) for n in names])
cos = U @ U.T

C = U - U.mean(0)
_, S, Wt = np.linalg.svd(C, full_matrices=False)
pcs = C @ Wt[:3].T
var = (S ** 2 / (S ** 2).sum())[:3]
R = np.random.default_rng(0).normal(size=(NH, U.shape[1]))
R /= np.linalg.norm(R, axis=1, keepdims=True)
proj = U @ R.T * np.sqrt(U.shape[1])           # about N(0, 1) per entry

def scale(x, lo, hi):
    x = np.asarray(x, float)
    return lo + (hi - lo) * (x - x.min()) / (x.max() - x.min())

f0 = 98 * 2 ** scale(pcs[:, 0], 0, 1)
cut = 600 * 2 ** scale(pcs[:, 1], 0, 3)
kurt = np.array([kurtosis(u) for u in U])
trem = scale(np.log1p(np.maximum(kurt, 0)), 0.5, 6)
norms = np.array([np.linalg.norm(V[n]) for n in names])
gain_db = scale(np.log(norms), -9, 0)

t = np.arange(int(SR * DUR)) / SR
meta = {"names": names, "cos": cos.round(4).tolist(), "pca_var": var.round(4).tolist(),
        "pca": pcs.round(4).tolist(), "sounds": {}}
for i, n in enumerate(names):
    L = np.zeros_like(t); Rr = np.zeros_like(t)
    a = np.abs(proj[i]); a /= a.max()
    for h in range(NH):
        f = f0[i] * (h + 1)
        if f > SR / 2.2:
            break
        ph = np.random.default_rng([i, h]).uniform(0, 2 * np.pi)
        w = a[h] / (h + 1) ** 0.6
        if proj[i, h] >= 0:
            s = np.sin(2 * np.pi * f * t + ph); L += w * s; Rr += 0.35 * w * s
        else:
            s = np.sin(2 * np.pi * f * 2 ** (9 / 1200) * t + ph) + np.sin(2 * np.pi * f * t)
            Rr += 0.5 * w * s; L += 0.175 * w * s
    alpha = 1 - np.exp(-2 * np.pi * cut[i] / SR)
    for ch in (L, Rr):
        y = 0.0
        for k in range(len(ch)):
            y += alpha * (ch[k] - y); ch[k] = y
    env = np.minimum(1, t / 0.8) * np.minimum(1, (DUR - t) / 1.5)
    env *= 1 - 0.35 * (0.5 + 0.5 * np.sin(2 * np.pi * trem[i] * t))
    st = np.stack([L, Rr], 1) * env[:, None]
    st *= 10 ** (gain_db[i] / 20) * 0.8 / np.abs(st).max()
    wavfile.write(OUT / f"tone_{n}.wav", SR, (st * 32767).astype(np.int16))
    meta["sounds"][n] = {"f0": round(float(f0[i]), 2), "cutoff": round(float(cut[i]), 1),
                         "tremolo_hz": round(float(trem[i]), 2), "gain_db": round(float(gain_db[i]), 2),
                         "norm": round(float(norms[i]), 3), "kurtosis": round(float(kurt[i]), 2),
                         "harmonics": proj[i].round(3).tolist()}
json.dump(meta, open(OUT / "sound_meta.json", "w"), indent=1)
print("pca var", var)
for n in names:
    s = meta["sounds"][n]; print(f"{n:11s} f0={s['f0']:6.1f} cut={s['cutoff']:7.0f} trem={s['tremolo_hz']:.2f} gain={s['gain_db']:5.1f} kurt={s['kurtosis']}")
print(np.array2string(cos, precision=2, max_line_width=200))
