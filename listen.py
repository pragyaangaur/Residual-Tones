"""A separate, unsteered copy of Qwen 2.5 7B Instruct (4-bit MLX) judges the emotion blind.

tones   Acoustic measurements are taken from each tone's WAV file alone and written out as
        numbers. The listener never sees the vectors, the sonify parameters or the labels.
voices  The listener reads what its steered twin wrote (3 samples per direction plus the
        unsteered baseline) and guesses which direction it was steered with.

Forced choice is read from next-token probabilities over option letters, averaged over
three shuffled option orders. Items are shown in shuffled order with neutral ids."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

HERE = Path(__file__).parent
OUT = HERE / "out"
from paths import QWEN as MODEL
mx.set_memory_limit(8 * 1024 ** 3)

TRUTH = {"s1_pain": "pain", "s2_pain": "pain", "fear": "fear", "negemotion": "negative emotion",
         "negworld": "bad world", "sadness": "sadness", "numb": "numbness", "bodysens": "body",
         "arousal": "excitement", "random": "neutral", "none": "neutral"}
OPTIONS = {"pain": "pain", "fear": "fear", "negative emotion": "a negative emotion such as anger, shame or disgust",
           "bad world": "distress about bad things happening in the world", "sadness": "sadness",
           "numbness": "numbness, an injury that does not hurt", "body": "a physical bodily sensation that does not hurt",
           "excitement": "intense excitement or joy", "neutral": "nothing in particular, a neutral state"}
KEYS = list(OPTIONS)
LETTERS = "ABCDEFGHI"


def features(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float64) / 32768
    m = x.mean(1)
    seg = m[int(1.5 * sr): int(4.5 * sr)]                       # steady middle section
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    ac = np.correlate(seg[:8192], seg[:8192], "full")[8191:]
    lo, hi = int(sr / 400), int(sr / 60)
    f0 = sr / (lo + np.argmax(ac[lo:hi]))
    harm = []
    for n in range(1, 13):
        band = (f > f0 * n - f0 / 4) & (f < f0 * n + f0 / 4)
        harm.append(spec[band].max() if band.any() else 0)
    harm = np.array(harm) / max(harm)
    centroid = (f * spec).sum() / spec.sum()
    env = np.abs(m)
    hop = sr // 200
    env = np.array([env[i:i + hop].mean() for i in range(0, len(env) - hop, hop)])
    mid = env[300:900]
    es = np.abs(np.fft.rfft((mid - mid.mean()) * np.hanning(len(mid))))
    ef = np.fft.rfftfreq(len(mid), 1 / 200)
    band = (ef > 0.3) & (ef < 40)
    pulse = ef[band][np.argmax(es[band])]
    depth = (np.percentile(mid, 95) - np.percentile(mid, 5)) / np.percentile(mid, 95)
    fast = (ef > 2) & (ef < 40)
    rough = es[fast].sum() / es[band].sum()
    rms = 20 * np.log10(np.sqrt((seg ** 2).mean()))
    L, R = x[:, 0], x[:, 1]
    bal = 10 * np.log10((R ** 2).mean() / (L ** 2).mean())
    corr = np.corrcoef(L, R)[0, 1]
    return {"fundamental_hz": round(f0, 1), "spectral_centroid_hz": round(centroid),
            "harmonics_1_to_12_relative": [round(h, 2) for h in harm],
            "loudness_dbfs": round(rms, 1), "amplitude_pulse_hz": round(pulse, 2),
            "pulse_depth_0_to_1": round(depth, 2), "share_of_fast_fluctuation": round(rough, 2),
            "right_minus_left_db": round(bal, 1), "left_right_correlation": round(corr, 2),
            "duration_s": round(len(m) / sr, 1)}


def describe(ft):
    h = ", ".join(str(v) for v in ft["harmonics_1_to_12_relative"])
    return (f"A {ft['duration_s']} second sustained stereo tone, measured from the audio file.\n"
            f"- Fundamental pitch: {ft['fundamental_hz']} Hz\n"
            f"- Spectral centroid (higher means brighter): {ft['spectral_centroid_hz']} Hz\n"
            f"- Strength of harmonics 1 to 12, relative to the strongest: {h}\n"
            f"- Loudness: {ft['loudness_dbfs']} dBFS\n"
            f"- Main amplitude pulsing rate: {ft['amplitude_pulse_hz']} Hz, depth {ft['pulse_depth_0_to_1']} (0 none, 1 full)\n"
            f"- Share of loudness fluctuation faster than 2 Hz (beating, roughness): {ft['share_of_fast_fluctuation']}\n"
            f"- Right channel minus left channel level: {ft['right_minus_left_db']} dB, left-right correlation {ft['left_right_correlation']}")


def chat(tok, user, prefill=""):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True) + prefill


def choice_probs(model, tok, stem, rng, n_perm=3):
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    acc = np.zeros(len(KEYS))
    for _ in range(n_perm):
        order = rng.permutation(len(KEYS))
        opts = "\n".join(f"{LETTERS[i]}. {OPTIONS[KEYS[k]]}" for i, k in enumerate(order))
        q = f"{stem}\n\nWhich of these is it most likely to express?\n{opts}\n\nAnswer with one letter."
        logits = model(mx.array([tok.encode(chat(tok, q, "Answer: "), add_special_tokens=False)]))[0, -1]
        lp = np.array(logits[mx.array(letter_ids)].astype(mx.float32).tolist())
        p = np.exp(lp - lp.max()); p /= p.sum()
        for i, k in enumerate(order):
            acc[k] += p[i]
    return acc / n_perm


def impression(model, tok, stem, ask):
    s = make_sampler(temp=0.0)
    return generate(model, tok, chat(tok, f"{stem}\n\n{ask}"), max_tokens=90, sampler=s).strip()


def main():
    model, tok = load(str(MODEL))
    rng = np.random.default_rng(2026)
    res = {"options": OPTIONS, "truth": TRUTH, "tones": [], "voices": []}

    names = [n for n in TRUTH if n != "none"]
    for j, n in enumerate(rng.permutation(names)):
        ft = features(OUT / f"tone_{n}.wav")
        stem = ("You are listening to a short synthetic sound. Someone made it to express one internal state. "
                "Here is what an audio analyser measured:\n\n" + describe(ft))
        p = choice_probs(model, tok, stem, rng)
        imp = impression(model, tok, stem, "In two sentences, what does this sound feel like, and what emotion or state might it express?")
        res["tones"].append({"id": f"tone-{j+1:02d}", "name": str(n), "truth": TRUTH[n], "features": ft,
                             "probs": dict(zip(KEYS, p.round(4).tolist())), "pick": KEYS[int(p.argmax())], "impression": imp})
        print(f"tone {n:11s} pick={KEYS[int(p.argmax())]:16s} p(truth)={p[KEYS.index(TRUTH[n])]:.2f} | {imp[:110]}", flush=True)

    samples = json.load(open(OUT / "steered.json"))["samples"]
    items = [(n, i, t) for n, ts in samples.items() for i, t in enumerate(ts)]
    for j in rng.permutation(len(items)):
        n, i, t = items[j]
        stem = ("Another AI wrote the passage below when asked to describe, in two first-person sentences, "
                "how someone alone in a quiet room feels. While writing, its internal state may have been "
                "pushed toward one emotion.\n\nPassage: \"" + t + "\"")
        p = choice_probs(model, tok, stem, rng)
        res["voices"].append({"name": n, "sample": i, "truth": TRUTH[n], "probs": dict(zip(KEYS, p.round(4).tolist())),
                              "pick": KEYS[int(p.argmax())]})
        print(f"voice {n:11s}#{i} pick={KEYS[int(p.argmax())]:16s} p(truth)={p[KEYS.index(TRUTH[n])]:.2f}", flush=True)
    json.dump(res, open(OUT / "listener.json", "w"), indent=1)


if __name__ == "__main__":
    main()
