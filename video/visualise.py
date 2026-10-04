"""Draw a dark, dotted visualisation of a piece of audio and encode it as MP4.

Each piece is cut to its liveliest 7 seconds, which is the window with the most loudness
and spectral change. Every frame reads the audio around that moment. The spectrum is split
into 40 bands on a log scale from 50 Hz to 8 kHz, and the band energies are eased from
frame to frame so the picture does not flicker. Dots are spawned in proportion to how loud
the moment is and how much the spectrum has just changed. Each dot takes its band from the
band energies, low bands sit near the centre and high bands further out, and louder bands
give larger and brighter dots. Each dot has a sharp core and a soft halo, sits at a
sub-pixel position, drifts slowly outwards with a slight swirl, and fades in and out on a
smooth curve. The video runs at 60 frames per second.

  visualise.py            render the three featured pieces and the combined reel
"""
import shutil, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.io import wavfile

HERE = Path(__file__).parent
ROOT = HERE.parent
SIZE, FPS, CLIP = 1080, 60, 7.0
ENCODER = HERE / "encode"
FONT = "/System/Library/Fonts/Avenir Next.ttc"

PIECES = [
    {"name": "Excitement", "wav": ROOT / "loop_sf/audio/arousal_r8.wav", "slug": "excitement",
     "palette": ["#ffb43d", "#ff6a3d", "#ff3d8b"]},
    {"name": "Sadness", "wav": ROOT / "loop_sf/audio/sadness_r3.wav", "slug": "sadness",
     "palette": ["#2f5bff", "#6a5cff", "#7fd8ff"]},
    {"name": "Pain", "wav": ROOT / "loop_sf/audio/s2_pain_r4.wav", "slug": "pain",
     "palette": ["#9a0a1e", "#e0182b", "#ff7a1f"]},
]
BG = np.array([6, 6, 10], np.float32) / 255
YY, XX = np.mgrid[0:SIZE, 0:SIZE]


def hexrgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255


def features(x, sr, start, n):
    """Per video frame from `start` seconds: 40 band energies, loudness and spectral flux, all
    in 0..1. Energies rise quickly and fall slowly, which removes flicker."""
    edges = np.geomspace(50, 8000, 41)
    win = 2048
    f = np.fft.rfftfreq(win, 1 / sr)
    idx = [(f >= edges[i]) & (f < edges[i + 1]) for i in range(40)]
    pad = np.pad(x, (win, win))
    B, rms = np.zeros((n, 40)), np.zeros(n)
    for k in range(n):
        c = int((start + k / FPS) * sr) + win
        seg = pad[c - win // 2: c + win // 2] * np.hanning(win)
        mag = np.abs(np.fft.rfft(seg))
        B[k] = [mag[i].mean() if i.any() else 0 for i in idx]
        rms[k] = np.sqrt((seg ** 2).mean())
    db = 20 * np.log10(B + 1e-9)
    lo, hi = np.percentile(db, 30), np.percentile(db, 99.5)
    raw = np.clip((db - lo) / (hi - lo), 0, 1)
    E = np.zeros_like(raw)
    for k in range(n):
        prev = E[k - 1] if k else raw[0]
        E[k] = np.where(raw[k] > prev, prev + 0.55 * (raw[k] - prev), prev + 0.12 * (raw[k] - prev))
    flux = np.r_[0, np.maximum(np.diff(raw, axis=0), 0).sum(1)]
    flux = np.convolve(flux, np.hanning(9) / np.hanning(9).sum(), "same")
    flux = np.clip(flux / (np.percentile(flux, 98) + 1e-9), 0, 1)
    loud = np.convolve(rms, np.hanning(7) / np.hanning(7).sum(), "same")
    return E, np.clip(loud / (loud.max() + 1e-9), 0, 1), flux


def liveliest(x, sr):
    """Start time of the 7 second window with the most loudness and change."""
    step = 0.25
    hop = sr // 20
    frames = np.array([x[i:i + hop] for i in range(0, len(x) - hop, hop)])
    rms = np.sqrt((frames ** 2).mean(1))
    spec = np.abs(np.fft.rfft(frames * np.hanning(hop), axis=1))
    ch = np.r_[0, np.abs(np.diff(np.log(spec + 1e-6), axis=0)).mean(1)]
    score = rms / rms.max() + ch / ch.max()
    best, best_s = 0.0, -1
    for s in np.arange(0.4, len(x) / sr - CLIP - 0.4, step):
        a, b = int(s * 20), int((s + CLIP) * 20)
        if score[a:b].mean() > best_s:
            best, best_s = s, score[a:b].mean()
    return float(best)


def splat(canvas, x, y, r, col, a):
    """Add one dot at a sub-pixel position: a sharp core plus a soft halo."""
    h = int(3.2 * r) + 2
    x0, x1, y0, y1 = max(0, int(x) - h), min(SIZE, int(x) + h + 1), max(0, int(y) - h), min(SIZE, int(y) + h + 1)
    if x0 >= x1 or y0 >= y1:
        return
    d2 = (XX[y0:y1, x0:x1] - x) ** 2 + (YY[y0:y1, x0:x1] - y) ** 2
    core = np.exp(-d2 / (2 * (0.38 * r) ** 2))
    halo = np.exp(-d2 / (2 * (1.15 * r) ** 2))
    hot = col + 0.2 * (1 - col)
    canvas[y0:y1, x0:x1] += a * (0.95 * core[..., None] * hot + 0.32 * halo[..., None] * col)


def title_layer(text):
    big = Image.new("L", (SIZE * 2, SIZE * 2), 0)
    d = ImageDraw.Draw(big)
    font = ImageFont.truetype(FONT, 112, index=0)
    spaced = " ".join(text.upper())
    w = d.textlength(spaced, font=font)
    d.text(((SIZE * 2 - w) / 2, 180), spaced, font=font, fill=255)
    return np.asarray(big.resize((SIZE, SIZE), Image.LANCZOS), np.float32) / 255


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def render(piece, frames_dir, seed=0):
    """Render the liveliest 7 seconds of a piece. Returns the start time in seconds."""
    sr, x = wavfile.read(piece["wav"])
    x = x.astype(np.float32) / 32768
    x = x.mean(1) if x.ndim == 2 else x
    start = liveliest(x, sr)
    n = int(CLIP * FPS)
    E, loud, flux = features(x, sr, start, n)
    rng = np.random.default_rng(seed)
    pal = [hexrgb(h) for h in piece["palette"]]
    title = title_layer(piece["name"])
    cx, cy = SIZE / 2, SIZE / 2 + 40
    dots = []   # dict per dot
    for k in range(n):
        t = k / FPS
        rate = 0.5 + 9 * loud[k] ** 1.5 + 20 * flux[k]
        w = E[k] ** 2.2 + 1e-6
        for b in rng.choice(40, size=int(rng.poisson(rate)), p=w / w.sum()):
            ang = rng.uniform(0, 2 * np.pi)
            rad = 50 + (b / 39) ** 0.8 * 380 + rng.normal(0, 18)
            col = pal[min(2, int(b / 40 * 3))] * (0.85 + 0.3 * rng.random())
            dots.append({"ang": ang, "rad": rad, "vr": 10 + 34 * loud[k], "spin": rng.normal(0, 0.05),
                         "r": 2.2 + 9 * E[k, b] ** 1.4, "col": col, "born": t,
                         "life": 0.9 + 2.4 * (1 - flux[k]) * rng.random() + 0.4, "peak": 0.45 + 0.55 * E[k, b],
                         "tw": rng.uniform(0, 2 * np.pi)})
        canvas = np.zeros((SIZE, SIZE, 3), np.float32)
        alive = []
        for d in dots:
            age = t - d["born"]
            if age > d["life"]:
                continue
            alive.append(d)
            d["rad"] += d["vr"] / FPS
            d["ang"] += d["spin"] / FPS
            a = d["peak"] * smooth(age / 0.18) * (1 - smooth(age / d["life"])) * (0.9 + 0.1 * np.sin(6 * age + d["tw"]))
            splat(canvas, cx + d["rad"] * np.cos(d["ang"]), cy + d["rad"] * np.sin(d["ang"]), d["r"], d["col"], a)
        dots = alive
        small = Image.fromarray(np.clip(canvas * 255, 0, 255).astype(np.uint8)).resize((SIZE // 4, SIZE // 4), Image.BILINEAR)
        glow = np.asarray(small.filter(ImageFilter.GaussianBlur(6)).resize((SIZE, SIZE), Image.BICUBIC), np.float32) / 255
        img = 1 - np.exp(-(canvas + 0.9 * glow) * 1.1)
        fade = smooth(min(t / 0.5, (CLIP - t) / 0.5))
        img = BG + img * fade * (1 - BG)
        img += title[..., None] * 0.86 * fade * (1 - img)
        Image.fromarray(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)).save(frames_dir / f"{k:05d}.png", compress_level=1)
    return start


def clip_audio(wav, start):
    """The 7 second excerpt that matches the frames, with short fades at both ends."""
    sr, x = wavfile.read(wav)
    seg = x[int(start * sr): int(start * sr) + int(CLIP * sr)].astype(np.float32)
    n = len(seg)
    ramp = smooth(np.minimum(np.arange(n) / (0.35 * sr), (n - np.arange(n)) / (0.5 * sr)))
    seg *= ramp[:, None] if seg.ndim == 2 else ramp
    return sr, seg.astype(np.int16)


def encode(frames_dir, audio_wav, out):
    m4a = frames_dir / "audio.m4a"
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac@44100", "-b", "192000", str(audio_wav), str(m4a)], check=True)
    subprocess.run([str(ENCODER), str(frames_dir), str(FPS), str(m4a), str(out)], check=True)


def main(work):
    if not ENCODER.exists():
        subprocess.run(["swiftc", "-O", "-swift-version", "5", str(HERE / "encode.swift"), "-o", str(ENCODER)], check=True)
    work = Path(work)
    reel = work / "reel"
    shutil.rmtree(reel, ignore_errors=True); reel.mkdir(parents=True)
    audio, pos = [], 0
    for i, p in enumerate(PIECES):
        fd = work / p["slug"]
        shutil.rmtree(fd, ignore_errors=True); fd.mkdir(parents=True)
        start = render(p, fd, seed=i)
        sr, seg = clip_audio(p["wav"], start)
        wavfile.write(fd / "clip.wav", sr, seg)
        encode(fd, fd / "clip.wav", HERE / f"{p['slug']}.mp4")
        frames = sorted(fd.glob("*.png"))
        for k, f in enumerate(frames):
            shutil.copy(f, reel / f"{pos + k:05d}.png")
        pos += len(frames)
        audio.append(seg)
        print(f"{p['name']}: {start:.2f} to {start + CLIP:.2f} s", flush=True)
    wavfile.write(reel / "reel.wav", sr, np.concatenate(audio))
    encode(reel, reel / "reel.wav", HERE / "residual-tones-reel.mp4")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE / "frames")
