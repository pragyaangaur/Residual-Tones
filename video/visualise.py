"""Draw a dark, dotted visualisation of a piece of audio and encode it as MP4.

Every frame reads the audio around that moment. The spectrum is split into 40 bands on a
log scale from 50 Hz to 8 kHz. Dots are spawned in proportion to how loud the moment is and
how much the spectrum has just changed, each dot takes its band from the band energies,
low bands sit near the centre and high bands further out, and louder bands give larger and
brighter dots. Dots fade in fast and fade out slowly, more slowly when the sound is steady.

  visualise.py            render the three featured pieces and the combined reel
"""
import json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.io import wavfile

HERE = Path(__file__).parent
ROOT = HERE.parent
SIZE, FPS = 1080, 30
ENCODER = HERE / "encode"
FONT = "/System/Library/Fonts/Avenir Next.ttc"

PIECES = [
    {"name": "Excitement", "wav": ROOT / "loop_sf/audio/arousal_r8.wav", "slug": "excitement",
     "palette": ["#ffc23d", "#ff6a3d", "#ff3d8b"]},
    {"name": "Sadness", "wav": ROOT / "loop_sf/audio/sadness_r3.wav", "slug": "sadness",
     "palette": ["#2f5bff", "#6a5cff", "#7fd8ff"]},
    {"name": "Pain", "wav": ROOT / "loop_sf/audio/s2_pain_r4.wav", "slug": "pain",
     "palette": ["#7a0616", "#d9142a", "#ff7a1f"]},
]
BG = np.array([7, 7, 11], np.float32) / 255


def hexrgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255


def analyse(x, sr, n_frames):
    """Per video frame: 40 band energies in 0..1, a loudness value and a spectral flux value."""
    edges = np.geomspace(50, 8000, 41)
    win = 2048
    f = np.fft.rfftfreq(win, 1 / sr)
    idx = [(f >= edges[i]) & (f < edges[i + 1]) for i in range(40)]
    B = np.zeros((n_frames, 40))
    rms = np.zeros(n_frames)
    pad = np.pad(x, (win, win))
    for k in range(n_frames):
        c = int(k / FPS * sr) + win
        seg = pad[c - win // 2: c + win // 2] * np.hanning(win)
        mag = np.abs(np.fft.rfft(seg))
        B[k] = [mag[i].mean() if i.any() else 0 for i in idx]
        rms[k] = np.sqrt((seg ** 2).mean())
    db = 20 * np.log10(B + 1e-9)
    lo, hi = np.percentile(db, 30), np.percentile(db, 99.5)
    E = np.clip((db - lo) / (hi - lo), 0, 1)
    flux = np.r_[0, np.maximum(np.diff(E, axis=0), 0).sum(1)]
    flux = np.clip(flux / (np.percentile(flux, 98) + 1e-9), 0, 1)
    loud = np.clip(rms / (rms.max() + 1e-9), 0, 1)
    return E, loud, flux


SPRITES = {}
def sprite(r):
    r = int(max(2, r))
    if r not in SPRITES:
        n = 3 * r
        y, x = np.mgrid[-n:n + 1, -n:n + 1]
        SPRITES[r] = np.exp(-(x ** 2 + y ** 2) / (2 * (r / 1.6) ** 2)).astype(np.float32)
    return SPRITES[r]


def title_layer(text):
    img = Image.new("L", (SIZE, SIZE), 0)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 58, index=0)
    spaced = " ".join(text.upper())
    w = d.textlength(spaced, font=font)
    d.text(((SIZE - w) / 2, 92), spaced, font=font, fill=255)
    return np.asarray(img, np.float32) / 255


def render(piece, frames_dir, seconds=None, start_frame=0, seed=0):
    sr, x = wavfile.read(piece["wav"])
    x = x.astype(np.float32) / 32768
    x = x.mean(1) if x.ndim == 2 else x
    n = int(len(x) / sr * FPS) if seconds is None else int(seconds * FPS)
    E, loud, flux = analyse(x, sr, n)
    rng = np.random.default_rng(seed)
    pal = [hexrgb(h) for h in piece["palette"]]
    title = title_layer(piece["name"])
    cx, cy = SIZE / 2, SIZE / 2 + 40
    dots = []   # [x, y, vx, vy, r, rgb, born, life, peak]
    for k in range(n):
        t = k / FPS
        spawn = int(rng.poisson(1.5 + 26 * loud[k] ** 1.5 + 55 * flux[k]))
        w = E[k] ** 2 + 1e-6
        for b in rng.choice(40, size=spawn, p=w / w.sum()):
            ang = rng.uniform(0, 2 * np.pi)
            rad = 40 + (b / 39) ** 0.8 * 400 + rng.normal(0, 22)
            col = pal[min(2, int(b / 40 * 3))] * (0.75 + 0.5 * rng.random())
            speed = 6 + 40 * loud[k]
            dots.append([cx + rad * np.cos(ang), cy + rad * np.sin(ang), speed * np.cos(ang), speed * np.sin(ang),
                         3 + 16 * E[k, b] ** 1.5, col, t, 0.5 + 2.2 * (1 - flux[k]) * rng.random() + 0.3, 0.35 + 0.65 * E[k, b]])
        canvas = np.zeros((SIZE, SIZE, 3), np.float32)
        alive = []
        for d in dots:
            age = t - d[6]
            if age > d[7]:
                continue
            alive.append(d)
            a = d[8] * min(1, age / 0.08) * (1 - age / d[7]) ** 1.5
            d[0] += d[2] / FPS; d[1] += d[3] / FPS
            s = sprite(d[4]); h = s.shape[0] // 2
            xi, yi = int(d[0]), int(d[1])
            x0, x1, y0, y1 = max(0, xi - h), min(SIZE, xi + h + 1), max(0, yi - h), min(SIZE, yi + h + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            canvas[y0:y1, x0:x1] += a * s[y0 - yi + h: y1 - yi + h, x0 - xi + h: x1 - xi + h, None] * d[5]
        dots = alive
        small = Image.fromarray(np.clip(canvas * 255, 0, 255).astype(np.uint8)).resize((SIZE // 4, SIZE // 4), Image.BILINEAR)
        glow = np.asarray(small.filter(ImageFilter.GaussianBlur(5)).resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255
        img = 1 - np.exp(-(canvas + 1.4 * glow) * 1.0)
        fade = min(1, t / 0.6, (n / FPS - t) / 0.6)
        img = BG + img * (1 - BG)
        img += title[..., None] * 0.86 * fade * (1 - img)
        Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8)).save(frames_dir / f"{start_frame + k:05d}.png", compress_level=1)
    return n


def blank(frames_dir, start, count):
    img = Image.fromarray((np.broadcast_to(BG, (SIZE, SIZE, 3)) * 255).astype(np.uint8))
    for k in range(count):
        img.save(frames_dir / f"{start + k:05d}.png", compress_level=1)


def encode(frames_dir, audio_wav, out):
    m4a = frames_dir / "audio.m4a"
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac@44100", "-b", "192000", str(audio_wav), str(m4a)], check=True)
    subprocess.run([str(ENCODER), str(frames_dir), str(FPS), str(m4a), str(out)], check=True)


def main(work):
    if not ENCODER.exists():
        subprocess.run(["swiftc", "-O", "-swift-version", "5", str(HERE / "encode.swift"), "-o", str(ENCODER)], check=True)
    work = Path(work)
    gap = 1.0
    reel_frames = work / "reel"
    shutil.rmtree(reel_frames, ignore_errors=True); reel_frames.mkdir(parents=True)
    reel_audio, pos = [], 0
    for i, p in enumerate(PIECES):
        fd = work / p["slug"]
        shutil.rmtree(fd, ignore_errors=True); fd.mkdir(parents=True)
        n = render(p, fd, seed=i)
        encode(fd, p["wav"], HERE / f"{p['slug']}.mp4")
        for k in range(n):
            shutil.copy(fd / f"{k:05d}.png", reel_frames / f"{pos + k:05d}.png")
        pos += n
        sr, x = wavfile.read(p["wav"])
        reel_audio.append(x[: int(n / FPS * sr)])
        if i < len(PIECES) - 1:
            g = int(gap * FPS)
            blank(reel_frames, pos, g); pos += g
            reel_audio.append(np.zeros((int(g / FPS * sr), x.shape[1]), x.dtype))
    wavfile.write(reel_frames / "reel.wav", sr, np.concatenate(reel_audio))
    encode(reel_frames, reel_frames / "reel.wav", HERE / "residual-tones-reel.mp4")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE / "frames")
