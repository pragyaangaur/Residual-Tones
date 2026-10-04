"""Read the steered passages aloud. The first of the three samples for each direction was
chosen before reading any of them, and is spoken with the macOS Samantha voice."""
import json, subprocess
from pathlib import Path

OUT = Path(__file__).parent / "out"


def main():
    samples = json.load(open(OUT / "steered.json"))["samples"]
    for n, texts in samples.items():
        txt, aiff, m4a = OUT / f"voice_{n}.txt", OUT / f"voice_{n}.aiff", OUT / f"voice_{n}.m4a"
        txt.write_text(texts[0])
        subprocess.run(["say", "-v", "Samantha", "-r", "165", "-f", str(txt), "-o", str(aiff)], check=True)
        subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "48000", str(aiff), str(m4a)], check=True)
        aiff.unlink()
        print("spoke", n)


if __name__ == "__main__":
    main()
