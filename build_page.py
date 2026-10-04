"""Build the Residual Tones page in site/ from template.html, the results in out/ and the
encoded audio. Audio is copied as .mp4 (AAC), which the artifact host serves."""
import json, shutil
from pathlib import Path

HERE = Path(__file__).parent
OUT, SITE = HERE / "out", HERE / "site"


def main():
    m = json.load(open(OUT / "sound_meta.json"))
    s = json.load(open(OUT / "steered.json"))
    data = {"names": m["names"], "cos": m["cos"], "pca": [p[:2] for p in m["pca"]], "pca_var": m["pca_var"],
            "sounds": m["sounds"], "samples": s["samples"], "k": s["k"], "prompt": s["prompt"], "fidelity": 0.751}
    json.dump(data, open(OUT / "page_data.json", "w"))
    (SITE / "audio").mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.m4a"):
        shutil.copy(f, SITE / "audio" / f"{f.stem}.mp4")
    page = (HERE / "template.html").read_text().replace("__DATA__", json.dumps(data))
    (SITE / "index.html").write_text(page)
    print("built", SITE / "index.html")


if __name__ == "__main__":
    main()
