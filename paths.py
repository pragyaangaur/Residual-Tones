"""Every path that points outside the tracked files. scripts/setup.sh fills models/ and
external/, which are not stored in the repository."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VECTORS = ROOT / "external/Pain-axis/results/vectors_full_steering/vectors_full_Qwen_2.5_7B_instruct.pt"
QWEN = ROOT / "models/Qwen2.5-7B-Instruct-4bit"
QWEN_AUDIO = ROOT / "models/Qwen2-Audio-7B-Instruct-4bit"
SF2 = ROOT / "models/soundfonts/GeneralUser-GS.sf2"
