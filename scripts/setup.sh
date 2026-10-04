#!/bin/sh
# Fetch everything that is not stored in this repository, at pinned versions:
#   external/Pain-axis                     the Pain Axis release, for the ten emotion vectors
#   models/Qwen2.5-7B-Instruct-4bit        the model that is steered and that composes
#   models/Qwen2-Audio-7B-Instruct-4bit    the model that listens
#   models/soundfonts/GeneralUser-GS.sf2   the General MIDI soundfont for the second loop
# About 11 GB in all. Existing folders or links are left alone.
set -e
cd "$(dirname "$0")/.."

if [ ! -e external/Pain-axis ]; then
  mkdir -p external
  git clone https://github.com/valen-research/Pain-axis external/Pain-axis
  git -C external/Pain-axis checkout 4d75cd90e206ea962f7a9101e65c85efea56723b
fi

mkdir -p models
if [ ! -e models/Qwen2.5-7B-Instruct-4bit ]; then
  hf download mlx-community/Qwen2.5-7B-Instruct-4bit --revision c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed \
    --local-dir models/Qwen2.5-7B-Instruct-4bit
fi
if [ ! -e models/Qwen2-Audio-7B-Instruct-4bit ]; then
  hf download mlx-community/Qwen2-Audio-7B-Instruct-4bit --revision c65570002626f41b4dc08b7b54f42f99f3e82e7f \
    --local-dir models/Qwen2-Audio-7B-Instruct-4bit
fi
SF=models/soundfonts/GeneralUser-GS.sf2
if [ ! -e "$SF" ]; then
  mkdir -p models/soundfonts
  curl -sSL -o "$SF" https://github.com/mrbumpy409/GeneralUser-GS/raw/97049183643d5fc5a9322a69c5b09efb667c6c3a/GeneralUser-GS.sf2
fi
echo "9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe  $SF" | shasum -a 256 -c -
echo "setup done"
