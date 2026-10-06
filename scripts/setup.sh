#!/usr/bin/env bash
# Python env + Unitree's G1 model and pretrained walking policy (BSD-3, pinned).
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
if [ ! -d third_party/unitree_rl_gym ]; then
  git clone https://github.com/unitreerobotics/unitree_rl_gym.git third_party/unitree_rl_gym
  git -C third_party/unitree_rl_gym checkout 276801e46c5d433564f24658bac64f254b7d2d4b
fi
