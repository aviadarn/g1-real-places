#!/usr/bin/env bash
# One-time setup on a vast.ai GPU box (pytorch/pytorch CUDA 12.x image).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git >/dev/null)
pip install -q uv
cd ~/g1
uv venv -q --clear --python 3.12 .venv
uv pip install -q --python .venv/bin/python -r train/requirements.txt "jax[cuda12]==0.9.2"
.venv/bin/python - <<'PY'
import jax; print("devices", jax.devices())
import mujoco_warp, warp; print("warp", warp.__version__)
PY
