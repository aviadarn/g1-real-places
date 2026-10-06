#!/usr/bin/env bash
# Stand-still fine-tune: +200M steps from an 800M policy, 25% all-zero commands and the
# stand-still penalty restricted to legs+waist. Usage: run_finetune.sh <max_payload> <run>
set -uo pipefail
cd ~/g1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
until [ -f "runs/$2_800M/params.pkl" ]; do sleep 30; done
.venv/bin/python train/train.py --max-payload "$1" --seed 2 --timesteps 200000000 --p-stand 0.25 --stand-fix \
  --restore "runs/$2_800M/params.pkl" --out "runs/$2_1B" > "runs/$2_1B.log" 2>&1
echo "$2_1B exit $?" >> runs/done.txt
