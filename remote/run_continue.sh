#!/usr/bin/env bash
# Continue both policies to 800M total steps: +600M each, restored from their 200M params.
# payload10 starts now (sharing the GPU with payload0's first leg); payload0 continues
# once its first 200M finish.
set -uo pipefail
cd ~/g1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
.venv/bin/python train/train.py --max-payload 10 --seed 1 --timesteps 600000000 \
  --restore runs/payload10/params.pkl --out runs/payload10_800M > runs/payload10_800M.log 2>&1 &
until grep -q "payload0 exit" runs/done.txt 2>/dev/null; do sleep 30; done
.venv/bin/python train/train.py --max-payload 0 --seed 1 --timesteps 600000000 \
  --restore runs/payload0/params.pkl --out runs/payload0_800M > runs/payload0_800M.log 2>&1
wait
echo "continue done" >> runs/done.txt
