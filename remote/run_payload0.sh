#!/usr/bin/env bash
# The no-payload control on its own GPU: 200M fresh, then +600M restored (same schedule
# as the payload policy: 200M seed 0, then 600M seed 1 from those params).
set -uo pipefail
cd ~/g1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
.venv/bin/python train/train.py --max-payload 0 --out runs/payload0 > runs/payload0.log 2>&1
echo "payload0 exit $?" >> runs/done.txt
.venv/bin/python train/train.py --max-payload 0 --seed 1 --timesteps 600000000 \
  --restore runs/payload0/params.pkl --out runs/payload0_800M > runs/payload0_800M.log 2>&1
echo "payload0_800M exit $?" >> runs/done.txt
