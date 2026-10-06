#!/usr/bin/env bash
# Both trainings back to back: the payload policy, then the no-payload ablation.
# Thread caps: avoid the pids.max blowup seen on vast boxes before.
set -uo pipefail
cd ~/g1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
for spec in "10 payload10" "0 payload0"; do
  set -- $spec
  .venv/bin/python train/train.py --max-payload "$1" --out "runs/$2" > "runs/$2.log" 2>&1
  echo "$2 exit $?" >> runs/done.txt
done
