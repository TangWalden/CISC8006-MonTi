#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MONTI_PYTHON="${MONTI_PYTHON:-/home/tjy/miniconda3/envs/monti/bin/python}"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$MONTI_PYTHON")")/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONHASHSEED=0 DGLBACKEND=pytorch
exec "$MONTI_PYTHON" -u "$@"
