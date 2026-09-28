#!/bin/bash
set -euo pipefail
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH="${R4R_WORKSPACE:-/root/octorl_r3}"
cd "$PYTHONPATH"
exec python3 "$PYTHONPATH/scripts/self_improve/r4r_preflight.py" "$@"
