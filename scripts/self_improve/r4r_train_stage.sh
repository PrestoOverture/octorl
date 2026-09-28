#!/bin/bash
set -euo pipefail
export PATH=/root/miniconda3/bin:$PATH
R4R_WORKSPACE="${R4R_WORKSPACE:-/root/octorl_r3}"
export PYTHONPATH="$R4R_WORKSPACE"
exec python3 "$PYTHONPATH/scripts/self_improve/r4r_launch.py" stage "$@"
