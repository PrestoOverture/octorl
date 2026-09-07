#!/usr/bin/env bash
set -u
export PATH=/root/miniconda3/bin:$PATH
export PYTHONHASHSEED=20260905
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd /root/autodl-tmp/s3
python measure.py selftest > remote_validation.log 2>&1 && python measure.py validate >> remote_validation.log 2>&1
validation_status=$?
if [ "$validation_status" -ne 0 ]; then
  printf '%s\n' "$validation_status" > validation_exit_status.txt
  exit "$validation_status"
fi
date -u +%s > job_started_unix.txt
timeout --signal=TERM --kill-after=60 18000 python -u measure.py run --output run_20260905 --batch-size 8 --max-hours 5 --hourly-rate-cny 2.18 > inference.log 2>&1
run_status=$?
printf '%s\n' "$run_status" > exit_status.txt
date -u +%s > job_finished_unix.txt
exit "$run_status"
