#!/bin/bash
set -eEuo pipefail

# R4: run all pre-registered jobs serially on one GPU. Any failure stops the queue.
# Re-running skips jobs with a DONE marker and refuses to touch a job that started but did not
# finish: continuing or discarding a stopped branch is a human stop-loss decision.

export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=/root/octorl_r3
cd "${R4_WORKSPACE:-/root/octorl_r3}"
ROOT="${R4_ROOT:-/root/autodl-tmp/octorl_r4}"
LOG="$ROOT/logs"
mkdir -p "$LOG"
CURRENT=start
trap 'echo "$(date -Iseconds) QUEUE_FAILED at: $CURRENT" | tee -a "$LOG/queue.log"' ERR

guard() {  # $1 = job name, $2 = job dir
    CURRENT="$1"
    if [ -f "$LOG/DONE_$1" ]; then
        echo "$(date -Iseconds) SKIP $1 (done)" | tee -a "$LOG/queue.log"
        return 1
    fi
    if [ -e "$2" ]; then
        echo "$(date -Iseconds) REFUSE $1: $2 exists without DONE marker" | tee -a "$LOG/queue.log"
        exit 3
    fi
    echo "$(date -Iseconds) START $1" | tee -a "$LOG/queue.log"
}

branch() {  # $1 = seed, $2 = arm
    local name="seed_$1_$2"
    guard "$name" "$ROOT/seed_$1/$2" || return 0
    python3 scripts/self_improve/r4_run_branch.py --seed "$1" --arm "$2" --execute 2>&1 | tee "$LOG/$name.log"
    touch "$LOG/DONE_$name"
    echo "$(date -Iseconds) DONE $name" | tee -a "$LOG/queue.log"
}

warmup() {
    local name=seed_2718_warmup dir="$ROOT/seed_2718/warmup"
    guard "$name" "$dir" || return 0
    bash scripts/self_improve/r4_warmup.sh 2>&1 | tee "$LOG/$name.log"
    test "$(wc -l < "$dir/attributed.jsonl")" -eq 320
    touch "$LOG/DONE_$name"
    echo "$(date -Iseconds) DONE $name" | tee -a "$LOG/queue.log"
}

branch 42 fixed
branch 42 failure_driven
branch 137 fixed
branch 137 failure_driven
warmup
branch 2718 fixed
branch 2718 failure_driven
CURRENT=none
echo "$(date -Iseconds) ALL_DONE" | tee -a "$LOG/queue.log"
