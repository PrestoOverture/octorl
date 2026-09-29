#!/bin/bash
# Wake Claude on: a newly completed stage/warm-up, queue exit, infra failure, exception, tmux gone, or 2 h silence.
probe() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 autodl-r4 '
    R=/root/autodl-tmp/octorl_r4r
    echo "STAGES $(find $R -path "*stage_*/attributed.jsonl" 2>/dev/null | wc -l) WARM $(ls $R/seed_2718/warmup/attributed.jsonl 2>/dev/null | wc -l) INFRA $(find $R -name INFRA_FAILED 2>/dev/null | wc -l)"
    grep -c "QUEUE_EXIT" $R/logs/queue.log
    grep -c -E "Traceback|Error|REFUSE|stop" $R/logs/queue.log
    tmux has-session -t r4r_queue 2>/dev/null && echo TMUX_UP || echo TMUX_GONE' 2>/dev/null
}
base=$(probe | head -1)
echo "baseline: $base  $(date)"
for i in $(seq 1 24); do
  sleep 300
  out=$(probe); [ -z "$out" ] && { echo "ssh probe failed $(date)"; continue; }
  now=$(echo "$out" | sed -n 1p); qexit=$(echo "$out" | sed -n 2p); err=$(echo "$out" | sed -n 3p); tm=$(echo "$out" | sed -n 4p)
  if [ "$now" != "$base" ] || [ "$qexit" != "0" ] || [ "$err" != "0" ] || [ "$tm" = "TMUX_GONE" ]; then
    echo "EVENT: $now qexit=$qexit err=$err $tm  $(date)"; exit 0
  fi
done
echo "TIMEOUT: no change in 2 h: $now  $(date)"
