#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/code/SeedMitigating}"
cd "$ROOT"

SESSION="${SESSION:-eval7}"
MONITOR_SESSION="${MONITOR_SESSION:-eval7_switch}"
MASTER_LOG="${MASTER_LOG:-$ROOT/output/eval_flashrag500_7models/eval7_master.log}"
TARGET_MODEL="${TARGET_MODEL:-7h80zdqz}"
TARGET_LAST_DATASET="${TARGET_LAST_DATASET:-flashrag_bamboogle}"
POLL_SECONDS="${POLL_SECONDS:-30}"

NEW_VLLM_MAX_NUM_BATCHED_TOKENS="${NEW_VLLM_MAX_NUM_BATCHED_TOKENS:-1536}"
NEW_VLLM_MAX_NUM_SEQS="${NEW_VLLM_MAX_NUM_SEQS:-48}"

DONE_MARKER="[DONE] model=${TARGET_MODEL} dataset=${TARGET_LAST_DATASET}"

echo "[$(date '+%F %T')] [SWITCH] waiting marker: ${DONE_MARKER}" | tee -a "$MASTER_LOG"
# Only watch newly appended lines after monitor starts.
start_line=0
if [ -f "$MASTER_LOG" ]; then
  start_line="$(wc -l < "$MASTER_LOG")"
fi
while true; do
  if [ -f "$MASTER_LOG" ] && tail -n +"$((start_line + 1))" "$MASTER_LOG" | rg -Fq "$DONE_MARKER"; then
    break
  fi
  sleep "$POLL_SECONDS"
done

echo "[$(date '+%F %T')] [SWITCH] marker observed, restarting ${SESSION} with tuned VLLM caps" | tee -a "$MASTER_LOG"
tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 2

tmux new -d -s "$SESSION" "cd $ROOT && \
VLLM_MAX_NUM_BATCHED_TOKENS=$NEW_VLLM_MAX_NUM_BATCHED_TOKENS \
VLLM_MAX_NUM_SEQS=$NEW_VLLM_MAX_NUM_SEQS \
bash rl/scripts/eval_flashrag500_7models.sh > output/eval_flashrag500_7models/eval7_master.log 2>&1; \
code=\$?; \
echo \"[EXIT_CODE]=\$code\" >> output/eval_flashrag500_7models/eval7_master.log; \
tail -n 80 output/eval_flashrag500_7models/eval7_master.log; \
exec bash"

echo "[$(date '+%F %T')] [SWITCH] restarted ${SESSION}: VLLM_MAX_NUM_BATCHED_TOKENS=$NEW_VLLM_MAX_NUM_BATCHED_TOKENS VLLM_MAX_NUM_SEQS=$NEW_VLLM_MAX_NUM_SEQS" | tee -a "$MASTER_LOG"
