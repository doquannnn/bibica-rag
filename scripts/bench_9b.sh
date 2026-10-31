#!/usr/bin/env bash
# Stage C: benchmark Qwen3.5-9B on the 240 test questions, unattended. Resumable: re-run the same command after an
# interruption and finished questions are skipped (per-question files in results/runs/).
#
#   scripts/bench_9b.sh                 # everything, then packs results/stageC_9B.tar.gz  <- send this file
#   SKIP_CAG=1 scripts/bench_9b.sh      # without the full-context CAG run
#   SKIP_LATENCY=1 scripts/bench_9b.sh  # without the sequential latency runs (30 questions, 1 worker)
#   WORKERS=2 scripts/bench_9b.sh       # fewer concurrent requests (default 4 = server slots)
#
# Steps:  1. llama-server 9B on :8080 (4 slots x 16k)  -> dspy, naive-best-s2b, naive-best (configs/stageC_9b.yaml)
#         2. sequential latency runs of the same pipelines (no response cache)
#         3. llama-server 9B on :8081 (2 slots x 32k)  -> cag-full-context (+ latency)
#         4. tar runs + latency + logs -> results/stageC_9B.tar.gz
# Needs models/qwen3.5-9B/{model-Q4_K_M.gguf,mmproj-F16.gguf} and the llama.cpp build used by scripts/serve.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SIZE=9B
export LLM_MODEL="qwen3.5-$SIZE"
WORKERS="${WORKERS:-4}"
OUT="$ROOT/results/stageC_9B"
LOG="$OUT/bench.log"
mkdir -p "$OUT"
SERVER_PID=""

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT
trap 'log "interrupted"; exit 130' INT TERM

start_server() {  # port ctx parallel
  local port=$1 ctx=$2 par=$3
  if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    log "ERROR: something is already listening on :$port — stop it first (it may be another model)."; exit 1
  fi
  log "starting llama-server $LLM_MODEL on :$port (ctx $ctx, $par slots)"
  "$ROOT/scripts/serve.sh" "$SIZE" "$port" "$ctx" "$par" >"$OUT/server_$port.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 300); do  # up to 10 min for the model to load
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then log "server ready"; return; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "ERROR: llama-server exited, last lines of $OUT/server_$port.log:"; tail -20 "$OUT/server_$port.log" | tee -a "$LOG"; exit 1
    fi
    sleep 2
  done
  log "ERROR: server not ready after 10 min"; exit 1
}

run() {  # base_url, then run_pipelines args
  local url=$1; shift
  log "run_pipelines $*"
  LLM_BASE_URL="$url" uv run python -m bibica_rag.eval.run_pipelines "$@" 2>&1 | tee -a "$LOG"
}

# ---- preflight -------------------------------------------------------------------------------------------------
M="$ROOT/models/qwen3.5-$SIZE"
for f in "$M/model-Q4_K_M.gguf" "$M/mmproj-F16.gguf" "$ROOT/tools/llama-vulkan-x64/llama-b11714/llama-server"; do
  [[ -f "$f" ]] || { log "ERROR: missing $f"; exit 1; }
done
command -v uv >/dev/null || { log "ERROR: uv not found"; exit 1; }
command -v curl >/dev/null || { log "ERROR: curl not found"; exit 1; }
{
  echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "git: $(git rev-parse HEAD 2>/dev/null || echo '?')"
  echo "model: $LLM_MODEL  workers: $WORKERS"; uname -a
  ls -l "$M"
  (command -v vulkaninfo >/dev/null && vulkaninfo --summary 2>/dev/null | grep -iE "deviceName|driverName|apiVersion") || true
} >"$OUT/env.txt"
T0=$(date +%s)
log "=== Stage C $LLM_MODEL start ==="

# ---- 1-2. main server pipelines --------------------------------------------------------------------------------
start_server 8080 65536 4
run http://127.0.0.1:8080/v1 --config configs/stageC_9b.yaml --split test --workers "$WORKERS"
if [[ -z "${SKIP_LATENCY:-}" ]]; then
  LLM_NO_CACHE_READ=1 run http://127.0.0.1:8080/v1 --config configs/stageC_9b.yaml --split test --latency
fi
stop_server

# ---- 3. CAG on its own server ----------------------------------------------------------------------------------
if [[ -z "${SKIP_CAG:-}" ]]; then
  start_server 8081 65536 2
  run http://127.0.0.1:8081/v1 --config configs/stageB_cag.yaml --split test --workers 2
  if [[ -z "${SKIP_LATENCY:-}" ]]; then
    LLM_NO_CACHE_READ=1 run http://127.0.0.1:8081/v1 --config configs/stageB_cag.yaml --split test --latency
  fi
  stop_server
fi

# ---- 4. summary + package --------------------------------------------------------------------------------------
log "=== done in $(( ($(date +%s) - T0) / 60 )) min ==="
for d in results/runs/*@"$LLM_MODEL" results/latency/*@"$LLM_MODEL"; do
  [[ -d "$d" ]] && log "$(printf '%-50s %4d answers' "$d" "$(ls "$d" | grep -vc '^_')")"
done
ERRS=$(grep -c '^ERROR' "$LOG" || true)
[[ "$ERRS" != "0" ]] && log "WARNING: $ERRS question errors in the log — re-run this script to retry them."
TAR="$ROOT/results/stageC_9B.tar.gz"
tar -czf "$TAR" -C "$ROOT" results/stageC_9B \
  $(cd "$ROOT" && ls -d results/runs/*@"$LLM_MODEL" results/latency/*@"$LLM_MODEL" 2>/dev/null)
log "packed $TAR ($(du -h "$TAR" | cut -f1)) — send this file"
