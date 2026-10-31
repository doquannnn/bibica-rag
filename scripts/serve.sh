#!/usr/bin/env bash
# Start llama-server (Vulkan) for a Qwen3.5 model.  usage: scripts/serve.sh 4B|9B [port] [ctx] [parallel]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SIZE="${1:-4B}"; PORT="${2:-8080}"; CTX="${3:-65536}"; PAR="${4:-4}"
M="$ROOT/models/qwen3.5-$SIZE"
exec "$ROOT/tools/llama-vulkan-x64/llama-b11714/llama-server" \
  -m "$M/model-Q4_K_M.gguf" --mmproj "$M/mmproj-F16.gguf" \
  -ngl 99 -c "$CTX" --parallel "$PAR" --cont-batching --cache-reuse 256 \
  --jinja --host 127.0.0.1 --port "$PORT" --alias "qwen3.5-$SIZE" --metrics
