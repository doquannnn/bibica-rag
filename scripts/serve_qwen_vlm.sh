#!/usr/bin/env bash
# Start the local Qwen VLM with llama.cpp's OpenAI-compatible server (shared by PO extraction and RAG chat).
# The model needs its vision projector (mmproj) to read PDF page images.
#
#   MODEL=models/<qwen-vlm>.gguf MMPROJ=models/<mmproj>.gguf scripts/serve_qwen_vlm.sh
#
# Defaults pick the first matching files in models/. Context must hold the page images
# (~1-2k tokens per page at 150 dpi) + the prompt + the JSON answer.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-$(ls models/*.gguf 2>/dev/null | grep -iv mmproj | head -1)}
MMPROJ=${MMPROJ:-$(ls models/*mmproj*.gguf 2>/dev/null | head -1)}
SERVER=${LLAMA_SERVER:-$(ls tools/*/llama-server tools/llama-server 2>/dev/null | head -1 || true)}
SERVER=${SERVER:-llama-server}
[ -n "$MODEL" ] || { echo "No model .gguf in models/ (set MODEL=...)"; exit 1; }
[ -n "$MMPROJ" ] || { echo "No mmproj .gguf in models/ (set MMPROJ=...) — needed for images"; exit 1; }
exec "$SERVER" -m "$MODEL" --mmproj "$MMPROJ" \
  --host 127.0.0.1 --port "${PORT:-8080}" \
  -c "${CTX:-16384}" -ngl "${NGL:-99}" --jinja "$@"
