#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP="$ROOT/packages/desktop"

for cmd in node npm cargo; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: $cmd is required but not installed." >&2
    exit 1
  fi
done

cd "$DESKTOP"

if [[ ! -d node_modules ]]; then
  echo "Installing npm dependencies..."
  npm install
fi

if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Warning: Ollama is not running at 127.0.0.1:11434."
  echo "  Search and Ask need Ollama with nomic-embed-text and granite4.1:3b."
fi

export RUST_LOG="${RUST_LOG:-info,mailfind=debug}"
exec npm run tauri:dev
