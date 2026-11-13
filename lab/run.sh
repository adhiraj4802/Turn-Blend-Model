#!/usr/bin/env bash
# Creates the Python environment on first use, then starts the demo.
#
#   ./run.sh              all models on the CPU (default)
#   ./run.sh gpu          also installs CUDA + TensorRT (NVIDIA GPU; several GB)
#   PORT=8001 ./run.sh    serve on another port (default 8000)
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-cpu}"
case "$MODE" in
  cpu) REQ=requirements.txt ;;
  gpu) REQ=requirements-gpu.txt ;;
  *) echo "usage: ./run.sh [cpu|gpu]" >&2; exit 2 ;;
esac

if [ ! -x .venv/bin/python ]; then
  echo "Creating Python environment in .venv ..."
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
fi

# Install once per mode. An environment already set up for gpu also serves plain ./run.sh.
if [ ! -f ".venv/.installed-$MODE" ] && ! { [ "$MODE" = cpu ] && [ -f .venv/.installed-gpu ]; }; then
  echo "Installing packages from $REQ ..."
  # onnxruntime and onnxruntime-gpu cannot be installed side by side
  [ "$MODE" = gpu ] && .venv/bin/pip uninstall -q -y onnxruntime 2>/dev/null || true
  .venv/bin/pip install -q -r "$REQ"
  touch ".venv/.installed-$MODE"
fi

exec .venv/bin/python -u server.py
