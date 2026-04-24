#!/usr/bin/env bash
# Bootstrap chess_agents on a Lightning AI Studio.
# Usage (from a Studio terminal):
#   curl -fsSL https://raw.githubusercontent.com/GiridharSalana/chess_agents/main/scripts/lightning_setup.sh | bash
# or:
#   bash lightning_setup.sh

set -euo pipefail

REPO_URL="https://github.com/GiridharSalana/chess_agents.git"
DEST="${HOME}/chess_agents"

echo "==> system info"
uname -a || true
nvidia-smi -L 2>/dev/null || echo "(no GPU detected — will use CPU profile)"
nproc
free -h | head -2

echo "==> ensuring uv is installed"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
uv --version

echo "==> clone or update repo"
if [ -d "${DEST}/.git" ]; then
  git -C "${DEST}" pull --ff-only
else
  git clone "${REPO_URL}" "${DEST}"
fi
cd "${DEST}"

echo "==> python env"
# Install with CUDA torch wheel if a GPU is present, otherwise the cpu wheel.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "  GPU detected → installing CUDA torch"
  uv venv --python 3.11
  uv pip install --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
      torch
  uv pip install python-chess numpy
else
  echo "  No GPU → installing CPU torch"
  uv sync
fi

echo "==> sanity tests"
.venv/bin/python tests/test_move_encoding.py
.venv/bin/python tests/smoke_test.py

echo "==> launching training (auto profile, detached)"
mkdir -p saved_models
nohup setsid .venv/bin/python -u train.py > nohup_train.out 2>&1 < /dev/null &
disown
sleep 3

echo
echo "Training PID(s):"
pgrep -af "python.*train.py" | grep -v pgrep || true
echo
echo "Tail the log with:  tail -f ${DEST}/nohup_train.out"
echo "Live training log:  tail -f ${DEST}/train.log"
