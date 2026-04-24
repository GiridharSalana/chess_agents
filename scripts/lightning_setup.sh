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

echo "==> clone or update repo"
if [ -d "${DEST}/.git" ]; then
  git -C "${DEST}" pull --ff-only
else
  git clone "${REPO_URL}" "${DEST}"
fi
cd "${DEST}"

echo "==> python env"
# Lightning Studios enforce a single conda env (no venvs allowed). Install
# directly into the active python. On a normal box we just use the system
# python too — keeps the script portable.
PYBIN="$(command -v python3)"
echo "  using python: ${PYBIN}"
"${PYBIN}" --version

# Check if torch is already installed (Lightning Studios usually have it).
if "${PYBIN}" -c "import torch" 2>/dev/null; then
  echo "  torch already present:"
  "${PYBIN}" -c "import torch; print('  torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"
else
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "  GPU detected → installing CUDA torch (cu121)"
    "${PYBIN}" -m pip install --quiet --index-url https://download.pytorch.org/whl/cu121 torch
  else
    echo "  No GPU → installing CPU torch"
    "${PYBIN}" -m pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch
  fi
fi

# python-chess + numpy (numpy is virtually always already there).
"${PYBIN}" -c "import chess" 2>/dev/null || "${PYBIN}" -m pip install --quiet python-chess
"${PYBIN}" -c "import numpy" 2>/dev/null || "${PYBIN}" -m pip install --quiet numpy

echo "  installed packages:"
"${PYBIN}" -c "import torch, chess, numpy; print(' torch=', torch.__version__, 'cuda=', torch.cuda.is_available()); print(' chess=', chess.__version__); print(' numpy=', numpy.__version__)"

echo "==> sanity tests"
"${PYBIN}" tests/test_move_encoding.py
"${PYBIN}" tests/smoke_test.py

echo "==> launching training (auto profile, detached)"
mkdir -p saved_models
nohup setsid "${PYBIN}" -u train.py > nohup_train.out 2>&1 < /dev/null &
disown
sleep 3

echo
echo "Training PID(s):"
pgrep -af "python.*train.py" | grep -v pgrep || true
echo
echo "Tail the log with:  tail -f ${DEST}/nohup_train.out"
echo "Live training log:  tail -f ${DEST}/train.log"
