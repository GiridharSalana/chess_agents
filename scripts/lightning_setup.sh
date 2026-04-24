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

echo "==> python env (plain venv + pip; avoids uv/conda interactions on Studios)"
# Pick the best available python; prefer 3.11 if present.
PYBIN="$(command -v python3.11 || command -v python3.10 || command -v python3.12 || command -v python3)"
echo "  using base python: ${PYBIN}"
"${PYBIN}" --version

# Recreate the venv cleanly.
rm -rf .venv
"${PYBIN}" -m venv .venv
VENV_PY="${DEST}/.venv/bin/python"
"${VENV_PY}" -m pip install --quiet --upgrade pip wheel

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "  GPU detected → installing CUDA torch (cu121)"
  "${VENV_PY}" -m pip install --quiet \
      --index-url https://download.pytorch.org/whl/cu121 \
      torch
else
  echo "  No GPU → installing CPU torch"
  "${VENV_PY}" -m pip install --quiet \
      --index-url https://download.pytorch.org/whl/cpu \
      torch
fi
"${VENV_PY}" -m pip install --quiet python-chess numpy

echo "  installed packages:"
"${VENV_PY}" -c "import torch, chess, numpy; print(' torch=', torch.__version__, 'cuda=', torch.cuda.is_available()); print(' chess=', chess.__version__); print(' numpy=', numpy.__version__)"

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
