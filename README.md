# chess_agents

AlphaZero-style chess RL agent trained via self-play (CPU-friendly).

Two networks share weights conceptually: a **candidate** trains on data from the
current **best**; promotion is gated by a head-to-head match.

## Components

| File | Purpose |
| --- | --- |
| `model.py` | Residual policy/value network + AlphaZero move encoding (4672-dim). |
| `mcts.py` | NN-guided MCTS with terminal handling and Dirichlet root noise. |
| `train.py` | Parallel self-play, sliding-window replay buffer, gated promotion. |
| `play.py` | Play against the saved best model in the terminal (UCI input). |
| `tests/test_move_encoding.py` | Round-trip move ↔ index check (catches silent corruption). |
| `tests/smoke_test.py` | End-to-end wiring sanity check. |

## Quick start

```bash
uv sync                          # install deps into .venv
.venv/bin/python tests/test_move_encoding.py
.venv/bin/python tests/smoke_test.py
.venv/bin/python train.py        # foreground
```

To run unattended (survives SSH disconnect):

```bash
nohup setsid .venv/bin/python -u train.py > nohup_train.out 2>&1 < /dev/null &
disown
```

Resume is automatic from `saved_models/checkpoint.pt`.

## Profiles (CPU vs GPU)

`train.py` auto-detects CUDA and switches profiles. Override with
`CHESS_PROFILE=cpu` or `CHESS_PROFILE=gpu`.

| Setting               | CPU profile    | GPU profile    |
|-----------------------|----------------|----------------|
| res blocks × channels | 6 × 96         | 10 × 128       |
| self-play workers     | min(8, n-1)    | min(16, n-2)   |
| games / iteration     | 16             | 48             |
| MCTS sims (train/eval)| 40 / 80        | 80 / 160       |
| batch size            | 256            | 512            |
| iterations            | 80             | 200            |
| replay buffer         | 100k           | 250k           |

Promotion gate: candidate must hit 55% win-rate (incl. half-points for draws)
against current best across `EVAL_GAMES` MCTS games starting from a small set
of common openings.

## Run on Lightning AI

In a Studio terminal:

```bash
curl -fsSL https://raw.githubusercontent.com/GiridharSalana/chess_agents/main/scripts/lightning_setup.sh | bash
```

This installs `uv`, sets up a venv with the right (CUDA or CPU) `torch` wheel,
runs the round-trip + smoke tests, and launches `train.py` under `nohup setsid`
so it survives terminal disconnect. Logs go to `nohup_train.out` and `train.log`.
