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

## Hyperparameters (CPU defaults)

Defined at the top of `train.py`:

- 6 res blocks × 96 channels (~10.6M params)
- 40 MCTS sims for self-play, 80 for evaluation
- 16 games / iter × 80 iterations
- Promotion threshold: 55% win-rate vs current best (eval every 4 iters)

For GPU, increase `MCTS_SIMS_TRAIN`, `GAMES_PER_ITERATION`, `BATCH_SIZE`,
and `NUM_CHANNELS` proportionally.
