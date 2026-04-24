"""AlphaZero-style self-play training (CPU-only).

Each iteration:
  1) Parallel self-play with MCTS-guided move selection.
  2) Train on a sliding-window replay buffer.
  3) Every EVAL_EVERY iterations, gated evaluation vs current `best`.
     Promote candidate -> best only if win-rate >= WIN_THRESHOLD.

Resume-safe: latest weights/optimizer go to checkpoint.pt; best gates
strictly into best_model.pt.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import time
import math
import chess
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
from collections import deque

from model import ChessNet, board_to_tensor, POLICY_SIZE
from mcts import MCTS

mp.set_sharing_strategy("file_system")


# ---------- paths ----------
ROOT = os.path.dirname(__file__)
SAVE_DIR = os.path.join(ROOT, "saved_models")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pt")
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint.pt")
LOG_PATH = os.path.join(ROOT, "train.log")

# ---------- profile (cpu | gpu | auto) ----------
# Auto-detection: GPU profile if CUDA is present, else CPU.
PROFILE = os.environ.get("CHESS_PROFILE", "auto").lower()
if PROFILE == "auto":
    PROFILE = "gpu" if (os.environ.get("CHESS_FORCE_GPU") == "1") else None  # decide after torch import

import torch as _torch  # noqa: E402
if PROFILE is None:
    PROFILE = "gpu" if _torch.cuda.is_available() else "cpu"

CPU_COUNT = os.cpu_count() or 4

if PROFILE == "gpu":
    # Single-process batched self-play that keeps the GPU saturated.
    # CONCURRENT_GAMES = how many games are played in lockstep so we can batch
    # leaf evaluations across them on every MCTS step.
    NUM_RES_BLOCKS = 10
    NUM_CHANNELS = 128
    NUM_WORKERS = 0  # unused on GPU path
    NUM_ITERATIONS = 200
    CONCURRENT_GAMES = 128         # all played in lockstep, batched on GPU
    LEAVES_PER_STEP = 8            # leaves per game per inner step (virtual loss)
    GAMES_PER_ITERATION = 128      # one concurrent batch per iteration
    EVAL_EVERY = 4
    EVAL_GAMES = 24
    WIN_THRESHOLD = 0.55
    MCTS_SIMS_TRAIN = 64
    MCTS_SIMS_EVAL = 120
    MAX_GAME_MOVES = 200
    RESIGN_VALUE = -0.92
    RESIGN_STREAK = 5
    BATCH_SIZE = 1024
    EPOCHS_PER_ITERATION = 6
    REPLAY_BUFFER_SIZE = 300_000
    LR_INIT = 1e-3
    LR_MIN = 1e-5
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
else:  # cpu
    CONCURRENT_GAMES = 0  # unused
    NUM_RES_BLOCKS = 6
    NUM_CHANNELS = 96
    NUM_WORKERS = max(1, min(8, CPU_COUNT - 1))
    NUM_ITERATIONS = 80
    GAMES_PER_ITERATION = 16
    EVAL_EVERY = 4
    EVAL_GAMES = 14
    WIN_THRESHOLD = 0.55
    MCTS_SIMS_TRAIN = 40
    MCTS_SIMS_EVAL = 80
    MAX_GAME_MOVES = 220
    RESIGN_VALUE = -0.92
    RESIGN_STREAK = 5
    BATCH_SIZE = 256
    EPOCHS_PER_ITERATION = 4
    REPLAY_BUFFER_SIZE = 100_000
    LR_INIT = 1e-3
    LR_MIN = 1e-5
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0


# A handful of common opening plies for evaluation diversity.
EVAL_OPENINGS = [
    [],
    ["e2e4"],
    ["d2d4"],
    ["c2c4"],
    ["g1f3"],
    ["e2e4", "c7c5"],
    ["e2e4", "e7e5"],
]


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def cosine_lr(iteration: int, total: int) -> float:
    return LR_MIN + 0.5 * (LR_INIT - LR_MIN) * (1 + math.cos(math.pi * iteration / total))


def temp_schedule(move: int) -> float:
    """High temp for the first moves (exploration), then near-greedy."""
    if move < 15:
        return 1.0
    if move < 30:
        return 0.5
    return 0.1


# ---------------------------------------------------------------------------
# GPU path: batched single-process self-play / eval
# ---------------------------------------------------------------------------

def gpu_self_play(model, device, total_games: int):
    from selfplay_gpu import selfplay_batch
    all_examples_flat = []
    lengths_all = []
    decisive_total = 0
    remaining = total_games
    while remaining > 0:
        batch = min(CONCURRENT_GAMES, remaining)
        per_game, lengths, decisive = selfplay_batch(
            model, device, n_games=batch,
            sims_per_move=MCTS_SIMS_TRAIN,
            max_moves=MAX_GAME_MOVES,
            temp_schedule_fn=temp_schedule,
            add_root_noise=True,
            leaves_per_step=LEAVES_PER_STEP,
        )
        for g in per_game:
            all_examples_flat.extend(g)
        lengths_all.extend(lengths)
        decisive_total += decisive
        remaining -= batch
    return all_examples_flat, lengths_all, decisive_total


def gpu_evaluate(new_model, old_model, device):
    from selfplay_gpu import evaluate_batch
    win_rate, eval_log = evaluate_batch(
        new_model, old_model, device,
        n_games=EVAL_GAMES,
        sims_per_move=MCTS_SIMS_EVAL,
        max_moves=MAX_GAME_MOVES,
        openings=EVAL_OPENINGS,
        leaves_per_step=LEAVES_PER_STEP,
    )
    for i, (res, new_is_white) in enumerate(eval_log):
        tag = "W" if new_is_white else "B"
        log(f"    eval {i+1}/{EVAL_GAMES}: {res} (new={tag})")
    return win_rate


# ---------------------------------------------------------------------------
# self-play worker
# ---------------------------------------------------------------------------

def _play_one_selfplay_game(args):
    state_dict, _game_idx = args
    torch.set_num_threads(1)

    device = torch.device("cpu")
    model = ChessNet(NUM_RES_BLOCKS, NUM_CHANNELS).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    mcts = MCTS(model, device, num_simulations=MCTS_SIMS_TRAIN)

    board = chess.Board()
    history = []  # (tensor, policy_target, side_to_move)
    move_count = 0
    resign_streak_white = 0
    resign_streak_black = 0
    resigned_winner = None  # chess.WHITE / chess.BLACK / None

    while not board.is_game_over(claim_draw=True) and move_count < MAX_GAME_MOVES:
        side = board.turn
        temp = temp_schedule(move_count)
        move, policy_target = mcts.search(board, temperature=temp, add_root_noise=True)
        if move is None:
            break

        history.append((board_to_tensor(board), policy_target, side))

        # Crude resignation: peek at root's value via a single NN eval (cheap).
        with torch.no_grad():
            tens = board_to_tensor(board).unsqueeze(0)
            _, v = model(tens)
            v = float(v.item())
        if v < RESIGN_VALUE:
            if side == chess.WHITE:
                resign_streak_white += 1
                resign_streak_black = 0
                if resign_streak_white >= RESIGN_STREAK:
                    resigned_winner = chess.BLACK
                    break
            else:
                resign_streak_black += 1
                resign_streak_white = 0
                if resign_streak_black >= RESIGN_STREAK:
                    resigned_winner = chess.WHITE
                    break
        else:
            resign_streak_white = 0
            resign_streak_black = 0

        board.push(move)
        move_count += 1

    if resigned_winner is not None:
        white_reward = 1.0 if resigned_winner == chess.WHITE else -1.0
    else:
        result = board.result(claim_draw=True)
        if result == "1-0":
            white_reward = 1.0
        elif result == "0-1":
            white_reward = -1.0
        else:
            white_reward = 0.0

    examples = []
    for tensor, policy, side in history:
        reward = white_reward if side == chess.WHITE else -white_reward
        examples.append((tensor, policy, reward))

    return examples, move_count, white_reward


def parallel_self_play(model: ChessNet, num_games: int):
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    args = [(state_dict, i) for i in range(num_games)]

    try:
        with mp.Pool(NUM_WORKERS) as pool:
            results = pool.map(_play_one_selfplay_game, args)
    except RuntimeError as exc:
        msg = str(exc)
        if "unable to mmap" in msg or "Cannot allocate memory" in msg:
            log("  WARN: shared-memory pool failed; falling back to sequential self-play.")
            results = [_play_one_selfplay_game(a) for a in args]
        else:
            raise

    all_examples = []
    lengths = []
    decisive = 0
    for examples, move_count, wr in results:
        all_examples.extend(examples)
        lengths.append(move_count)
        if wr != 0.0:
            decisive += 1
    return all_examples, lengths, decisive


# ---------------------------------------------------------------------------
# evaluation worker
# ---------------------------------------------------------------------------

def _play_eval_game(args):
    new_sd, old_sd, game_idx = args
    torch.set_num_threads(1)

    device = torch.device("cpu")
    new_model = ChessNet(NUM_RES_BLOCKS, NUM_CHANNELS).to(device)
    new_model.load_state_dict(new_sd); new_model.eval()
    old_model = ChessNet(NUM_RES_BLOCKS, NUM_CHANNELS).to(device)
    old_model.load_state_dict(old_sd); old_model.eval()

    new_mcts = MCTS(new_model, device, num_simulations=MCTS_SIMS_EVAL)
    old_mcts = MCTS(old_model, device, num_simulations=MCTS_SIMS_EVAL)

    opening = EVAL_OPENINGS[game_idx % len(EVAL_OPENINGS)]
    new_is_white = (game_idx // len(EVAL_OPENINGS)) % 2 == 0

    board = chess.Board()
    for ucistr in opening:
        mv = chess.Move.from_uci(ucistr)
        if mv in board.legal_moves:
            board.push(mv)

    moves = 0
    while not board.is_game_over(claim_draw=True) and moves < MAX_GAME_MOVES:
        is_new_turn = (board.turn == chess.WHITE) == new_is_white
        active = new_mcts if is_new_turn else old_mcts
        mv, _ = active.search(board, temperature=0.0, add_root_noise=False)
        if mv is None:
            break
        board.push(mv)
        moves += 1

    return board.result(claim_draw=True), new_is_white


def parallel_evaluate(new_model: ChessNet, old_model: ChessNet) -> float:
    new_sd = {k: v.cpu() for k, v in new_model.state_dict().items()}
    old_sd = {k: v.cpu() for k, v in old_model.state_dict().items()}
    args = [(new_sd, old_sd, i) for i in range(EVAL_GAMES)]

    try:
        with mp.Pool(NUM_WORKERS) as pool:
            results = pool.map(_play_eval_game, args)
    except RuntimeError as exc:
        msg = str(exc)
        if "unable to mmap" in msg or "Cannot allocate memory" in msg:
            log("  WARN: shared-memory pool failed; falling back to sequential evaluation.")
            results = [_play_eval_game(a) for a in args]
        else:
            raise

    new_wins = 0
    draws = 0
    for i, (result, new_is_white) in enumerate(results):
        tag = "W" if new_is_white else "B"
        log(f"    eval {i+1}/{EVAL_GAMES}: {result} (new={tag})")
        if result == "1-0":
            new_wins += 1 if new_is_white else 0
        elif result == "0-1":
            new_wins += 0 if new_is_white else 1
        else:
            draws += 1
    return (new_wins + 0.5 * draws) / EVAL_GAMES


# ---------------------------------------------------------------------------
# training step
# ---------------------------------------------------------------------------

def train_step(model: ChessNet, optimizer, replay_buffer: deque, device):
    if len(replay_buffer) < BATCH_SIZE:
        return None

    indices = np.random.choice(len(replay_buffer), BATCH_SIZE, replace=False)
    batch = [replay_buffer[i] for i in indices]

    states = torch.stack([b[0] for b in batch]).to(device)
    target_policies = torch.stack([torch.from_numpy(b[1]) for b in batch]).to(device)
    target_values = torch.tensor(
        [b[2] for b in batch], dtype=torch.float32
    ).unsqueeze(1).to(device)

    policy_logits, pred_values = model(states)
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(target_policies * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(pred_values, target_values)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    return float(policy_loss.item()), float(value_loss.item())


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    if PROFILE == "cpu":
        mp.set_start_method("spawn", force=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if (PROFILE == "gpu" and torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = ChessNet(NUM_RES_BLOCKS, NUM_CHANNELS).to(device)
    best_model = ChessNet(NUM_RES_BLOCKS, NUM_CHANNELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)

    start_iter = 1
    total_games = 0

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                log(f"  WARN: optimizer state not restored: {e}")
        start_iter = ckpt.get("iteration", 0) + 1
        total_games = ckpt.get("total_games", 0)
        if "best_model" in ckpt:
            best_model.load_state_dict(ckpt["best_model"])
        elif os.path.exists(BEST_MODEL_PATH):
            best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
        else:
            best_model.load_state_dict(model.state_dict())
        log(f"Resumed from checkpoint iter {start_iter - 1}")
    elif os.path.exists(BEST_MODEL_PATH):
        sd = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
        try:
            model.load_state_dict(sd)
            best_model.load_state_dict(sd)
            log("Resumed from best_model.pt")
        except RuntimeError as e:
            log(f"Could not load existing best_model.pt (architecture changed): {e}")
            log("Starting from random init.")
            best_model.load_state_dict(model.state_dict())
    else:
        best_model.load_state_dict(model.state_dict())

    replay_buffer: deque = deque(maxlen=REPLAY_BUFFER_SIZE)

    param_count = sum(p.numel() for p in model.parameters())
    log("=" * 60)
    log(f"Chess RL Self-Play (AlphaZero-style, profile={PROFILE}, train_device={device.type})")
    if device.type == "cuda":
        log(f"  GPU: {torch.cuda.get_device_name(0)}")
    log(f"  workers={NUM_WORKERS}  iters={NUM_ITERATIONS}  games/iter={GAMES_PER_ITERATION}")
    log(f"  model: {NUM_RES_BLOCKS} res blocks x {NUM_CHANNELS} ch  ({param_count:,} params)")
    log(f"  MCTS: train={MCTS_SIMS_TRAIN} sims  eval={MCTS_SIMS_EVAL} sims")
    log(f"  batch={BATCH_SIZE}  epochs/iter={EPOCHS_PER_ITERATION}")
    log(f"  LR: {LR_INIT} -> {LR_MIN} cosine")
    log("=" * 60)

    training_start = time.time()

    for it in range(start_iter, NUM_ITERATIONS + 1):
        t0 = time.time()
        lr = cosine_lr(it, NUM_ITERATIONS)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        log(f"--- Iter {it}/{NUM_ITERATIONS}  lr={lr:.6f} ---")

        # Self-play with the current `best` (standard AlphaZero recipe):
        # data is generated by the strongest known network so the targets
        # represent strong play.
        best_model.eval()
        sp_start = time.time()
        if PROFILE == "gpu":
            examples, lengths, decisive = gpu_self_play(best_model, device, GAMES_PER_ITERATION)
        else:
            examples, lengths, decisive = parallel_self_play(best_model, GAMES_PER_ITERATION)
        sp_time = time.time() - sp_start
        total_games += GAMES_PER_ITERATION

        log(f"  self-play: {GAMES_PER_ITERATION} games in {sp_time:.1f}s "
            f"(avg {np.mean(lengths):.0f} moves, {decisive}/{GAMES_PER_ITERATION} decisive)")

        replay_buffer.extend(examples)
        log(f"  buffer: {len(replay_buffer)} positions ({len(examples)} new)")

        # Train candidate
        model.train()
        p_losses, v_losses = [], []
        steps_per_epoch = max(1, len(examples) // BATCH_SIZE)
        for _ in range(EPOCHS_PER_ITERATION):
            for _ in range(steps_per_epoch):
                out = train_step(model, optimizer, replay_buffer, device)
                if out is None:
                    continue
                pl, vl = out
                p_losses.append(pl); v_losses.append(vl)

        if p_losses:
            log(f"  train: {len(p_losses)} steps  pol={np.mean(p_losses):.4f}  val={np.mean(v_losses):.4f}")
        else:
            log("  train: skipped (buffer too small)")

        # Gated promotion
        promoted = False
        if it % EVAL_EVERY == 0:
            model.eval()
            log(f"  evaluating candidate vs best ({EVAL_GAMES} games, {MCTS_SIMS_EVAL} sims)...")
            if PROFILE == "gpu":
                wr = gpu_evaluate(model, best_model, device)
            else:
                wr = parallel_evaluate(model, best_model)
            log(f"  win rate: {wr:.1%}  (threshold {WIN_THRESHOLD:.0%})")
            if wr >= WIN_THRESHOLD:
                log("  >>> NEW BEST! promoting candidate.")
                best_model.load_state_dict(model.state_dict())
                torch.save(best_model.state_dict(), BEST_MODEL_PATH)
                promoted = True
            else:
                log("  candidate did not beat best; keeping best, continuing training.")

        # Always persist resume-state, but never silently overwrite best.
        torch.save({
            "model": model.state_dict(),
            "best_model": best_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": it,
            "total_games": total_games,
            "promoted_this_iter": promoted,
        }, CHECKPOINT_PATH)

        elapsed = time.time() - training_start
        eta = elapsed / max(it - start_iter + 1, 1) * (NUM_ITERATIONS - it)
        log(f"  iter time: {time.time()-t0:.1f}s  total: {elapsed/60:.1f}m  "
            f"ETA: {eta/60:.1f}m  games: {total_games}")
        log("")

    # Final flush
    if not os.path.exists(BEST_MODEL_PATH):
        torch.save(best_model.state_dict(), BEST_MODEL_PATH)
    log("=" * 60)
    log(f"DONE. {total_games} games in {(time.time()-training_start)/60:.1f}m")
    log(f"Best model: {BEST_MODEL_PATH}")
    log("=" * 60)


if __name__ == "__main__":
    main()
