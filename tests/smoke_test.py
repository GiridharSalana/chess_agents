"""End-to-end smoke test: tiny model + few MCTS sims + 1 self-play game."""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import chess
from model import ChessNet, board_to_tensor, POLICY_SIZE
from mcts import MCTS

torch.set_num_threads(1)
device = torch.device("cpu")

print("instantiate small model...")
model = ChessNet(num_res_blocks=2, channels=32).to(device)
model.eval()
print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

print("MCTS search at start position (10 sims)...")
mcts = MCTS(model, device, num_simulations=10)
t0 = time.time()
move, target = mcts.search(chess.Board(), temperature=1.0)
print(f"  chose {move.uci()} in {time.time()-t0:.2f}s, target sums to {target.sum():.4f}")
assert abs(target.sum() - 1.0) < 1e-4
assert target.shape == (POLICY_SIZE,)

print("play one tiny self-play game (5 moves, 5 sims)...")
mcts = MCTS(model, device, num_simulations=5)
board = chess.Board()
moves_played = 0
t0 = time.time()
while not board.is_game_over() and moves_played < 5:
    mv, _ = mcts.search(board, temperature=1.0)
    if mv is None:
        break
    board.push(mv)
    moves_played += 1
print(f"  played {moves_played} moves in {time.time()-t0:.2f}s")

print("forward batch...")
x = torch.stack([board_to_tensor(chess.Board()) for _ in range(8)])
p, v = model(x)
assert p.shape == (8, POLICY_SIZE)
assert v.shape == (8, 1)

print("instantiate full-size model used in training...")
big = ChessNet(num_res_blocks=6, channels=96).to(device)
print(f"  params: {sum(p.numel() for p in big.parameters()):,}")
print("  one forward pass...")
t0 = time.time()
big(torch.stack([board_to_tensor(chess.Board())]))
print(f"  took {(time.time()-t0)*1000:.1f}ms")

print("ALL SMOKE TESTS PASSED.")
