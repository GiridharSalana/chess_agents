"""Round-trip test: every legal chess move must encode to a unique policy
index in [0, POLICY_SIZE) and decode back to the same move.

A silent bug here would corrupt every training target.

Run from repo root:
    python -m pytest tests/ -q
or:
    python tests/test_move_encoding.py
"""

import os
import sys
import random
import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import move_to_index, index_to_move, POLICY_SIZE  # noqa: E402


def _walk(board: chess.Board, depth: int, seen_positions: set):
    if depth == 0 or board.is_game_over():
        return
    fen = board.fen()
    if fen in seen_positions:
        return
    seen_positions.add(fen)

    legal = list(board.legal_moves)
    indices = set()
    for move in legal:
        idx = move_to_index(move)
        assert 0 <= idx < POLICY_SIZE, f"index {idx} out of range for {move.uci()} in {fen}"
        assert idx not in indices, (
            f"collision at index {idx}: two legal moves map to the same plane "
            f"in position {fen}"
        )
        indices.add(idx)

        decoded = index_to_move(idx, board)
        assert decoded == move, (
            f"round-trip failed: {move.uci()} -> {idx} -> {decoded} "
            f"in position {fen}"
        )

    # Recurse on a random subset to keep runtime bounded.
    sample = random.sample(legal, min(3, len(legal)))
    for move in sample:
        board.push(move)
        _walk(board, depth - 1, seen_positions)
        board.pop()


def test_round_trip_from_initial_position():
    random.seed(0)
    board = chess.Board()
    seen: set = set()
    _walk(board, depth=4, seen_positions=seen)
    assert len(seen) >= 30, f"only walked {len(seen)} positions"


def test_round_trip_promotion_positions():
    fens = [
        # white pawn ready to promote on a/h-files (forces non-queen promotions
        # via knight/bishop/rook to be exercised explicitly)
        "8/P7/8/8/8/8/8/4K2k w - - 0 1",
        "8/7P/8/8/8/8/8/4K2k w - - 0 1",
        # black pawn promoting
        "K7/8/8/8/8/8/p7/4k3 b - - 0 1",
        # capture-promotion on white side
        "1n6/P7/8/8/8/8/8/4K2k w - - 0 1",
    ]
    for fen in fens:
        board = chess.Board(fen)
        for move in board.legal_moves:
            idx = move_to_index(move)
            assert 0 <= idx < POLICY_SIZE
            decoded = index_to_move(idx, board)
            assert decoded == move, f"promotion round-trip failed: {move.uci()} in {fen}"


def test_round_trip_castling_positions():
    fens = [
        # both sides can castle either way
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
    ]
    for fen in fens:
        board = chess.Board(fen)
        for move in board.legal_moves:
            idx = move_to_index(move)
            decoded = index_to_move(idx, board)
            assert decoded == move, f"castling round-trip failed: {move.uci()} in {fen}"


if __name__ == "__main__":
    test_round_trip_from_initial_position()
    test_round_trip_promotion_positions()
    test_round_trip_castling_positions()
    print("all move-encoding round-trip tests passed.")
