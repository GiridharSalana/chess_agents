"""Play against the saved best model in the terminal."""

import os
import chess
import torch
from model import ChessNet
from mcts import MCTS

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pt")


def print_board(board: chess.Board):
    print()
    print("  a b c d e f g h")
    print("  +-+-+-+-+-+-+-+-+")
    for rank in range(7, -1, -1):
        row = f"{rank+1}|"
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            row += (piece.symbol() if piece else ".") + "|"
        print(row + f" {rank+1}")
    print("  +-+-+-+-+-+-+-+-+")
    print("  a b c d e f g h")
    print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet(num_res_blocks=6, channels=96).to(device)

    if not os.path.exists(BEST_MODEL_PATH):
        print(f"No saved model found at {BEST_MODEL_PATH}")
        print("Run train.py first to train a model.")
        return

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully!")

    mcts = MCTS(model, device, num_simulations=400)

    board = chess.Board()
    human_color = input("Play as (w)hite or (b)lack? ").strip().lower()
    human_is_white = human_color != "b"

    print(f"\nYou are {'White' if human_is_white else 'Black'}.")
    print("Enter moves in UCI format (e.g., e2e4, g1f3)")
    print("Type 'quit' to exit.\n")

    while not board.is_game_over():
        print_board(board)
        turn_name = "White" if board.turn == chess.WHITE else "Black"

        if (board.turn == chess.WHITE) == human_is_white:
            print(f"Your turn ({turn_name}).")
            while True:
                user_input = input("Your move: ").strip()
                if user_input.lower() == "quit":
                    print("Goodbye!")
                    return
                try:
                    move = chess.Move.from_uci(user_input)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    print(f"Illegal move. Legal moves: {', '.join(m.uci() for m in board.legal_moves)}")
                except ValueError:
                    print("Invalid format. Use UCI notation (e.g., e2e4)")
        else:
            print(f"AI thinking ({turn_name})...")
            move, _ = mcts.search(board, temperature=0.0, add_root_noise=False)
            print(f"AI plays: {move.uci()}")
            board.push(move)

    print_board(board)
    result = board.result()
    print(f"Game over: {result}")
    if result == "1-0":
        print("White wins!" if human_is_white else "Black wins! (AI)")
    elif result == "0-1":
        print("Black wins!" if not human_is_white else "White wins! (AI)")
    else:
        print("Draw!")


if __name__ == "__main__":
    main()
