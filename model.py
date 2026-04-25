"""Neural network for chess: policy head (move probabilities) + value head (position evaluation)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess

NUM_SQUARES = 64
# 73 possible "move planes" in AlphaZero-style encoding:
#   56 queen moves (8 directions x 7 distances)
#   8 knight moves
#   9 underpromotions (3 piece types x 3 directions)
NUM_MOVE_PLANES = 73
POLICY_SIZE = NUM_SQUARES * NUM_MOVE_PLANES  # 4672


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Encode a chess board into a (19, 8, 8) float tensor.

    Planes 0-11:  piece presence (P,N,B,R,Q,K) x (white,black)
    Plane 12:     castling rights (white kingside)
    Plane 13:     castling rights (white queenside)
    Plane 14:     castling rights (black kingside)
    Plane 15:     castling rights (black queenside)
    Plane 16:     en-passant square
    Plane 17:     side to move (all 1s if white, all 0s if black)
    Plane 18:     move count (normalized)
    """
    planes = np.zeros((19, 8, 8), dtype=np.float32)

    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
    }

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        row, col = divmod(sq, 8)
        color_offset = 0 if piece.color == chess.WHITE else 6
        planes[piece_map[piece.piece_type] + color_offset, row, col] = 1.0

    planes[12, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[13, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[14, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[15, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))

    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        planes[16, row, col] = 1.0

    planes[17, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    planes[18, :, :] = min(board.fullmove_number / 100.0, 1.0)

    return torch.from_numpy(planes)


def _bb_to_plane(bb: int, out_plane: np.ndarray) -> None:
    """Convert a 64-bit chess bitboard into an (8,8) float32 plane.

    python-chess square 0 == a1, square 63 == h8. Bit i is set iff that
    square is occupied. We unpack the bits into a row-major (rank, file)
    grid where row 0 == rank 1. This is consistent with `board_to_tensor`,
    which uses divmod(sq, 8).
    """
    if bb == 0:
        return
    arr = np.frombuffer(bb.to_bytes(8, "little"), dtype=np.uint8)
    out_plane[:] = np.unpackbits(arr, bitorder="little").reshape(8, 8).astype(np.float32)


def fill_board_planes(board: chess.Board, out: np.ndarray) -> None:
    """Bitboard-vectorized encoder: 12 `to_bytes`+`unpackbits` calls
    instead of 64 python-chess `piece_at` lookups per board. ~10-20x
    faster on hot MCTS paths."""
    out.fill(0.0)
    occ_w = board.occupied_co[chess.WHITE]
    occ_b = board.occupied_co[chess.BLACK]
    # piece_map order matches board_to_tensor: P, N, B, R, Q, K
    piece_bbs = (
        board.pawns, board.knights, board.bishops,
        board.rooks, board.queens, board.kings,
    )
    for idx, bb in enumerate(piece_bbs):
        _bb_to_plane(bb & occ_w, out[idx])
        _bb_to_plane(bb & occ_b, out[idx + 6])
    if board.has_kingside_castling_rights(chess.WHITE):
        out[12, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        out[13, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        out[14, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        out[15, :, :] = 1.0
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        out[16, row, col] = 1.0
    if board.turn == chess.WHITE:
        out[17, :, :] = 1.0
    out[18, :, :] = min(board.fullmove_number / 100.0, 1.0)


def boards_to_batch_tensor(boards) -> torch.Tensor:
    """Encode a list of boards into a single (N, 19, 8, 8) cpu tensor with
    one allocation. Preferred when N is large (batched MCTS / training)."""
    n = len(boards)
    arr = np.zeros((n, 19, 8, 8), dtype=np.float32)
    for i, b in enumerate(boards):
        fill_board_planes(b, arr[i])
    return torch.from_numpy(arr)


_PROMO_MAP = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
_KNIGHT_DELTAS = {
    (-2, -1): 0, (-2, 1): 1, (-1, -2): 2, (-1, 2): 3,
    (1, -2): 4, (1, 2): 5, (2, -1): 6, (2, 1): 7,
}
_DIRECTION_MAP = {
    (1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
    (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7,
}

# Cache by (from_sq, to_sq, promotion). At most ~2k distinct chess moves
# exist; the cache fills in the first iter and serves every later call as
# a single dict lookup.
_MOVE_INDEX_CACHE: dict = {}


def _compute_move_index(from_sq: int, to_sq: int, promotion) -> int:
    dr = (to_sq // 8) - (from_sq // 8)
    dc = (to_sq % 8) - (from_sq % 8)

    if promotion and promotion != chess.QUEEN:
        direction = 1 + (1 if dc > 0 else (-1 if dc < 0 else 0))
        plane = 64 + _PROMO_MAP[promotion] * 3 + direction
    else:
        kn = _KNIGHT_DELTAS.get((dr, dc))
        if kn is not None:
            plane = 56 + kn
        else:
            norm_dr = 1 if dr > 0 else (-1 if dr < 0 else 0)
            norm_dc = 1 if dc > 0 else (-1 if dc < 0 else 0)
            dist = max(abs(dr), abs(dc))
            direction_idx = _DIRECTION_MAP.get((norm_dr, norm_dc), 0)
            plane = direction_idx * 7 + (dist - 1)
    return from_sq * NUM_MOVE_PLANES + plane


def move_to_index(move: chess.Move) -> int:
    """Map a chess.Move to a flat policy index in [0, 4672). Cached."""
    key = (move.from_square, move.to_square, move.promotion)
    idx = _MOVE_INDEX_CACHE.get(key)
    if idx is None:
        idx = _compute_move_index(*key)
        _MOVE_INDEX_CACHE[key] = idx
    return idx


def index_to_move(index: int, board: chess.Board) -> chess.Move | None:
    """Try to reverse-map a policy index to a legal move (returns None if invalid)."""
    legal_indices = {move_to_index(m): m for m in board.legal_moves}
    return legal_indices.get(index)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class ChessNet(nn.Module):
    """Dual-headed network: policy (move probabilities) + value (win prediction)."""

    def __init__(self, num_res_blocks: int = 12, channels: int = 160):
        super().__init__()
        self.conv_in = nn.Conv2d(19, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 64, POLICY_SIZE)

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v
