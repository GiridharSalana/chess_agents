"""Monte Carlo Tree Search guided by a neural network (AlphaZero-style).

Designed for CPU: small per-node footprint, legal-move masking on logits
(no global softmax over the 4672-dim policy), terminal nodes flagged once,
and child boards built without copying the move stack.
"""

import math
import chess
import torch
import numpy as np

from model import board_to_tensor, move_to_index, POLICY_SIZE


C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_WEIGHT = 0.25


class MCTSNode:
    __slots__ = (
        "board", "parent", "move", "children",
        "visit_count", "value_sum", "prior",
        "is_expanded", "is_terminal", "terminal_value",
    )

    def __init__(self, board: chess.Board, parent=None, move=None, prior: float = 0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.children: list["MCTSNode"] = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value = 0.0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb(self, parent_visits: int) -> float:
        return self.q_value + C_PUCT * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)


def _mark_terminal(node: MCTSNode) -> None:
    """Set terminal flags + value (from POV of side to move at this node)."""
    node.is_expanded = True
    node.is_terminal = True
    res = node.board.result(claim_draw=True)
    if res == "1-0":
        node.terminal_value = 1.0 if node.board.turn == chess.BLACK else -1.0
    elif res == "0-1":
        node.terminal_value = 1.0 if node.board.turn == chess.WHITE else -1.0
    else:
        node.terminal_value = 0.0


class MCTS:
    def __init__(self, model, device: torch.device, num_simulations: int = 50):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations

    @torch.no_grad()
    def _evaluate(self, board: chess.Board):
        """Return (legal_priors: ndarray over legal_moves, value: float, legal_moves: list)."""
        legal = list(board.legal_moves)
        if not legal:
            return None, 0.0, legal

        tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
        logits, value = self.model(tensor)
        legal_idx = np.fromiter((move_to_index(m) for m in legal), dtype=np.int64, count=len(legal))
        legal_logits = logits[0].cpu().numpy()[legal_idx]
        legal_logits -= legal_logits.max()
        probs = np.exp(legal_logits)
        s = probs.sum()
        probs = probs / s if s > 0 else np.full(len(legal), 1.0 / len(legal), dtype=np.float32)
        return probs.astype(np.float32), float(value.item()), legal

    def _expand(self, node: MCTSNode, probs: np.ndarray, legal_moves: list, add_noise: bool = False) -> None:
        if not legal_moves:
            _mark_terminal(node)
            return

        if add_noise:
            noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(legal_moves)).astype(np.float32)
            probs = (1.0 - DIRICHLET_WEIGHT) * probs + DIRICHLET_WEIGHT * noise

        for mv, p in zip(legal_moves, probs):
            child_board = node.board.copy(stack=False)
            child_board.push(mv)
            node.children.append(MCTSNode(child_board, parent=node, move=mv, prior=float(p)))
        node.is_expanded = True

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.is_expanded and not node.is_terminal:
            pv = max(node.visit_count, 1)
            node = max(node.children, key=lambda c: c.ucb(pv))
        return node

    def _backprop(self, node: MCTSNode, value: float) -> None:
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent

    def search(self, board: chess.Board, temperature: float = 1.0,
               add_root_noise: bool = True):
        """Run MCTS from `board`. Returns (chosen_move, policy_target).

        policy_target is the visit-count distribution over the full POLICY_SIZE,
        used as the supervised target for training.
        """
        if board.is_game_over(claim_draw=True):
            return None, np.zeros(POLICY_SIZE, dtype=np.float32)

        root = MCTSNode(board.copy(stack=False))
        probs, _v, legal = self._evaluate(root.board)
        self._expand(root, probs, legal, add_noise=add_root_noise)

        for _ in range(self.num_simulations):
            leaf = self._select(root)

            if leaf.is_terminal:
                self._backprop(leaf, -leaf.terminal_value)
                continue

            if leaf.board.is_game_over(claim_draw=True):
                _mark_terminal(leaf)
                self._backprop(leaf, -leaf.terminal_value)
                continue

            probs, value, legal = self._evaluate(leaf.board)
            self._expand(leaf, probs, legal)
            self._backprop(leaf, -value)

        visits = np.array([c.visit_count for c in root.children], dtype=np.float32)
        moves = [c.move for c in root.children]

        if visits.sum() == 0:
            visits = np.array([c.prior for c in root.children], dtype=np.float32)

        # Training target is always the visit distribution at temp=1
        target_probs = visits / visits.sum()
        policy_target = np.zeros(POLICY_SIZE, dtype=np.float32)
        for mv, tp in zip(moves, target_probs):
            policy_target[move_to_index(mv)] = tp

        if temperature < 0.01:
            chosen_idx = int(np.argmax(visits))
        else:
            visits_temp = visits ** (1.0 / temperature)
            sample_probs = visits_temp / visits_temp.sum()
            chosen_idx = int(np.random.choice(len(moves), p=sample_probs))

        return moves[chosen_idx], policy_target
