"""Batched single-process self-play and evaluation.

Runs N games concurrently in one process. At every MCTS step, leaves from
all active games are gathered into one batch and sent to the GPU as a
single forward pass. This keeps the GPU saturated -- the standard
single-machine alternative to spawning many CPU workers.

Key idea: virtual loss is applied on the path to each batched leaf so
multiple in-flight evaluations don't all converge to the same node within
a single tree (relevant when collecting >1 leaf per game per batch).
"""

from __future__ import annotations

import math
import chess
import torch
import numpy as np

from model import board_to_tensor, move_to_index, POLICY_SIZE


C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_WEIGHT = 0.25


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class Node:
    __slots__ = (
        "board", "parent", "move", "children",
        "visit_count", "value_sum", "prior",
        "is_expanded", "is_terminal", "terminal_value",
        "virtual_loss",
    )

    def __init__(self, board: chess.Board, parent=None, move=None, prior: float = 0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.children: list[Node] = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value = 0.0
        self.virtual_loss = 0

    def q(self) -> float:
        denom = self.visit_count + self.virtual_loss
        if denom == 0:
            return 0.0
        # Virtual loss adds pessimism: pretend in-flight sims will lose.
        return (self.value_sum - self.virtual_loss) / denom

    def ucb(self, parent_visits: int) -> float:
        return self.q() + C_PUCT * self.prior * math.sqrt(parent_visits) / (
            1 + self.visit_count + self.virtual_loss
        )


def _mark_terminal(node: Node) -> None:
    node.is_expanded = True
    node.is_terminal = True
    res = node.board.result(claim_draw=True)
    if res == "1-0":
        node.terminal_value = 1.0 if node.board.turn == chess.BLACK else -1.0
    elif res == "0-1":
        node.terminal_value = 1.0 if node.board.turn == chess.WHITE else -1.0
    else:
        node.terminal_value = 0.0


def _select(root: Node):
    path = [root]
    node = root
    while node.is_expanded and not node.is_terminal:
        pv = max(node.visit_count, 1)
        node = max(node.children, key=lambda c: c.ucb(pv))
        path.append(node)
    return node, path


def _expand(node: Node, priors, legal_moves, add_noise: bool = False) -> None:
    if not legal_moves:
        _mark_terminal(node)
        return
    if add_noise:
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(legal_moves)).astype(np.float32)
        priors = (1.0 - DIRICHLET_WEIGHT) * priors + DIRICHLET_WEIGHT * noise
    for mv, p in zip(legal_moves, priors):
        cb = node.board.copy(stack=False)
        cb.push(mv)
        node.children.append(Node(cb, parent=node, move=mv, prior=float(p)))
    node.is_expanded = True


def _backprop(node: Node, value: float) -> None:
    while node is not None:
        node.visit_count += 1
        node.value_sum += value
        value = -value
        node = node.parent


def _apply_vl(path):
    for n in path:
        n.virtual_loss += 1


def _undo_vl(path):
    for n in path:
        n.virtual_loss -= 1


# ---------------------------------------------------------------------------
# batched NN evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _batch_evaluate(model, device, boards):
    """Return list of (legal_priors_or_None, value, legal_moves) parallel to `boards`."""
    if not boards:
        return []
    tensors = torch.stack([board_to_tensor(b) for b in boards]).to(device, non_blocking=True)
    logits, values = model(tensors)
    logits_np = logits.detach().cpu().numpy()
    values_np = values.detach().cpu().numpy().reshape(-1)

    out = []
    for i, board in enumerate(boards):
        legal = list(board.legal_moves)
        if not legal:
            out.append((None, float(values_np[i]), legal))
            continue
        legal_idx = np.fromiter(
            (move_to_index(m) for m in legal), dtype=np.int64, count=len(legal)
        )
        ll = logits_np[i, legal_idx]
        ll -= ll.max()
        probs = np.exp(ll)
        probs /= probs.sum()
        out.append((probs.astype(np.float32), float(values_np[i]), legal))
    return out


# ---------------------------------------------------------------------------
# game state
# ---------------------------------------------------------------------------

class GameSlot:
    __slots__ = ("board", "history", "move_count", "done", "white_reward",
                 "max_moves", "root")

    def __init__(self, max_moves: int, starting_board: chess.Board | None = None):
        self.board = starting_board if starting_board is not None else chess.Board()
        self.history: list = []  # (tensor, policy_target, side_to_move)
        self.move_count = 0
        self.done = False
        self.white_reward = 0.0
        self.max_moves = max_moves
        self.root: Node | None = None


def _finalize(game: GameSlot) -> None:
    if game.done:
        return
    game.done = True
    res = game.board.result(claim_draw=True)
    if res == "1-0":
        game.white_reward = 1.0
    elif res == "0-1":
        game.white_reward = -1.0
    else:
        game.white_reward = 0.0


def _run_sims_for_group(model, device, game_indices, games, sims_per_move,
                        add_root_noise: bool):
    """Run sims_per_move MCTS sims for each game in `game_indices`, batching
    leaf evaluations across games in one forward pass per inner step.
    Mutates games[i].root in place."""
    if not game_indices:
        return

    # 1. fresh root per game; batched root-eval to seed priors
    for i in game_indices:
        games[i].root = Node(games[i].board.copy(stack=False))
    root_results = _batch_evaluate(model, device, [games[i].board for i in game_indices])
    for i, (probs, _v, legal) in zip(game_indices, root_results):
        _expand(games[i].root, probs, legal, add_noise=add_root_noise)

    # remove games whose root is terminal (no legal moves)
    active = [i for i in game_indices if not games[i].root.is_terminal]

    # 2. simulation loop, one leaf per game per inner step
    sims_done = {i: 0 for i in active}
    while True:
        to_eval = []
        for i in active:
            if sims_done[i] >= sims_per_move:
                continue
            leaf, path = _select(games[i].root)
            if leaf.is_terminal:
                _backprop(leaf, -leaf.terminal_value)
                sims_done[i] += 1
                continue
            if leaf.board.is_game_over(claim_draw=True):
                _mark_terminal(leaf)
                _backprop(leaf, -leaf.terminal_value)
                sims_done[i] += 1
                continue
            _apply_vl(path)
            to_eval.append((i, leaf, path))

        if not to_eval and all(sims_done[i] >= sims_per_move for i in active):
            return
        if not to_eval:
            continue

        results = _batch_evaluate(model, device, [t[1].board for t in to_eval])
        for (i, leaf, path), (probs, value, legal) in zip(to_eval, results):
            _undo_vl(path)
            _expand(leaf, probs, legal)
            _backprop(leaf, -value)
            sims_done[i] += 1


def _pick_move_from_root(root: Node, temperature: float):
    visits = np.array([c.visit_count for c in root.children], dtype=np.float32)
    moves = [c.move for c in root.children]
    if visits.sum() == 0:
        visits = np.array([c.prior for c in root.children], dtype=np.float32)

    target_probs = visits / visits.sum()
    policy_target = np.zeros(POLICY_SIZE, dtype=np.float32)
    for mv, tp in zip(moves, target_probs):
        policy_target[move_to_index(mv)] = tp

    if temperature < 0.01:
        chosen = int(np.argmax(visits))
    else:
        vt = visits ** (1.0 / temperature)
        p = vt / vt.sum()
        chosen = int(np.random.choice(len(moves), p=p))
    return moves[chosen], policy_target


# ---------------------------------------------------------------------------
# self-play
# ---------------------------------------------------------------------------

def selfplay_batch(model, device, n_games: int, sims_per_move: int,
                   max_moves: int, temp_schedule_fn,
                   add_root_noise: bool = True):
    """Play n_games concurrently, batched on `device`. Returns
    (per_game_examples, per_game_lengths, decisive_count)."""
    games = [GameSlot(max_moves) for _ in range(n_games)]

    while any(not g.done for g in games):
        active = [i for i, g in enumerate(games) if not g.done]
        _run_sims_for_group(model, device, active, games, sims_per_move,
                            add_root_noise=add_root_noise)

        for i in active:
            game = games[i]
            if game.done:  # terminal-root case set it
                continue
            temp = temp_schedule_fn(game.move_count)
            move, policy_target = _pick_move_from_root(game.root, temp)
            game.history.append((board_to_tensor(game.board), policy_target,
                                 game.board.turn))
            game.board.push(move)
            game.move_count += 1
            if game.board.is_game_over(claim_draw=True) or game.move_count >= max_moves:
                _finalize(game)
            game.root = None

    all_examples = []
    lengths = []
    decisive = 0
    for game in games:
        examples = []
        for tensor, policy, side in game.history:
            reward = game.white_reward if side == chess.WHITE else -game.white_reward
            examples.append((tensor, policy, reward))
        all_examples.append(examples)
        lengths.append(game.move_count)
        if game.white_reward != 0.0:
            decisive += 1
    return all_examples, lengths, decisive


# ---------------------------------------------------------------------------
# evaluation (batched, two models)
# ---------------------------------------------------------------------------

def evaluate_batch(new_model, old_model, device, n_games: int,
                   sims_per_move: int, max_moves: int, openings):
    """Head-to-head: returns (win_rate_for_new, list_of_(result_str, new_is_white))."""
    games = []
    new_is_white_flags = []
    for i in range(n_games):
        opening = openings[i % len(openings)]
        b = chess.Board()
        for ucistr in opening:
            mv = chess.Move.from_uci(ucistr)
            if mv in b.legal_moves:
                b.push(mv)
        games.append(GameSlot(max_moves, starting_board=b))
        new_is_white_flags.append((i // len(openings)) % 2 == 0)

    while any(not g.done for g in games):
        # Group active games by which model is to move.
        new_to_move = []
        old_to_move = []
        for i, g in enumerate(games):
            if g.done:
                continue
            is_new_turn = (g.board.turn == chess.WHITE) == new_is_white_flags[i]
            (new_to_move if is_new_turn else old_to_move).append(i)

        # Run sims separately per side using its own model, but each in a
        # single batched pass over all games whose turn it currently is.
        for indices, mdl in ((new_to_move, new_model), (old_to_move, old_model)):
            if not indices:
                continue
            _run_sims_for_group(mdl, device, indices, games, sims_per_move,
                                add_root_noise=False)
            for i in indices:
                game = games[i]
                if game.done:
                    continue
                move, _ = _pick_move_from_root(game.root, temperature=0.0)
                game.board.push(move)
                game.move_count += 1
                if game.board.is_game_over(claim_draw=True) or game.move_count >= max_moves:
                    _finalize(game)
                game.root = None

    new_wins = 0
    draws = 0
    log = []
    for i, g in enumerate(games):
        res = g.board.result(claim_draw=True)
        log.append((res, new_is_white_flags[i]))
        if res == "1-0":
            new_wins += 1 if new_is_white_flags[i] else 0
        elif res == "0-1":
            new_wins += 0 if new_is_white_flags[i] else 1
        else:
            draws += 1
    return (new_wins + 0.5 * draws) / n_games, log
