"""Batched single-process self-play and evaluation.

Runs N games concurrently in one process. At every MCTS step, leaves from
all active games are gathered into one batch and sent to the GPU as a
single forward pass. This keeps the GPU saturated -- the standard
single-machine alternative to spawning many CPU workers.

Performance notes:
- We never store boards on tree nodes. Each game keeps ONE mutable board
  for MCTS; selection walks down by pushing moves, expansion stores only
  (move, prior), and we pop the moves back off after backprop. Avoids
  ~6k python-chess `Board.copy()` calls per move that otherwise dwarf
  the GPU work.
- Virtual loss is applied on the path to each batched leaf so multiple
  in-flight evaluations don't all converge to the same node within a
  single tree (relevant when collecting >1 leaf per game per batch).
"""

from __future__ import annotations

import math
import chess
import torch
import numpy as np

from model import boards_to_batch_tensor, board_to_tensor, move_to_index, POLICY_SIZE


C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_WEIGHT = 0.25


# ---------------------------------------------------------------------------
# Tree node (board-free; the board lives on the GameSlot and is mutated
# along the selection path)
# ---------------------------------------------------------------------------

class Node:
    __slots__ = (
        "parent", "move", "children",
        "visit_count", "value_sum", "prior",
        "is_expanded", "is_terminal", "terminal_value",
        "virtual_loss",
    )

    def __init__(self, parent=None, move=None, prior: float = 0.0):
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
        return (self.value_sum - self.virtual_loss) / denom

    def ucb(self, parent_visits: int) -> float:
        return self.q() + C_PUCT * self.prior * math.sqrt(parent_visits) / (
            1 + self.visit_count + self.virtual_loss
        )


def _mark_terminal(node: Node, board: chess.Board) -> None:
    node.is_expanded = True
    node.is_terminal = True
    res = board.result(claim_draw=True)
    if res == "1-0":
        node.terminal_value = 1.0 if board.turn == chess.BLACK else -1.0
    elif res == "0-1":
        node.terminal_value = 1.0 if board.turn == chess.WHITE else -1.0
    else:
        node.terminal_value = 0.0


def _select(root: Node, board: chess.Board):
    """Walk down the tree from `root`, pushing each child's move onto
    `board` in place. Returns (leaf_node, path_of_nodes). Caller must
    pop `len(path) - 1` moves once done with the leaf."""
    path = [root]
    node = root
    while node.is_expanded and not node.is_terminal:
        pv = max(node.visit_count, 1)
        node = max(node.children, key=lambda c: c.ucb(pv))
        board.push(node.move)
        path.append(node)
    return node, path


def _expand(node: Node, priors, legal_moves, add_noise: bool = False) -> None:
    if not legal_moves:
        # Caller marks terminal with the board it has; here we just flag
        # expansion so the search loop stops descending.
        node.is_expanded = True
        return
    if add_noise:
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(legal_moves)).astype(np.float32)
        priors = (1.0 - DIRICHLET_WEIGHT) * priors + DIRICHLET_WEIGHT * noise
    # Children store only the move + prior; their board is computed on
    # demand by walking the path from the root.
    node.children = [Node(parent=node, move=mv, prior=float(p))
                     for mv, p in zip(legal_moves, priors)]
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


def _pop_path(board: chess.Board, depth: int) -> None:
    """Undo the moves pushed during selection (one per non-root node in path)."""
    for _ in range(depth):
        board.pop()


# ---------------------------------------------------------------------------
# batched NN evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _batch_evaluate(model, device, boards):
    """Return list of (legal_priors_or_None, value, legal_moves) parallel to `boards`.

    Uses AMP fp16 on CUDA and a single allocation for the input batch.
    """
    if not boards:
        return []
    tensors = boards_to_batch_tensor(boards)
    if device.type == "cuda":
        tensors = tensors.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits, values = model(tensors)
        logits = logits.float()
        values = values.float()
    else:
        tensors = tensors.to(device)
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
                        add_root_noise: bool, leaves_per_step: int = 8):
    """Run sims_per_move MCTS sims for each game in `game_indices`, batching
    leaf evaluations across all games AND across `leaves_per_step` selections
    per game per inner step. Virtual loss steers concurrent selections away
    from already-pending nodes within each tree.

    Mutates games[i].root and games[i].board (push/pop) in place.
    """
    if not game_indices:
        return

    # 1. fresh root per game; batched root-eval to seed priors.
    for i in game_indices:
        games[i].root = Node()
    root_results = _batch_evaluate(model, device, [games[i].board for i in game_indices])
    for i, (probs, _v, legal) in zip(game_indices, root_results):
        if not legal:
            _mark_terminal(games[i].root, games[i].board)
            continue
        _expand(games[i].root, probs, legal, add_noise=add_root_noise)

    active = [i for i in game_indices if not games[i].root.is_terminal]

    # 2. simulation loop. Per inner step we collect up to
    # `leaves_per_step` leaves *per game* (each protected by virtual loss),
    # then a single GPU forward processes all of them at once.
    sims_done = {i: 0 for i in active}
    while True:
        # (game_idx, leaf_node, path, leaf_board_snapshot, depth_to_pop)
        to_eval = []
        any_pending = False
        for i in active:
            need = sims_per_move - sims_done[i]
            if need <= 0:
                continue
            any_pending = True
            k = min(leaves_per_step, need)
            board = games[i].board
            for _ in range(k):
                leaf, path = _select(games[i].root, board)
                depth = len(path) - 1  # moves pushed onto board

                if leaf.is_terminal:
                    _backprop(leaf, -leaf.terminal_value)
                    sims_done[i] += 1
                    _pop_path(board, depth)
                    continue

                # Detect terminal on first encounter (without needing
                # board.copy(); we already have the live board state).
                if board.is_game_over(claim_draw=True) or not any(True for _ in board.legal_moves):
                    _mark_terminal(leaf, board)
                    _backprop(leaf, -leaf.terminal_value)
                    sims_done[i] += 1
                    _pop_path(board, depth)
                    continue

                _apply_vl(path)
                # Snapshot the board for this leaf because we need to
                # restore the live board for the next selection in this
                # game. This is one copy per leaf, vs the previous
                # one-copy-per-child-of-every-node.
                leaf_board = board.copy(stack=False)
                _pop_path(board, depth)
                to_eval.append((i, leaf, path, leaf_board))

        if not any_pending:
            return
        if not to_eval:
            continue

        results = _batch_evaluate(model, device, [t[3] for t in to_eval])
        for (i, leaf, path, _lb), (probs, value, legal) in zip(to_eval, results):
            _undo_vl(path)
            if not legal:
                _mark_terminal(leaf, _lb)
                _backprop(leaf, -leaf.terminal_value)
            else:
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
                   add_root_noise: bool = True,
                   leaves_per_step: int = 8):
    """Play n_games concurrently, batched on `device`. Returns
    (per_game_examples, per_game_lengths, decisive_count)."""
    games = [GameSlot(max_moves) for _ in range(n_games)]

    while any(not g.done for g in games):
        active = [i for i, g in enumerate(games) if not g.done]
        _run_sims_for_group(model, device, active, games, sims_per_move,
                            add_root_noise=add_root_noise,
                            leaves_per_step=leaves_per_step)

        for i in active:
            game = games[i]
            if game.done or game.root is None or not game.root.children:
                _finalize(game)
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
                   sims_per_move: int, max_moves: int, openings,
                   leaves_per_step: int = 8):
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
        new_to_move = []
        old_to_move = []
        for i, g in enumerate(games):
            if g.done:
                continue
            is_new_turn = (g.board.turn == chess.WHITE) == new_is_white_flags[i]
            (new_to_move if is_new_turn else old_to_move).append(i)

        for indices, mdl in ((new_to_move, new_model), (old_to_move, old_model)):
            if not indices:
                continue
            _run_sims_for_group(mdl, device, indices, games, sims_per_move,
                                add_root_noise=False,
                                leaves_per_step=leaves_per_step)
            for i in indices:
                game = games[i]
                if game.done or game.root is None or not game.root.children:
                    _finalize(game)
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
