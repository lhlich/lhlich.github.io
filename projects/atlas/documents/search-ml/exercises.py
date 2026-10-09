"""Study implementations. Requires Python 3 and NumPy. Run: python exercises.py"""
import heapq
import numpy as np


class MatrixSum:
    """Immutable rectangular matrix; inclusive rectangle coordinates."""
    def __init__(self, matrix):
        self.rows = len(matrix)
        self.cols = len(matrix[0]) if self.rows else 0
        if any(len(row) != self.cols for row in matrix):
            raise ValueError("matrix must be rectangular")
        self.prefix = [[0] * (self.cols + 1)
                       for _ in range(self.rows + 1)]
        for r in range(self.rows):
            for c in range(self.cols):
                self.prefix[r + 1][c + 1] = (
                    matrix[r][c] + self.prefix[r][c + 1]
                    + self.prefix[r + 1][c] - self.prefix[r][c]
                )

    def sum(self, r1, c1, r2, c2):
        if not (0 <= r1 <= r2 < self.rows
                and 0 <= c1 <= c2 < self.cols):
            raise ValueError("invalid inclusive rectangle")
        p = self.prefix
        return (p[r2 + 1][c2 + 1] - p[r1][c2 + 1]
                - p[r2 + 1][c1] + p[r1][c1])


def top_k(items, k):
    """Return (ID, finite score) pairs; score descending, earlier ties first.

    Each input occurrence is a candidate. IDs need not be comparable.
    """
    if k < 0:
        raise ValueError("k must be nonnegative")
    if k == 0:
        return []
    heap = []
    for order, (doc_id, score) in enumerate(items):
        if not np.isfinite(score):
            raise ValueError("scores must be finite")
        entry = (score, -order, doc_id)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)
    return [(doc_id, score) for score, _, doc_id
            in sorted(heap, key=lambda e: e[:2], reverse=True)]


def softmax_ce(logits, labels):
    """Mean hard-label CE and dL/dlogits. Finite BxC logits, B>0, C>0."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels)
    if z.ndim != 2 or min(z.shape) == 0 or not np.isfinite(z).all():
        raise ValueError("logits must be a nonempty finite BxC matrix")
    if (y.shape != (z.shape[0],) or y.dtype.kind not in 'iu'
            or (y < 0).any() or (y >= z.shape[1]).any()):
        raise ValueError("one valid integer class index per row required")
    shifted = z - z.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(len(y)), y].mean()
    grad = np.exp(log_probs)
    grad[np.arange(len(y)), y] -= 1
    grad /= len(y)
    return float(loss), grad


def linear_step(x, y, w, b, lr=0.1, l2=0.0):
    """One SGD step; loss includes l2*||w||^2/2, bias unregularized."""
    loss, dz = softmax_ce(x @ w + b, y)
    dw = x.T @ dz + l2 * w
    db = dz.sum(axis=0)
    objective = loss + 0.5 * l2 * np.sum(w * w)
    return float(objective), w - lr * dw, b - lr * db


def in_batch_loss(q, d, tau=1.0):
    """Q,D: Bxd, matched diagonal pairs; no normalization inside."""
    q, d = np.asarray(q, dtype=float), np.asarray(d, dtype=float)
    if q.ndim != 2 or q.shape != d.shape or not np.isfinite(tau) or tau <= 0:
        raise ValueError("matched Bxd matrices and positive finite tau required")
    loss, ds = softmax_ce(q @ d.T / tau, np.arange(q.shape[0]))
    return loss, ds @ d / tau, ds.T @ q / tau


def self_attention(q, k, v, allowed=None):
    """Single head, no batch: Q,K=Lxd_k, V=Lxd_v. True means allowed.

    Requires at least one allowed key per query row. No dropout/projections.
    """
    q, k, v = (np.asarray(a, dtype=float) for a in (q, k, v))
    if (q.ndim != 2 or k.shape != q.shape or v.ndim != 2
            or v.shape[0] != q.shape[0] or min(q.shape) == 0):
        raise ValueError("incompatible self-attention shapes")
    if not all(np.isfinite(a).all() for a in (q, k, v)):
        raise ValueError("inputs must be finite")
    scores = q @ k.T / np.sqrt(q.shape[1])
    if allowed is not None:
        allowed = np.asarray(allowed, dtype=bool)
        if allowed.shape != scores.shape or not allowed.any(axis=1).all():
            raise ValueError("every query needs at least one allowed key")
        scores = np.where(allowed, scores, -np.inf)
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ v, weights


if __name__ == '__main__':
    matrix = [[1, 1, 0, 2, 1], [1, 0, 0, 1, 0], [1, 1, 1, 1, 1],
              [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]]
    print('Full matrix:', MatrixSum(matrix).sum(0, 0, 4, 4))
    print('Top 2:', top_k([('A', 1), ('B', 3), ('C', 3)], 2))
    print('CE and gradient:', softmax_ce([[2, 1, 0]], [0]))
    print('Contrastive loss:', in_batch_loss(np.eye(3), np.eye(3))[0])
