"""Search & retrieval coding drills (Python 3.9+, NumPy).

Each drill states its contract in the docstring; the body is a reference solution.
Practice loop:
    1. Open search_drills_blank.py (same signatures, bodies stubbed).
    2. Implement one drill in <= 20 min, no references.
    3. DRILLS=search_drills_blank python test_search_drills.py -k <DrillName>
Reference check:
    python test_search_drills.py

Drills
    1. BM25Index        incremental inverted index + BM25 + top-k
    2. ndcg/mrr/recall  ranking metrics with explicit conventions
    3. rrf              reciprocal rank fusion for hybrid retrieval
    4. Autocomplete     as-you-type trie with cached top-k per node
    5. FuzzyLexicon     typo tolerance: bounded Levenshtein + SymSpell deletes
    6. Frecency         O(1) exponentially-decayed usage score
    7. mmr/pack_context diversify + fit evidence into a small LLM context
    8. int8 retrieval   quantized brute-force vector search for on-device
"""
from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict

import numpy as np


def _check_k(k):
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("a nonnegative integer is required")


def _unique_ids(ids):
    if len(set(ids)) != len(ids):
        raise ValueError("ranked document IDs must be unique")


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text):
    """Lowercase word tokens. (Real systems: Unicode normalization, CJK segmentation.)"""
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Drill 1: incremental BM25 index
# ---------------------------------------------------------------------------
class BM25Index:
    """Inverted index with BM25 scoring and incremental add/replace/delete.

    add(doc_id, text)  insert, or replace if doc_id exists
    delete(doc_id)     -> bool (False if absent)
    search(query, k)   -> [(doc_id, score)], score desc, ties by doc_id asc.
                          Only docs matching >= 1 query term. Duplicate query
                          terms count once. Doc ids must be mutually comparable.
    IDF = ln(1 + (N - df + 0.5) / (df + 0.5))   (always positive)
    Complexity: search is O(sum of matched posting lengths + M log(k+1)), plus query analysis.
    """

    def __init__(self, k1=1.2, b=0.75):
        if not math.isfinite(k1) or k1 < 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("require finite k1 >= 0 and 0 <= b <= 1")
        self.k1, self.b = k1, b
        self.postings = defaultdict(dict)  # term -> {doc_id: tf}
        self.doc_len = {}                  # doc_id -> token count
        self.doc_tf = {}                   # doc_id -> Counter, needed for delete
        self.total_len = 0

    def __len__(self):
        return len(self.doc_len)

    def add(self, doc_id, text):
        if doc_id in self.doc_len:
            self.delete(doc_id)
        tf = Counter(tokenize(text))
        for term, count in tf.items():
            self.postings[term][doc_id] = count
        n = sum(tf.values())
        self.doc_len[doc_id] = n
        self.doc_tf[doc_id] = tf
        self.total_len += n

    def delete(self, doc_id):
        tf = self.doc_tf.pop(doc_id, None)
        if tf is None:
            return False
        for term in tf:
            plist = self.postings[term]
            del plist[doc_id]
            if not plist:
                del self.postings[term]
        self.total_len -= self.doc_len.pop(doc_id)
        return True

    def idf(self, term):
        n = len(self.doc_len)
        df = len(self.postings.get(term, ()))
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query, k=10):
        _check_k(k)
        if k <= 0 or not self.doc_len:
            return []
        avgdl = (self.total_len / len(self.doc_len)) or 1.0
        scores = defaultdict(float)
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            w = self.idf(term)
            for doc_id, f in plist.items():
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / avgdl)
                scores[doc_id] += w * f * (self.k1 + 1) / (f + norm)
        return heapq.nsmallest(k, scores.items(), key=lambda x: (-x[1], x[0]))


# ---------------------------------------------------------------------------
# Drill 2: ranking metrics
# ---------------------------------------------------------------------------
def dcg_at_k(grades, k):
    """Nonnegative finite grades; gain=2**grade-1; rank discount=log2(rank+1)."""
    _check_k(k)
    if any(not math.isfinite(g) or g < 0 for g in grades):
        raise ValueError("grades must be finite and nonnegative")
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def ndcg_at_k(ranked_ids, qrels, k):
    """Unique ranked IDs. Unjudged=0 by convention. IDCG uses all qrels.
    Returns None if IDCG=0 (including k=0); caller chooses aggregation policy.
    """
    _check_k(k)
    _unique_ids(ranked_ids)
    idcg = dcg_at_k(sorted(qrels.values(), reverse=True), k)
    if idcg == 0:
        return None
    return dcg_at_k([qrels.get(d, 0) for d in ranked_ids], k) / idcg


def reciprocal_rank(ranked_ids, relevant):
    """1/rank of first relevant doc; 0.0 if none retrieved."""
    for i, d in enumerate(ranked_ids, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked_ids, relevant, k):
    """Recall over known relevant IDs; unique results; None for no relevant IDs."""
    _check_k(k)
    _unique_ids(ranked_ids)
    relevant = set(relevant)
    if not relevant:
        return None
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def mean_defined(values):
    """Macro-average over queries, skipping None (undefined) entries."""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------------------
# Drill 3: reciprocal rank fusion
# ---------------------------------------------------------------------------
def rrf(ranked_lists, c=60, weights=None, top_n=None):
    """Weighted reciprocal-rank fusion; original 1-based ranks retained.
    Within-list duplicates keep their first rank; ties use comparable ID order.
    """
    if not math.isfinite(c) or c < 0:
        raise ValueError("c must be finite and nonnegative")
    if top_n is not None:
        _check_k(top_n)
    weights = [1.0] * len(ranked_lists) if weights is None else list(weights)
    if len(weights) != len(ranked_lists) or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("one finite nonnegative weight per list required")
    scores = defaultdict(float)
    for w, lst in zip(weights, ranked_lists):
        seen = set()
        for rank, d in enumerate(lst, 1):
            if d in seen:
                continue
            seen.add(d)
            scores[d] += w / (c + rank)
    out = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return out if top_n is None else out[:top_n]


# ---------------------------------------------------------------------------
# Drill 4: as-you-type autocomplete
# ---------------------------------------------------------------------------
def normalize_phrase(s):
    return " ".join(s.lower().split())


def normalize_prefix(s):
    """Like normalize_phrase but keeps one trailing space: 'new ' must not match 'newark'."""
    return re.sub(r"\s+", " ", s.lower()).lstrip()


class _TrieNode:
    __slots__ = ("children", "top")

    def __init__(self):
        self.children = {}
        self.top = []  # sorted list of (-weight, phrase), length <= cache_k


class Autocomplete:
    """add(phrase, delta>0) accumulates weight. complete(prefix, k) -> [(phrase, weight)]
    by weight desc, then phrase asc; requires k <= cache_k.

    Each node caches the top cache_k phrases in its subtree, so complete() is
    O(len(prefix) + k) and add() is O(len(phrase) * cache_k log cache_k).
    Invariant holds because weights only increase: a phrase evicted from a
    node's cache can never re-qualify without being re-added on that path.
    Decreases/deletes would need the subtree to recompute caches (or lazy repair).
    """

    def __init__(self, cache_k=10):
        _check_k(cache_k)
        self.root = _TrieNode()
        self.weight = {}
        self.cache_k = cache_k

    def add(self, phrase, delta=1.0):
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError("weights only increase in this contract")
        p = normalize_phrase(phrase)
        if not p:
            raise ValueError("empty phrase")
        w = self.weight.get(p, 0.0) + delta
        self.weight[p] = w
        node = self.root
        self._update(node, p, w)
        for ch in p:
            node = node.children.setdefault(ch, _TrieNode())
            self._update(node, p, w)

    def _update(self, node, p, w):
        top = [e for e in node.top if e[1] != p]
        top.append((-w, p))
        top.sort()
        node.top = top[: self.cache_k]

    def complete(self, prefix, k=5):
        _check_k(k)
        if k > self.cache_k:
            raise ValueError("k exceeds cache_k")
        node = self.root
        for ch in normalize_prefix(prefix):
            node = node.children.get(ch)
            if node is None:
                return []
        return [(p, -nw) for nw, p in node.top[:k]]


# ---------------------------------------------------------------------------
# Drill 5: typo tolerance
# ---------------------------------------------------------------------------
def levenshtein(a, b, max_dist=None):
    """Edit distance (insert/delete/substitute = 1). If max_dist is given and the
    distance exceeds it, returns max_dist + 1 early. O(len(a) * len(b)) time,
    O(min) space."""
    if max_dist is not None:
        _check_k(max_dist)
    if len(a) < len(b):
        a, b = b, a
    if max_dist is not None and len(a) - len(b) > max_dist:
        return max_dist + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if max_dist is not None and min(cur) > max_dist:
            return max_dist + 1
        prev = cur
    d = prev[-1]
    return d if max_dist is None or d <= max_dist else max_dist + 1


def _deletes(word, max_edit):
    """All strings reachable from word by <= max_edit single-char deletions."""
    out, frontier = {word}, {word}
    for _ in range(max_edit):
        frontier = {w[:i] + w[i + 1:] for w in frontier for i in range(len(w))}
        out |= frontier
    return out


class FuzzyLexicon:
    """SymSpell-style candidate generation. Precompute delete-variants of every
    vocab word; at query time, delete-variants of the term hit the table, and
    candidates are verified with bounded Levenshtein.
    Complete for Levenshtein <= max_edit: a substitution is one delete on each
    side, an insertion is one delete on one side.

    lookup(term) -> [(word, dist, freq)] sorted by dist asc, freq desc, word asc.
    """

    def __init__(self, max_edit=2):
        _check_k(max_edit)
        self.max_edit = max_edit
        self.freq = {}
        self.table = defaultdict(set)

    def add(self, word, freq=1):
        word = word.lower()
        if word not in self.freq:
            for v in _deletes(word, self.max_edit):
                self.table[v].add(word)
        self.freq[word] = self.freq.get(word, 0) + freq

    def lookup(self, term, max_edit=None):
        if max_edit is not None:
            _check_k(max_edit)
            if max_edit > self.max_edit:
                raise ValueError("requested radius exceeds indexed radius")
        d = self.max_edit if max_edit is None else max_edit
        term = term.lower()
        cands = set()
        for v in _deletes(term, d):
            cands |= self.table.get(v, set())
        hits = []
        for w in cands:
            dist = levenshtein(term, w, d)
            if dist <= d:
                hits.append((w, dist, self.freq[w]))
        hits.sort(key=lambda x: (x[1], -x[2], x[0]))
        return hits


# ---------------------------------------------------------------------------
# Drill 6: frecency
# ---------------------------------------------------------------------------
class Frecency:
    """score(item, t) = sum over past uses t_i <= t of 0.5 ** ((t - t_i) / half_life).
    O(1) per record/score by storing (score at last update, last update time):
        s <- s * exp(-lam * (t - t_last)) + 1,   lam = ln 2 / half_life
    Timestamps per item must be non-decreasing."""

    def __init__(self, half_life):
        if not math.isfinite(half_life) or half_life <= 0:
            raise ValueError("half_life must be positive")
        self.lam = math.log(2) / half_life
        self.state = {}  # item -> (score_at_t_last, t_last)

    def record(self, item, t):
        if not math.isfinite(t):
            raise ValueError("finite timestamp required")
        s, t0 = self.state.get(item, (0.0, t))
        if t < t0:
            raise ValueError("out-of-order timestamp")
        self.state[item] = (s * math.exp(-self.lam * (t - t0)) + 1.0, t)

    def score(self, item, t):
        if not math.isfinite(t):
            raise ValueError("finite timestamp required")
        if item not in self.state:
            return 0.0
        s, t0 = self.state[item]
        if t < t0:
            raise ValueError("query time before last use")
        return s * math.exp(-self.lam * (t - t0))

    def top(self, t, k):
        _check_k(k)
        return heapq.nsmallest(k, ((i, self.score(i, t)) for i in self.state),
                               key=lambda x: (-x[1], x[0]))


# ---------------------------------------------------------------------------
# Drill 7: diversify and pack evidence for a small-context LLM
# ---------------------------------------------------------------------------
def _unit(x):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("finite vectors required")
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("zero vector")
    return x / n


def mmr(query_vec, doc_vecs, k, lam=0.7):
    """Maximal marginal relevance with cosine similarity.
    Greedy: argmax_d  lam * sim(q, d) - (1 - lam) * max_{s in selected} sim(d, s).
    Returns indices in selection order; ties -> lower index. O(k * n * dim)."""
    _check_k(k)
    if not math.isfinite(lam) or not 0 <= lam <= 1:
        raise ValueError("lambda must be in [0,1]")
    if k == 0 or len(doc_vecs) == 0:
        return []
    query_vec = np.asarray(query_vec, dtype=np.float64)
    doc_vecs = np.asarray(doc_vecs, dtype=np.float64)
    if query_vec.ndim != 1 or doc_vecs.ndim != 2 or doc_vecs.shape[1] != query_vec.size:
        raise ValueError("a 1D query and compatible 2D document matrix are required")
    q, D = _unit(query_vec), _unit(doc_vecs)
    rel = D @ q
    k = min(k, len(D))
    selected = []
    max_red = np.full(len(D), -np.inf)  # max sim to any selected doc
    avail = np.ones(len(D), dtype=bool)
    for _ in range(k):
        red = np.where(np.isfinite(max_red), max_red, 0.0)
        score = np.where(avail, lam * rel - (1 - lam) * red, -np.inf)
        i = int(np.argmax(score))  # argmax returns first max -> lower index wins
        selected.append(i)
        avail[i] = False
        max_red = np.maximum(max_red, D @ D[i])
    return selected


def pack_context(passages, budget_tokens, overhead_tokens=0, count_tokens=None):
    """Priority-order greedy packing; returns kept IDs, skipping oversized items.

    Requires an explicit token counter for the target model. Candidate contexts
    are serialized as '[ID]\ntext' blocks separated by two newlines, and the
    WHOLE candidate context is counted each time. overhead_tokens optionally
    reserves extra per-item tokens outside that serialization. This budget is
    for evidence only; reserve instructions, history, tools and output first.
    A toy counter may be injected in tests but is not an LLM token guarantee.
    """
    _check_k(budget_tokens)
    _check_k(overhead_tokens)
    if not callable(count_tokens):
        raise ValueError("provide the target model's token counter")
    _unique_ids([pid for pid, _ in passages])
    kept, blocks = [], []
    for pid, text in passages:
        candidate = blocks + [f"[{pid}]\n{text}"]
        count = count_tokens("\n\n".join(candidate))
        _check_k(count)
        if count + overhead_tokens * len(candidate) <= budget_tokens:
            kept.append(pid)
            blocks = candidate
    return kept


# ---------------------------------------------------------------------------
# Drill 8: int8 quantized brute-force retrieval
# ---------------------------------------------------------------------------
def quantize_int8(x):
    """Symmetric per-row quantization: float32 input -> int8 codes + float32 scales.
    x_hat=codes*scales[:,None]. Rounding error <= scale/2, up to float roundoff,
    relative to the float32 input when scale is representable and no clipping.
    Zero rows use scale=1. Not an exact ranking guarantee.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float32))
    if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("finite vectors with nonzero dimension required")
    maxima = np.abs(x).max(axis=1)
    scales = maxima / 127.0
    if np.any((maxima > 0) & (scales == 0)):
        raise ValueError("quantization scale underflow")
    scales[maxima == 0] = 1.0
    codes = np.clip(np.rint(x / scales[:, None]), -127, 127).astype(np.int8)
    return codes, scales.astype(np.float32)


def top_k_indices(scores, k):
    """Finite 1D scores; descending score, then ascending index.
    Partition + a linear boundary-tie scan + sort of only k chosen entries:
    O(n + k log(k+1)) time and O(n) auxiliary memory in this NumPy version.
    """
    _check_k(k)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("finite 1D scores required")
    k = min(k, len(scores))
    if k == 0:
        return np.array([], dtype=int)
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)[:k - len(above)]
    chosen = np.concatenate((above, tied))
    return chosen[np.lexsort((chosen, -scores[chosen]))]


def int8_search(query, codes, scales, k, block_rows=1024):
    """Exhaustive search using quantized scores, approximate versus float inputs.
    Int32 accumulation is performed in bounded row blocks. This NumPy reference
    widens integers; it is NOT a native optimized int8 SIMD/ANE kernel.
    Work memory: O(block_rows*d + n), excluding input and top-k temporaries.
    """
    _check_k(k)
    _check_k(block_rows)
    q = np.asarray(query, dtype=np.float32)
    codes, scales = np.asarray(codes), np.asarray(scales, dtype=np.float64)
    if (q.ndim != 1 or codes.ndim != 2 or codes.dtype != np.int8
            or q.size == 0 or codes.shape[1] != q.size
            or scales.shape != (len(codes),) or block_rows == 0
            or not np.isfinite(q).all() or not np.isfinite(scales).all()
            or np.any(scales <= 0)):
        raise ValueError("compatible finite query, int8 codes and positive scales required")
    if (codes.size and codes.min() == -128) or q.size * 127 * 127 > np.iinfo(np.int32).max:
        raise ValueError("require [-127,127] codes and int32-safe dimension")
    qc, qs = quantize_int8(q)
    query32 = qc[0].astype(np.int32)
    approx = np.empty(len(codes), dtype=np.float64)
    for start in range(0, len(codes), block_rows):
        end = min(start + block_rows, len(codes))
        raw = codes[start:end].astype(np.int32) @ query32
        approx[start:end] = raw.astype(np.float64) * scales[start:end] * qs[0]
    return top_k_indices(approx, k), approx


def recall_vs_exact(approx_ids, exact_ids):
    """Agreement with float/exact top-k, not relevance recall. None if empty gold."""
    approx, exact = set(np.asarray(approx_ids).tolist()), set(np.asarray(exact_ids).tolist())
    return len(approx & exact) / len(exact) if exact else None


if __name__ == "__main__":
    idx = BM25Index()
    idx.add("n1", "Hike at Mission Peak with Alex, 6 miles")
    idx.add("n2", "Grocery list: eggs, milk")
    idx.add("n3", "Mission Peak sunrise hike notes")
    print("BM25:", idx.search("mission peak hike", 3))
    print("NDCG@5:", round(ndcg_at_k(list("abcde"), {"b": 2, "c": 1, "e": 3}, 5), 3))
    print("RRF:", rrf([["A", "B", "C"], ["B", "D", "A"]]))
    ac = Autocomplete()
    for p, w in [("photos", 5), ("phone", 9), ("photo booth", 2), ("pages", 1)]:
        ac.add(p, w)
    print("Autocomplete 'pho':", ac.complete("pho", 3))
    fz = FuzzyLexicon()
    for w, f in [("calendar", 50), ("calender", 1), ("camera", 40)]:
        fz.add(w, f)
    print("Fuzzy 'calandar':", fz.lookup("calandar"))
