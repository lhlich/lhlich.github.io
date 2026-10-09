"""Tests for the drills. Brute-force oracles where possible.

    python test_search_drills.py                                  # reference
    DRILLS=search_drills_blank python test_search_drills.py       # your attempt
    DRILLS=search_drills_blank python test_search_drills.py -k Autocomplete
"""
import importlib
import math
import os
import random
import string
import unittest

import numpy as np

D = importlib.import_module(os.environ.get("DRILLS", "search_drills"))


def brute_bm25(docs, query, k1=1.2, b=0.75):
    toks = {d: D.tokenize(t) for d, t in docs.items()}
    n = len(toks)
    avgdl = sum(len(t) for t in toks.values()) / n
    out = {}
    for d, t in toks.items():
        s = 0.0
        for term in set(D.tokenize(query)):
            f = t.count(term)
            if not f:
                continue
            df = sum(term in x for x in toks.values())
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(t) / avgdl))
        if s > 0:
            out[d] = s
    return sorted(out.items(), key=lambda x: (-x[1], x[0]))


class TestBM25Index(unittest.TestCase):
    def test_against_brute_force_with_updates(self):
        rng = random.Random(0)
        vocab = ["red", "run", "shoe", "hike", "peak", "milk", "egg", "note"]
        idx, docs = D.BM25Index(), {}
        for step in range(300):
            op = rng.random()
            doc_id = f"d{rng.randrange(25):02d}"
            if op < 0.65:
                text = " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 12)))
                idx.add(doc_id, text)
                docs[doc_id] = text
            else:
                self.assertEqual(idx.delete(doc_id), doc_id in docs)
                docs.pop(doc_id, None)
            if docs and step % 10 == 0:
                q = " ".join(rng.sample(vocab, 2))
                got = idx.search(q, 5)
                exp = brute_bm25(docs, q)[:5]
                self.assertEqual([g[0] for g in got], [e[0] for e in exp])
                for g, e in zip(got, exp):
                    self.assertAlmostEqual(g[1], e[1], places=9)

    def test_known_numbers_and_edges(self):
        idx = D.BM25Index()
        self.assertEqual(idx.search("x", 3), [])
        big = D.BM25Index()
        for i in range(100):  # N=100, df("rare")=10 -> idf = ln(1 + 90.5/10.5) ~= 2.264
            big.add(i, "rare filler" if i < 10 else "filler other")
        self.assertAlmostEqual(big.idf("rare"), math.log(1 + 90.5 / 10.5))
        self.assertAlmostEqual(big.idf("rare"), 2.264, places=3)
        idx.add(1, "a a b")
        idx.add(2, "b c")
        self.assertEqual(idx.search("zzz", 3), [])
        self.assertEqual(idx.search("a", 0), [])
        self.assertEqual([d for d, _ in idx.search("a a a", 5)], [1])
        idx.add(1, "c")  # replace
        self.assertEqual(idx.search("a", 5), [])
        self.assertFalse(idx.delete(99))


class TestMetrics(unittest.TestCase):
    def test_ndcg_example(self):
        qrels = {"b": 2, "c": 1, "e": 3}
        self.assertAlmostEqual(D.ndcg_at_k(list("abcde"), qrels, 5), 0.5430, places=3)
        self.assertAlmostEqual(D.ndcg_at_k(list("ebc"), qrels, 3), 1.0)
        self.assertIsNone(D.ndcg_at_k(list("abc"), {"a": 0}, 3))

    def test_rr_recall_mean(self):
        self.assertEqual(D.reciprocal_rank(list("abc"), {"b", "c"}), 0.5)
        self.assertEqual(D.reciprocal_rank(list("abc"), {"z"}), 0.0)
        self.assertEqual(D.recall_at_k(list("abcd"), {"a", "d", "z"}, 2), 1 / 3)
        self.assertIsNone(D.recall_at_k(list("abc"), set(), 2))
        self.assertEqual(D.mean_defined([1.0, None, 0.0]), 0.5)


class TestRRF(unittest.TestCase):
    def test_example(self):
        out = dict(D.rrf([["A", "B", "C"], ["B", "D", "A"]]))
        self.assertAlmostEqual(out["A"], 1 / 61 + 1 / 63)
        self.assertAlmostEqual(out["B"], 1 / 62 + 1 / 61)
        self.assertEqual([d for d, _ in D.rrf([["A", "B", "C"], ["B", "D", "A"]])],
                         ["B", "A", "D", "C"])

    def test_duplicates_and_weights(self):
        out = dict(D.rrf([["A", "A", "B"]]))
        self.assertAlmostEqual(out["B"], 1 / 63)  # duplicate A keeps rank 1; B stays rank 3
        out = dict(D.rrf([["A"], ["B"]], weights=[2.0, 1.0]))
        self.assertGreater(out["A"], out["B"])


class TestAutocomplete(unittest.TestCase):
    def test_against_brute_force(self):
        rng = random.Random(1)
        words = ["photo", "phone", "photos", "pages", "maps", "mail", "music",
                 "new york", "newark", "news", "notes", "numbers"]
        ac, weights = D.Autocomplete(cache_k=6), {}
        for _ in range(400):
            w = rng.choice(words)
            delta = rng.choice([0.5, 1, 2, 3])
            ac.add(w, delta)
            weights[w] = weights.get(w, 0) + delta
            prefix = rng.choice(["", "p", "ph", "pho", "phot", "n", "new", "new ", "m", "ma", "x"])
            k = rng.randint(1, 6)
            exp = sorted(((p, s) for p, s in weights.items() if p.startswith(prefix)),
                         key=lambda x: (-x[1], x[0]))[:k]
            self.assertEqual(ac.complete(prefix, k), exp)

    def test_normalization_and_contract(self):
        ac = D.Autocomplete(cache_k=3)
        ac.add("  New   York ", 2)
        ac.add("newark", 5)
        self.assertEqual(ac.complete("NEW ", 3), [("new york", 2)])
        with self.assertRaises(ValueError):
            ac.add("x", 0)
        with self.assertRaises(ValueError):
            ac.complete("n", 4)


class TestFuzzy(unittest.TestCase):
    def test_levenshtein(self):
        self.assertEqual(D.levenshtein("kitten", "sitting"), 3)
        self.assertEqual(D.levenshtein("", "abc"), 3)
        self.assertEqual(D.levenshtein("abc", "abc"), 0)
        self.assertEqual(D.levenshtein("kitten", "sitting", max_dist=1), 2)
        self.assertEqual(D.levenshtein("a", "abcdef", max_dist=2), 3)

    def test_lookup_matches_brute_force(self):
        rng = random.Random(2)
        alpha = "abcde"
        vocab = {"".join(rng.choice(alpha) for _ in range(rng.randint(1, 7))) for _ in range(150)}
        fz = D.FuzzyLexicon(max_edit=2)
        freq = {}
        for w in vocab:
            f = rng.randint(1, 9)
            fz.add(w, f)
            freq[w] = f
        for _ in range(100):
            q = "".join(rng.choice(alpha) for _ in range(rng.randint(1, 8)))
            for d in (1, 2):
                exp = sorted(((w, D.levenshtein(q, w), freq[w]) for w in vocab
                              if D.levenshtein(q, w) <= d), key=lambda x: (x[1], -x[2], x[0]))
                self.assertEqual(fz.lookup(q, max_edit=d), exp)


class TestFrecency(unittest.TestCase):
    def test_against_direct_sum(self):
        rng = random.Random(3)
        fr = D.Frecency(half_life=7.0)
        uses = {}
        t = 0.0
        for _ in range(200):
            t += rng.random() * 3
            item = rng.choice("abcde")
            fr.record(item, t)
            uses.setdefault(item, []).append(t)
        now = t + 1.5
        for item, ts in uses.items():
            exp = sum(0.5 ** ((now - ti) / 7.0) for ti in ts)
            self.assertAlmostEqual(fr.score(item, now), exp, places=9)
        top = fr.top(now, 2)
        exp_top = sorted(((i, fr.score(i, now)) for i in uses), key=lambda x: (-x[1], x[0]))[:2]
        self.assertEqual([i for i, _ in top], [i for i, _ in exp_top])
        self.assertEqual(fr.score("zzz", now), 0.0)
        with self.assertRaises(ValueError):
            fr.record("a", 0.0)


class TestContext(unittest.TestCase):
    def test_mmr(self):
        q = [1.0, 0.0]
        docs = [[1.0, 0.10], [1.0, 0.11], [0.8, -0.6], [0.0, 1.0]]
        self.assertEqual(D.mmr(q, docs, 4, lam=1.0), [0, 1, 2, 3])
        self.assertEqual(D.mmr(q, docs, 2, lam=0.5), [0, 2])  # near-duplicate 1 suppressed
        self.assertEqual(len(D.mmr(q, docs, 10)), 4)

    def test_pack_context(self):
        ps = [("a", "w " * 50), ("b", "w " * 100), ("c", "w " * 20)]
        self.assertEqual(D.pack_context(ps, budget_tokens=90, overhead_tokens=5, count_tokens=lambda s: len(s.split())), ["a", "c"])
        self.assertEqual(D.pack_context(ps, budget_tokens=10, count_tokens=lambda s: len(s.split())), [])


class TestInt8(unittest.TestCase):
    def test_quantize_error_bound(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(50, 32)).astype(np.float32)
        codes, scales = D.quantize_int8(x)
        self.assertEqual(codes.dtype, np.int8)
        err = np.abs(codes * scales[:, None] - x)
        self.assertTrue(np.all(err <= scales[:, None] / 2 + 1e-6))
        c0, s0 = D.quantize_int8(np.zeros((1, 4)))
        self.assertTrue(np.all(c0 == 0))

    def test_top_k_and_recall(self):
        self.assertEqual(D.top_k_indices(np.array([1.0, 3.0, 3.0, 2.0]), 2).tolist(), [1, 2])
        self.assertEqual(D.top_k_indices(np.array([5.0, 5.0, 5.0]), 2).tolist(), [0, 1])
        rng = np.random.default_rng(5)
        X = rng.normal(size=(3000, 64))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        codes, scales = D.quantize_int8(X)
        recalls = []
        for _ in range(20):
            q = rng.normal(size=64)
            exact = D.top_k_indices(X @ q, 10)
            approx, _ = D.int8_search(q, codes, scales, 10)
            recalls.append(D.recall_vs_exact(approx, exact))
        self.assertGreaterEqual(np.mean(recalls), 0.95)


if __name__ == "__main__":
    unittest.main(verbosity=1)
