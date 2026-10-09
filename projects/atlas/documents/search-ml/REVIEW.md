# Two-round consolidation review — October 4, 2026

## Round 1: technical accuracy and scope

Consolidated derivations and evaluation material with on-device system notes and eight search exercises. Reviewed five technical chapters and the reading/source map. Public interfaces are distinguished from unknown internal architecture; numerical budgets are illustrative.

Corrections and checks:

- Two designs: interactive search and assistant evidence retrieval; complete aggregation distinguished from top-k evidence; hard constraints preserved on empty results.
- Incremental-prefix counterexample: filtering previous top-k can lose valid completions. Cached trie top-k explicitly assumes positive score increases.
- Quantized exhaustive scanning is exact only for its quantized scoring function. Float ranking can differ. Removed corpus-size prescriptions and unsupported latency/quality promises.
- Per-query integer widening is blockwise; removed a full-matrix Boolean validation temporary. Stable top-k sorts only retained entries, including deterministic boundary ties.
- Context packing requires an injected counter and counts entire serialized evidence, including IDs/separators. Word counting remains an explicitly synthetic test fixture.
- Ranking metrics reject duplicate IDs and negative cutoffs and distinguish undefined denominators from zero performance.
- Exposure calibration, user-level DP sensitivity/adjacency, protocol assumptions, runtime-specific context capacity, and optimization diagnosis are qualified.
- Checked BM25/RRF/NDCG, CE/contrastive gradients, attention, vector storage, adapter parameters and KV arithmetic. LoRA/MRL and public Apple references are linked selectively.
- 39 tests passed: 15 foundational, 15 adapted search tests, 9 regression tests. These include brute-force update comparisons, independent stable-sort and edit-distance oracles, finite-difference gradients, full-matrix quantized-score comparison across block sizes, and a reproducible float/quantized rank mismatch. Blank exercise file compiles.
- Shape validation was added for MMR after review.

## Round 2: reader integration and layout

- 238 internal links checked across seven chapter pages and both copies of the continuous reader; no missing files/anchors, duplicate IDs, template tokens, missing image alternative text, or external reading assets.
- Chromium: all seven linked pages plus the single reader at 1365×1000 and 390×844 (16 page/viewport cases). No page-level overflow, broken equation images, or JavaScript errors.
- Chapter/single-reader navigation and show/hide controls passed. Print events expand and restore solutions. Inspected print-media styling.
- Visually inspected new mobile efficiency and priority-drill sections, desktop system, and equation/code/print screenshots. Tables and code deliberately scroll inside their containers on small screens.
- Updated chapter title, reading order, source map, runnable-code appendix, and practice instructions; rebuilt and reran affected integration checks.

Limitations: no independent external review, native-device benchmarks, Safari/iOS testing, or fixed-page PDF pagination validation. Reference tokenization and NumPy kernels are teaching implementations. These checks found no remaining defect within their tested scope; they do not imply that no future improvement is possible.
