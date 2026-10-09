# Search & ML Study Notes

Open `index.html` in a browser. Keep chapter files together. Alternatively open `Study-Guide.html`, a self-contained reader with the same chapters and full code. Reading, equations, and solutions work offline. Primary-source links require internet.

## Reading track

1. Search: lexical analysis, prefix/fuzzy lookup, BM25, WAND, ANN and hybrid retrieval.
2. Modeling: targets, gradients, ranking losses, evaluation, exposure bias and distillation.
3. System: interactive personal search and assistant evidence retrieval; updates and privacy-aware learning.
4. Efficiency: quantization, embedding dimensions, LoRA, KV memory and device benchmarks.
5. Coding: prioritized search drills, stable ML numerics and foundational implementation.

The start page separates core exercises from optional topics. Open solutions only after attempting a problem. Show/hide controls apply to all expandable content. Printing expands solutions and restores their state afterward. Tables/code scroll on narrow screens.

## Practice files

- `search_drills.py`: corrected reference implementations.
- `search_drills_blank.py`: matching practice stubs; supporting normalization/helpers supplied.
- `test_search_drills.py`: 15 drill tests, including update and brute-force comparisons.
- `test_regressions.py`: 9 tests for corrected edge cases and independent oracles.
- `exercises.py`, `test_exercises.py`: foundational ML/coding implementations and 15 tests.

Requires Python 3.9+ and NumPy for code. Run reference verification:

```sh
python -m unittest discover -p 'test_*.py'
```

For practice, copy the blank file into a separate directory, implement a drill, and use `DRILLS=search_drills_blank python test_search_drills.py -k Autocomplete` (adjust drill name). Or save your attempt as `search_drills.py` there and run the tests. Unimplemented stubs are expected to fail.

`source/` contains editable fragments, generator, and an HTML checking script. Rebuilding additionally requires Matplotlib. `REVIEW.md` records the two completed review rounds and limitations.

The examples and budgets are teaching scenarios, not production measurements. The reference analyzer is small and English-like. Inject the actual model tokenizer for evidence budgets. The NumPy int8 scan is not a native optimized device kernel.
