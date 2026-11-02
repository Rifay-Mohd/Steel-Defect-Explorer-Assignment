# Frozen experiment regression tests

Run from the project root using the existing environment:

```bash
snip env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests -q -o cache_dir=tests/.pytest_cache --basetemp=tests/.tmp
```

The suite reads existing data and artifacts without training models or regenerating outputs. Metadata is validated through the installed `jq` CLI. Source tests compare the retained ZIP members, hashes, original one-hot labels, exact partition bytes, seed-42 membership, and training-only CV fold identities. Pipeline tests check training-only scaler statistics, repeated inference, saved source-row ordering, offline metrics, and bit-exact results after column reordering. These checks verify persisted evidence; they cannot prove the historical execution of training code beyond the recorded artifacts.

Metric tests use independently specified seven-class perfect and missing-class oracles, shuffled probability columns, deliberately different hard decisions, and invalid matrices. Upload tests exercise raw duplicate headers, target indicators, malformed records, invalid values and labels, BOM handling, difficult float representations, and both actual size boundaries.

Streamlit tests execute unchanged `app.py` through `runpy` inside real `AppTest` runs. Only `st.file_uploader` is monkeypatched to return a named `BytesIO`; validation, model loading, inference, charts, controls, and session caches remain real. Upload changes are supplied between reruns. This exercises the app's byte-identity invalidation, but does not simulate the browser uploader transport or its `on_change` callback. Assertions cover initial state, all five model selections, the comparison button, metric reruns, valid replacement, invalid replacement, recovery, and removal. Mocking failures are ordinary test failures, not skipped or replaced with synthetic app results.

## Observed issues

- The CSV reader also enforces Python's current `csv.field_size_limit()` (131,072 characters in this environment). A syntactically valid numeric field above that limit is reported as “Malformed CSV syntax,” even if the whole upload is below 5 MiB. The byte-boundary test distributes padding over 100 bounded fields to isolate the documented upload cap.
- The run emits deprecation warnings for pandas `float_precision` and seaborn's use of Matplotlib's `vert` parameter. A one-class metric oracle also triggers scikit-learn's single-label warning. Warnings are retained in pytest output.
