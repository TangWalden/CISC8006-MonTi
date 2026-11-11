# Reproducing the standalone rule and reported counts

Run these commands from the repository root with Python. The standalone rule,
example, functional checks and saved-count checker use the standard library;
they do not require PyTorch, DGL, dataset downloads, checkpoints or API keys.

```bash
python -m unittest discover -s tests -v
python -m tools.apply_reallocation --input examples/reallocation_input.json --output work/reallocated.json
python -m tools.check_results
```

The example command applies all four arms to an explicit synthetic capture.
It preserves the supplied physical node IDs and writes the full result to a
new file. Parent directories are created; an existing output is never
overwritten. Choose a different filename when running it again.

Use the recorded Python 3.8.20 for exact historical control sampling; support
for newer Python does not promise identical random sampling across versions.

The tests check the isolated edge rule and its constraints. The result checker
recounts the published compact numbers, denominators and comparisons; it does
not rerun a neural model or independently validate original raw logits.
See [results](../results/README.md) for the exported files and their source hashes.

## What this checkout reproduces

This curated checkout provides the standalone deterministic algorithm and
saved-count verification. It does not provide a fresh-clone end-to-end rerun
of the historical training and model evaluation.

The original scientific reallocation run used commit
[`c3238ee62272d39f0337828718a3560abe101a00`](https://github.com/TangWalden/CISC8006-MonTi/tree/c3238ee62272d39f0337828718a3560abe101a00)
and protocol SHA256
`24dfa6cadd2b1c2a90a1b283bf659a5a0234e4d336e10d5142a813fe8ada0d14`.
Its implementation, registration, full evidence and prior project state are
preserved in the
[`ca093d8` archive](https://github.com/TangWalden/CISC8006-MonTi/tree/ca093d8b12ea153a076b02bbdbc4ede58508560c).

An exact historical rerun additionally requires official processed data/models
and private frozen run artifacts: the three terminal checkpoints, earlier
protocols, caches and hash-bound input files. Those assets are not supplied by
the compact exports. A protocol that references retained private paths is an
evidence record, not a self-contained public execution bundle.

## Historical ML environment

The original paper implementation under `run.py`, `modules/`, `models/` and
`utils/` is distinct from the standalone method. Recorded model runs reused
Linux with Python 3.8.20, PyTorch 1.13.1+cu117, DGL 1.1.2+cu117, NumPy 1.24.4,
SciPy 1.10.1 and CUDA 11.7. The archived
[runtime record](https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/environment/runtime.json),
[package lock](https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/environment/pip-freeze.txt)
and [upstream setup notes](https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/docs/UPSTREAM_README.md)
document that environment. No dependency installation is needed for the three
commands above; the historical ML environment was not rebuilt for this curation.

The formal learned/random reproduction and the later VALID experiment are
separate studies. Recounting either cannot establish final TEST effectiveness,
independent news-level uncertainty or completed human verification.

The retained upstream source license does not establish dataset/model
redistribution rights. Those separate permissions remain unconfirmed here;
the data and model archives are not included in this checkout.
