# Publication support

This directory retains only publication verification and historical figure tools.

- `verify_publication.py`: English-history and committed-checksum guard.
- `plot_evidence.py` and `plot_results_first.py`: sources for the historical
  figures under `docs/figures/`; they do not describe the latest method.
- `run.sh`: launcher for the recorded ML environment. Override `MONTI_PYTHON`
  when using a different environment location.

Current algorithm code is in [methods/](../methods/). Historical experiment
pipelines and protocols are available through the [archive](../docs/ARCHIVE.md).
