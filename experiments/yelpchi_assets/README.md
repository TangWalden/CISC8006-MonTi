# Executed asset and cache preparation

These exact scripts prepared the isolated QY stage on 14 November 2026.
`remote_asset_prepare.py` expects the recorded `asset_manifest.json` and a
runtime at `../../` with the locked original modules. It retrieves official
bytes, verifies expected hashes, records source/input hashes, then invokes
`cache_prepare.py`. Existing destination files deliberately stop replay.

The cache script builds original GAGA input caches and GCN representations,
then checks author/victim state compatibility without attack or GAGA forward
calls. It does not train or evaluate YelpChi. The recorded state is prepared
and on hold, so do not launch these scripts as a recurring diagnostic.

See [readiness, scope and hashes](../../results/yelpchi_readiness/).
