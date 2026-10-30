# Score-role replay amendment 1: CUDA visibility compatibility

Recorded after startup-only v1 and before v2 model execution. Preserve the original source/protocol at commit `e5cb731a10ec83b0c328faf63a61201ea6c816cd` and its failed registration, guardian and log. V1 completed **zero GCN forwards and zero MonTi forwards**, so this change follows no new attack outcomes.

## Observed setup discrepancy

With the intended GPU UUID in CUDA_VISIBLE_DEVICES, this locked PyTorch 1.13.1 environment reports `torch.cuda.device_count() = 0` while its native CUDA runtime reports one device and CUDA availability is true. A fresh metadata-only process with ordinal `CUDA_VISIBLE_DEVICES=0` reports one through both interfaces. The [metadata record](../evidence/score_role_replay_v1/visibility_diagnostic.json) retains both observations. This documents the observed environment incompatibility without asserting its library-internal cause.

V1's guarded process elapsed 2.4935 seconds. The two metadata-only checks elapsed 1.3874 and 1.3958 seconds; conservatively charge all three durations against the aggregate 360-second forensic reservation, although the metadata checks did not load models.

## Revised setup and unchanged measurement

Use separate v2 source/protocol and an exclusive v2 output directory. Select logical ordinal zero through CUDA visibility, require one device through both interfaces, and establish the **physical UUID** before model work: allocate one float as setup, synchronize, and query this process's own GPU UUID. Allow at most three metadata reads, with 0.1-second spacing only when the process is not yet listed. Stop if the sole observed UUID does not match the preregistered GPU 0 UUID; do not switch devices. This setup and all metadata operations count toward elapsed cost.

Keep the complete cohort, three checkpoints, RNG, weights, original edge construction, score bands, 96 MonTi / 224 GCN forwards and 192 algebra-only top-k references unchanged. Keep the 300-second main limit, 330-second guardian and hard kill after another 10 seconds. Reserve at most **0.095 GPU-hour (342 seconds)** for v2, within the remaining aggregate 0.1-hour reservation after prior costs. Charge the greater main/guardian elapsed time. No automatic retry, training, corrected-edge evaluation, victim/test query, policy selection or full evaluation is authorized by this amendment.
