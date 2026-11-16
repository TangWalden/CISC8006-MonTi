# Owner-requested GPU acceleration check

The owner explicitly requested GPU use during the frozen complete evaluation.
This isolated check does not read TEST outcomes, change the search objective,
retune parameters, modify the running CPU study or stop unrelated tasks.

A single bounded GPU service holds the two fixed teachers. Graph construction,
physical parity checks and endpoint ranking stay on CPU. Only float32 teacher
forward/input-gradient computation crosses the process boundary; TF32 is off.
The service receives numerical sequences, not target labels or IDs. GPU 0 is
eligible only with at least 3,000 MiB free, verified ordinal/physical UUID and
both Torch/native device counts. The allocator cap is 6% of its 24 GiB memory.
No other GPU process is stopped. The initial check has a 360-second GPU ceiling.

Two predetermined development cases per dataset compare CPU/GPU complete
search plans, final logits and input gradients. All plans must reproduce the
frozen choices, logit error must be at most 2e-5 and gradient error at most 2e-6.
End-to-end timings include CPU graph work and communication. A continuation
amendment may be registered only after these results; no speedup or numerical
equivalence is assumed in advance. Existing CPU work continues meanwhile.

Completed result: [measured timing and parity](../../results/channel_gpu/). The limited teacher-offload check passed endpoint parity but did not demonstrate useful end-to-end acceleration; the CPU full study continued unchanged.
