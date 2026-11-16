# Requested GPU acceleration check

GPU execution was actually tested on four predetermined development cases
while the CPU complete-cohort study continued. All four selected endpoint
plans were identical; maximum logit difference was 2.15e-6 and maximum input
gradient difference 1.35e-7. This does not prove equality for every case.

| Mean end-to-end search time per case | CPU | Teacher on GPU |
|---|---:|---:|
| GossipCop, two fixed cases | 11.2124 s | 11.0086 s |
| YelpChi, two fixed cases | 11.0622 s | 12.3123 s |

The measured bridge did not demonstrate useful overall acceleration. Graph
construction and endpoint ranking remained on CPU; the experiment did not
test a fully CUDA-native graph implementation. The original CPU full study
was left uninterrupted. No TEST result influenced this operational decision.

One GPU 0 service made 24 teacher calls / 368 contexts / 12 input-gradient
calls, with 25.79 MiB peak allocated memory. Conservative GPU lifetime charge
was 0.0188359131 hours. No training, official victim query or API call occurred.
No other GPU process was stopped or modified; shared-device delay was not
measured. An initial launcher quoting failure occurred before remote execution
and is preserved separately. RPC credentials and model exports remain private.

[Measured summary](summary.json), [protocol, checks and source hashes](execution.json),
[implementation](../../experiments/channel_gpu/). These are root-authored checks,
not independent or human verification.
