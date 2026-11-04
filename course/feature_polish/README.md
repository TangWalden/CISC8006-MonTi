# Fixed-graph injected-feature refinement

Status: new owner-authorized development implementation and source freeze in preparation; no outcome claimed.

The preceding source-role correction did not improve aggregate development MR. Its saved online TRAIN objective was not globally weaker in the poorly transferring seed, so merely extending epochs is unsupported. The earlier joint-teacher companion also showed differing teacher/victim responses on the same saved graphs. This study preserves existing strong GCN-generator hard graphs and asks whether an independently fitted relationship model can improve only the injected features.

The [protocol](../FEATURE_POLISH_PROTOCOL_V1.json) fixes five conditions: unchanged features, GCN feature-PGD, full-TRAIN GAGA feature-PGD, terminal GAGA-PGD with a GCN prediction-preservation guard, and seeded random-direction modification. Each seed/group is generated once and all conditions use identical topology and initial injected features. Feature updates remain within the original box; extra optimization calls and actual movement are explicitly charged/reported. No victim feedback selects a proposal or step.

A fresh GAGA is fitted from its independently seeded initialization for 38 fixed epochs on original TRAIN labels and corresponding full-TRAIN group inputs. It never loads the victim weights, and its post-fit TRAIN performance is in-sample. At fixed topology, seven-token means are affine in injected features, allowing exact feature derivatives without a straight-through edge/reachability derivative. Hard-update equivalence and finite differences are required before refinement.

All refinements finish before official GAGA development evaluation. Three attack seeds and all 128 frozen VALID groups remain. A bounded CPU route is prepared because shared GPUs may be occupied: the teacher is explicitly CPU; then a single available GPU or CPU device is chosen before graph generation and used for the complete paired study. No other workload is interrupted, no device-identity comparison with prior runs is assumed, and CPU wall/process time is reported separately from GPU hours.

Positive effects are a research goal, not a promised result. Prior attempts stay immutable. No final TEST, LLM API, novelty, human review, course acceptance or submission completion is implied.

The [saved TRAIN diagnosis](../../evidence/feature_polish_v1/saved_train_diagnosis.json) is post-hoc online evidence, not a frozen-checkpoint replay or a paired TRAIN/VALID gap. Source-fixed seed 3 ends with 94.0146% online TRAIN wrong versus 93.7918% legacy and lower group-mean Softplus (0.239847 versus 0.249276). This supports testing a more direct hard-graph feature optimization path; it does not prove its cause or benefit.

A development candidate must lower its own GAGA mean group loss in every seed, improve official GAGA MR by at least one percentage point on average with all three seed differences positive, and exceed GCN-PGD/random means. The guarded arm is primary; the unguarded GAGA arm is a predeclared secondary candidate and never replaces the primary result. All intervals and negative controls remain reported; independent final TEST confirmation is separate.
