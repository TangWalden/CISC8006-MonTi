# LLM-mediated evidence fusion: owner-selected direction

Status: 2 November 2026. The owner selected detector-side defense and clarified
that an externally called LLM agent should fuse self/content evidence with
relationship evidence, not select neighbors or edges. The owner additionally
proposed describing both evidence streams in text and learning how to decide
from those descriptions. Direction and LLM use are confirmed; the exact learning
mechanism, inputs, API budget and experiment protocol are not frozen. No new
training, real-data inference, LLM call or final test is launched by this draft.

The earlier uncommitted edge-gate interpretation was superseded before any
implementation or experiment. V3 grouping/evaluation remains declined. Budget
conditioning is unselected. All frozen reproduction and diagnostic files stand.

## Intended pipeline

1. A self-only predictor reads the node's own features. A graph-context predictor
   reads node features and relationships. Keep original GAGA unchanged as a
   possible graph branch; this is an outer fusion interface, not a new edge
   filtering or graph-to-sequence pipeline.
2. Build an immutable evidence card containing measured branch outputs,
   calibration status, current-graph summaries and explicit missing fields.
   Graph-context predictions already use the center's features, so the streams
   overlap and must not be assumed independent or called pure relationship data.
3. The LLM produces short, evidence-referenced textual descriptions of both
   streams and their conflicts. Descriptions must preserve numeric facts and
   missingness; they are not independent verification or hidden proof of truth.
4. An LLM decision stage reads the evidence/descriptions and returns a validated
   fusion weight. Fixed code combines the registered branch probabilities and
   applies a frozen decision threshold. It does not accept an unconstrained
   prose verdict overriding that rule.

For example, define the self weight alpha in a preregistered finite set and use
`p_final = alpha * p_self + (1 - alpha) * p_graph`. Neither the weight set nor
threshold is fixed by this discussion draft. A trained fusion model can be a
separate comparison, but is not the same deployment as a live LLM decision agent.

## What is actually available

[The loader](../utils/utils.py) and [input manifest](../evidence/data/input_manifest.json)
expose graph/tensor features, labels, partitions and derived GAGA caches, not a
readable article body/title/URL or an audited node-to-article mapping. A metadata-
only listing of the server's processed `Datasets/GossipCop` directory found 13
files and no standalone txt/json/jsonl/csv/tsv/html/md candidate. This is a bounded
interface/directory observation, not proof that raw text is absent from every
archive or original source. No binary dataset was opened for this inspection.

An MLP self-only class exists, but the manifest has no verified self-only
checkpoint. It must be trained/registered before treating its probabilities as
available evidence. GCN/GAGA provide logits and representations, but softmax,
margin and entropy are not already-calibrated reliability. Calibration needs its
own declared data boundary. Current graph statistics must be recomputed from the
received graph, not privileged clean caches.

Until an auditable raw-text mapping is established, describe the self stream as
numeric-feature/model evidence. Do not invent article sentences, citations,
events, users or content-level reasons from an embedding. Common graph neighbors
are not automatically shared users. A semantic-text extension needs a separate
input/provenance check, and must be distinguished from evidence-card fusion.

## Why an LLM might help, and why it might not

Hypotheses, not results: a pretrained language model may use meaningful semantic
descriptions, recognize certain cross-source conflicts, or exploit training
examples with limited task-specific labels. Evidence-referenced text can also
make selected inputs and missingness easier to inspect. These possibilities do
not establish better accuracy, calibration, robustness or cost than a neural gate.

Simply rewriting identical numeric facts as sentences adds no new case evidence.
A numeric gate may be more accurate, stable and economical. Summaries can omit
facts or fabricate plausible rationales. Fluency and self-reported confidence
must not be scored as detection success. If raw news or external knowledge is
added, attribute that information difference separately from the fusion method;
an appropriate same-information comparison is required.

The selected old GPU diagnosis has 39 new flips in 243 target occurrences, all
at the selected direct endpoint. It motivates examining evidence conflict but
does not show that the streams disagree, that self-only prediction is correct,
or that an LLM can recover any of those cases. Obtain those missing paired
measurements only under the later registered protocol.

For binary branches evaluated at the same fixed threshold, a convex probability
mixture cannot correct a case when both branches are on the wrong side. Register
joint-error and disagreement tables to quantify whether useful routing opportunity
exists. A label-based oracle comparison is an offline analysis bound, never an
agent input or deployed choice rule. If the LLM independently judges new semantic
evidence and overrides both branches, that is an additional predictor, not pure
weight fusion, and requires an LLM-only/same-information comparison.

## Learning from the text

Keep these mechanisms distinct before selecting one:

- Frozen API LLM with training-only worked examples or prompt selection. This
  is in-context/prompt-based adaptation, not updating the LLM's weights. The
  live agent can remain the decision-maker required by the owner.
- A trainable text encoder/gate that learns from evidence descriptions, or
  distillation of LLM decisions into a smaller model. This directly learns a
  text-to-fusion mapping, but changes the deployment: a smaller trained model,
  not necessarily a live LLM call, makes the final decision.
- Fine-tuning a language model. This needs its own supported model/interface,
  training data and compute plan; no available API fine-tuning capability or
  resource feasibility is assumed.

Descriptions must be produced without the case's ground-truth label at both
training-example generation and evaluation. Training labels can supervise the
fusion learner or provide training-example feedback afterward. Prepare honest
held-out or cross-fitted base-model evidence for supervised stacking. Splitting
saved predictions after a base model saw those labels does not make them
out-of-fold; the cost of generating proper evidence remains unresolved. Do not
silently repurpose validation/test labels to train the gate or choose examples.

## Comparisons needed to answer the owner's question

Retain self-only, graph-only and fixed-fusion baselines. The following planned
contrasts separate two claims:

| Method | Evidence description | Decision-maker | Purpose |
|---|---|---|---|
| Numeric neural gate | Original measured card | Small trained network | Ordinary learned fusion baseline |
| Template-text LLM fusion | Deterministic faithful serialization | Same frozen LLM | LLM decision without generated descriptions |
| Generated-text LLM fusion | Evidence-referenced LLM descriptions | Same frozen LLM | Added value of generated textual representation |
| Text-trained gate, if selected | Same generated descriptions | Trained text model | Value and cost of learning the textual representation |

Give comparisons the same underlying case facts and branch outputs; record added
information, model capacity, examples and generation/decision token budgets. Mask
each stream separately, check numerical/citation fidelity, and test misleading or
missing descriptions without changing real labels. Cache by complete evidence,
graph context, model and prompt version, not node ID alone across different group
attacks. API model/version, prompts, raw usage, retries, failures, latency and
actual charged cost must be recorded. The 12 GPU-hour limit does not define an
LLM-token or monetary allowance; freeze an explicit additional API ceiling.

Keep every case in the denominator. Invalid output, timeout or missing evidence
uses a predeclared static fallback; do not omit hard cases. Article text and
generated descriptions are data, not instructions: no tool execution from them,
schema/weight validation and prompt-injection tests are required before live use.
The agent never receives attack-target status, injected IDs, private clean-graph
differences or evaluation labels. Train/pilot data must include both benign and
fraud cases; an always-fraud policy cannot count as a successful defense.

Measure clean utility/benign false positives and attacked recall under unchanged
case weights, plus adaptive attacks against the full fusion pipeline. Explain
gradient-free/adaptive attack limits; lack of a gradient through an API is not
evidence of robustness. Report negative outcomes and cost, not just fluent text.

## Prior work and remaining checkpoint

[ARG, AAAI 2024](https://arxiv.org/abs/2309.12247) already studies LLM-generated
rationales guiding a smaller detector and a distilled deployment. Its abstract
also reports that its tested LLM alone underperformed fine-tuned BERT. Thus
LLM reasoning is not a general superiority guarantee, and generating descriptions
for learned decisions is not claimed as a new invention. Related semantic/graph
work includes [LLM Enhanced Semantics Mining, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.31/).
This is a bounded primary-source check, not an exhaustive novelty review or a
reproduction of those results. No claims from inaccessible DOI pages are added.

The candidate research question is whether grounded descriptions of self versus
graph-context evidence improve fusion robustness under MonTi perturbations,
beyond same-information neural and template-text controls, without harming clean
utility. Exact originality and effect remain unestablished. Next resolve raw-text
mapping, branch readiness and learning mode, then prepare a bounded validation-
only protocol for confirmation. Full testing, final tag, actual course acceptance
and genuine human verification retain separate checkpoints.

Root and two agents inspected existing code/metadata; primary sources above were
read. No model import, raw prediction read, model inference, optimizer update,
training, GPU work, dependency download or LLM API request occurred.
