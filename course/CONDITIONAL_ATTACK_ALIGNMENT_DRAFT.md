# Conditional relationship alignment: additional attack-side candidate

Status: 2 November 2026. The owner proposed relationship-distribution alignment
for injected nodes and explicitly clarified the attack side: make injected
nodes resemble normal nodes, with a gate deciding when alignment is enabled.
This is an additional method candidate, not an instruction to cancel the selected
LLM self/graph-evidence fusion defense. No training or new model/API query starts
from this draft; the exact descriptors, reference, gate and protocol are unfrozen.

## Existing evidence and prior work

The current MonTi diagnostics measured residual outcomes, endpoint effects and
execution differences, not a camouflage distribution gap or defense rejection
caused by that gap. Neither the 39 selected GPU direct-endpoint flips nor the
low-resource residual association establishes a need for alignment. First define
and measure the proposed gap on saved validation perturbations, with a training-
only reference and registered information boundaries, before selecting a remedy.

[CANA](https://arxiv.org/html/2208.01819v4) is a close precedent: Section 4.1 defines
camouflage through distribution similarity of injected versus original nodes'
two-hop ego networks, including structure and attributes. Section 4.2 uses a GIN
discriminator to guide camouflage. Its Section 5.2 GraphFD is an evaluation metric,
not its training objective. CANA's normal population is original graph nodes;
that does not mean verified benign news in a fraud-labeling task.
[HAO](https://arxiv.org/html/2202.08057v2), ICLR 2022, already regularizes injected-
node homophily alongside attack effectiveness; it is not a full multidimensional
distribution match. Thus alignment itself, a Boolean switch, or adding a gate
does not establish novelty. Exact conditional-policy originality remains unknown.

## Concrete hypothesis for a MonTi extension

Question: under the same legal attack resources, can an observable condition
identify when alignment is useful, preserving attack effectiveness while improving
camouflage relative to a fixed always-on penalty?

Define a relationship signature using quantities affected by permitted generated
features and edges, such as feature-to-neighbor similarity or a stated neighborhood
representation discrepancy. Define the normal reference explicitly: original
training nodes, known benign training nodes, or training pseudo-benign nodes from
a frozen surrogate are different populations. Non-target does not imply benign.
Freeze population selection, matching, distance and tolerances without test labels
or victim feedback. Degree/available-edge matching may be necessary; report no
compatible reference rather than quietly change populations.

An illustrative objective is `L_attack + lambda * g * L_align`, where the gate
controls only alignment activation/strength. It does not increase node/edge caps,
alter the target lists or remove evaluation cases. A distance-threshold gate is a
necessary simple control; it is not automatically an intelligent or novel policy.
A freely optimized gate multiplying a nonnegative penalty can collapse to zero;
its learning objective or constraint must prevent this trivial interpretation.
Do not equate low activation with learning when to camouflage.

In the one-node/one-edge cases, non-self degree is fixed at one. With a fixed
endpoint, the neighbor raw-feature mean is that target's feature. These are not
freely adjustable losses. One injected observation cannot establish distribution
matching. Use a justified reference-compatibility calculation or enough independent
training/reference observations, account for grouped samples, and disclose
infeasible desired signatures. Never add edges to manufacture a better match.

The attacker legitimately knows its own injected identities. The defender and its
LLM evidence cards must not receive those identities, injection labels or private
clean-graph differences. Alignment applies to attacker-generated objects, not to
collapsing genuine benign and fraudulent populations into a common distribution.
The attack-side gate is distinct from the selected defense-side LLM fusion gate;
whether the former is a threshold, learned policy or another API call is unresolved.

## Required controls and success criteria

Compare no alignment, fixed always-on alignment, a frozen training-derived
distance trigger, and a random trigger matched in activation rate. A learned gate
must beat suitable nonlearned controls, not merely an off switch. Keep reference
information, model capacity, optimization limits, seeds and physical attack caps
comparable; record actual nodes/edges and extra training/query/API cost.

Report attack effectiveness and camouflage separately, plus their trade-off.
Use evaluation features or a held-out detector beyond the distance directly
optimized, to avoid declaring success solely on the training regularizer. Lower
distance with lost attack effect is not a universal improvement. An isolated
distance match does not imply general imperceptibility or reliable evasion.

This candidate can later stress-test the selected LLM fusion defense: does that
defense remain useful when simple relationship anomalies are reduced? Such a
claim requires adaptation against the complete frozen defense and independent
validation; it is not established by an unmodified old attack. New victim/test
queries and any combined attack/defense training need the concrete later protocol.

Root and two agents reviewed design and primary sources. No new model import,
dataset/checkpoint load, inference, training, optimizer step, GPU work or LLM call
occurred. Every prior scientific result remains unchanged; course acceptance,
human review and final testing retain their separate checkpoints.
