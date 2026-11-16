# Full-pool gradient retrieval: registered two-dataset development study

The owner requests continued evidence-based development on both GossipCop and
YelpChi. The prior YelpChi rewriter changed logits on 676/690 occurrences but
changed no classes; GossipCop improved modestly. This motivates testing search
coverage and the selection objective, not assuming either explains all failures.

The frozen author generator is reused through exact saved captures. A downstream
optimizer ranks the entire legal TRAIN-benign endpoint pool by a surrogate
input derivative, physically preserving injected features, target attachments,
injected degrees and all node/edge budgets. At most two of the same three donor
regions are substituted. Each step verifies eight candidate endpoints per
available region with the independent TRAIN-fitted teacher, and may STOP.

This adapts the discrete gradient-ranking idea of
[HotFlip (ACL 2018)](https://aclanthology.org/P18-2006/) to grouped graph context.
The objective comparison is also motivated by the warning about attack-objective
failures in [Croce and Hein (ICML 2020)](https://proceedings.mlr.press/v119/croce20b.html).
We do not implement AutoAttack/DLR or claim their guarantees. The primary
candidate minimizes mean sigmoid of the teacher's fraud-minus-benign margin;
an ordinary Softplus arm isolates that choice. Temperature is fixed at one,
not calibrated confidence. This is a new downstream optimizer, not a newly
trained neural network, proof of novelty or a promised improvement.

Controls retain native, existing single neural and existing two-step rewriter
graphs. A uniform-shortlist search shares the full pool, regions, horizon and
per-step shortlist size. Each new arm also has a random endpoint control with
the same selected donor slots and exact exchange dose. Gradients/algebraic
pool scoring are additional compute; equal physical budgets do not imply equal
runtime. Different STOP decisions can also change total search work.

All existing 64 development groups and three repeats per dataset remain;
GossipCop has 571 occurrences/240 unique targets and YelpChi 230/230 per repeat.
These repeatedly inspected cohorts are exploratory, not untouched confirmation.
The primary candidate must beat native, both existing models and its own random
control in each dataset. Its uniform-search contrast and every alternative are
reported; no outcome-based case removal, retry, or budget expansion is allowed.

A graph-specific affine relation template accelerates endpoint evaluation.
Only remote TRAIN-benign nodes outside the original target one/two-hop set are
legal. Fixed TRAIN preflights compare all seven tokens and teacher logits with
physical canonical construction, including two simultaneous substitutions.
Every final new and random graph is physically checked before victim scoring.
Failure stops the pipeline. Equality is to the locked canonical implementation,
not an independent proof of globally recomputed graph semantics.

`prepare.py` freezes sources, parent outputs and cohorts. `driver.py` executes
both preflights, both selection stages, then both official VALID evaluations,
and a root-authored saved audit. No TEST, upstream generation, optimizer update,
new teacher, API or GPU call. CPU four threads/nice 10; finite phase deadlines.
Raw evidence remains private. Results are not known at registration.
