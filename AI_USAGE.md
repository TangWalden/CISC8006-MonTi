# AI use

AI assistants supported paper/code review, experiment design, implementation,
execution supervision, saved-record audits and documentation. AI-written code
and agent audits are not human technical review or verified student contributions.

## Recorded experimental use

- An external DeepSeek fusion experiment made 120 physical calls. Its selected
  cases did not establish superiority over graph-only, static or neural gates.
- A later scheduling experiment made 15 physical calls; 11 decisions were applied,
  with registered fallbacks. No overall LLM scheduling advantage was established.
- The current edge-reallocation method uses no LLM/API call or extra training.
  Its development gain and failed stability criterion are both retained in
  [the results](results/).

## Publication cleanup

The 11 November 2026 cleanup extracted the existing algorithm without changing
its bytes, exported compact saved results, checked all case denominators, and
ran synthetic tests and the JSON example. It added no model inference, training,
dataset/checkpoint load, external experimental API call or GPU research.

Earlier positive, negative and failed experiments remain in the
[immutable archive](docs/ARCHIVE.md). Full raw-output review and genuine human
verification remain distinct from these automated checks.
