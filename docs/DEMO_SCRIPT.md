# Saved-evidence demonstration

This is a proposed speaking and demonstration plan for Junyi Tang and Yifan
Fan. It does not record actual contributions, a completed rehearsal or a course
timing requirement. The short demonstration allocates about 90 seconds to each
member. The separate slide notes propose a six-minute main talk, divided
equally between them.

## What the demonstration opens

Open the [saved-case viewer](demo/index.html). It reads the audited
[formal public export](../results/final_staged/final_public.json) and displays
only recorded cases, budgets, outcomes and graph edges. It does not load a
model, generate an attack, train a policy or query a detector. The formal
protocol and its source identities are in [FINAL_EVALUATION.md](FINAL_EVALUATION.md),
with [publication provenance](../results/final_staged/provenance.json).

Until the final audited export is attached, the page explicitly says that
formal results are pending. Keep that state visible. Development-pilot values
must never fill an empty formal-results chart or table.

The editable Bento presentation is a separate file with native text, charts,
tables and pipeline objects. Use the current published presentation links in
the [project README](../README.md). The published Bento is a key-free editable
file copy; no live coediting room is configured. Editing that copy does not
change the committed evidence snapshot.

## Proposed demonstration

### Junyi Tang: the case and the denominator, about 90 seconds

1. Point to the page's completed/pending status and named data source. If formal
   evidence is unavailable, demonstrate the method description and stop before
   claiming an experimental score.
2. Use the default first case with an included graph and choose **Native MonTi**.
   Graph inclusion uses the pre-outcome geometry rule: the first two registered
   TEST groups per seed with at most eight auxiliary nodes, 32 targets, 128
   edges and a positive candidate quota. The menu still includes every case.
   Do not choose a case because it makes the proposed method look best.
3. Explain the three roles. Targets are the fixed known fraud-target list.
   Candidates are original graph context and possible endpoints; their class
   is not assumed benign. Auxiliary nodes are newly injected under a budget.
4. Read the selected case's missed-target numerator and complete denominator.
   Explain that MR includes targets already missed on the clean graph. The
   separately reported new-flip count measures clean-correct targets that
   became wrong, when that count is supplied.
5. Point to the structural budget. The original node and edge totals remain
   fixed across the compared policies.

Suggested explanation: “This is a saved case, not a new attack running now.
The target list and budget are already fixed. The page reports how many of
those fraud targets the recorded detector missed.”

### Yifan Fan: the staged policy and fair comparison, about 90 seconds

1. Keep the same case selected and switch to **Staged pipeline**. Explain the
   fixed upstream planner and the trained downstream candidate-edge selector.
   Only the downstream receives optimizer updates in this method.
2. Read the target and candidate quotas. Explain that the requested half split
   is projected to what the legal endpoints can support. It is not literally
   50:50 in every case. Individual auxiliary degrees and the native
   auxiliary-to-auxiliary policy can also differ.
3. Switch to **Random candidate connections** and read the same-case comparison.
   The random control shares the staged quota and upstream input. A case can
   improve, tie or decline; one selected graph does not establish the overall
   result.
4. If a saved edge list is included, use the edge-role filter to distinguish
   target and candidate connections. Large graphs use an explicitly limited
   view. Read the displayed-versus-recorded edge counts. A case without an edge
   list shows counts only and no invented topology.
5. Finish at **Complete formal summary**. State the actual formal mean and
   paired interval as recorded, including an inconclusive or negative result.
   Keep the random comparison visible. The complete pipeline comparison does
   not prove that role division alone caused the entire difference.

Suggested explanation: “The downstream chooses the remaining candidate links
after the upstream is frozen. We compare the complete pipeline on the same
cases and budgets. Its final effect comes from the full table, not this one
picture.”

## Reading the graph accurately

- Labels are local indices such as A0, T0 and C0. They are not original dataset
  node identities. An edge record has an auxiliary source, a destination role
  and a local destination index.
- The compact view displays at most 18 nodes per role and 220 stored edges.
  Hidden endpoints and edges remain in the saved record and full budget.
- Target fill denotes known fraud membership. Prediction rings appear only
  when saved per-target correctness values are supplied.
- A stored edge and a stored prediction do not identify which edge caused the
  model's decision. The viewer performs no causal intervention.
- Native MonTi's actual target/candidate/auxiliary edge counts must not be
  replaced by the staged arm's planned quota.
- Only the bounded geometry-selected cases include full edge arrays. All
  2,727 formal seed/group cases retain their counts and saved predictions in
  the public export. Cases without edge arrays use a counts-only view.

## Offline and published use

The viewer's `monti-result-snapshot` and `monti-result-sha256` metadata are blank
until the audited data is committed. With both fields blank, it keeps its
pending state unless a completed local export is available through the existing
repository-relative paths or **Load local result JSON**.

For GitHub Pages, publication will set an immutable 40-character commit and the
export's 64-character SHA-256. The viewer then fetches the single public JSON
from that commit on `raw.githubusercontent.com`, without credentials, and
verifies its bytes with browser SHA-256 before parsing or displaying it. The
data-source label shows the actual URL and verified hash. The JSON is not copied
into the documentation folder. A malformed pointer, unavailable verification,
failed fetch or hash mismatch produces an explicit error, with no fallback to
another result or development scores.

For offline use, choose the saved `final_public.json` with **Load local result
JSON**. When a hash is configured, the local file must match that same hash
before it can be displayed. With blank metadata, the existing formal-export
scope checks still apply. The file stays in the browser; no upload or model
service is involved.

The export uses MR percentages from 0 to 100 and percentage-point contrasts.
It identifies completed formal results, seeds 0/1/2, the actual TRAIN/TEST group
counts and epochs, three arm means and seed values, paired contrasts, and
per-case records. Missing edge arrays remain missing. The optional
`candidate_count` gives the full original candidate-pool size. Optional
`attacked_correct` arrays supply target prediction rings. Do not silently
convert development records into this formal interface.

The current Bento file has no configured **Invite to edit** live room. Its
key-free editable copy and GitHub Pages snapshot do not grant GitHub write
access. Save and review a new presentation snapshot separately if the canonical
presentation needs updating; keep its scientific data tied to the committed
result and verified hash.

## Actual checks still requiring people

Both members should open the final presentation in their intended browser and
the exported PowerPoint file in their intended presentation application.
They should rehearse the complete explanation, check the frozen protocol and
own the final interpretation. Automated layout checks and saved-record audits
do not establish those human actions. Record real confirmations in
[CONTRIBUTIONS.md](../CONTRIBUTIONS.md), without converting this proposed plan
into unverified personal work.
