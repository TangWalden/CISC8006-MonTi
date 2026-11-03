"""Post-hoc, stdlib-only recount of PRIVATE saved validation predictions.

No dataset, model, checkpoint, test prediction, GPU, or API is loaded. Run only
after this source is committed. Outputs contain private group/node identities
and must remain outside the public repository. Best-checkpoint comparisons are
between separately trained arms, not fixed-weight causal interventions.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import pathlib
import statistics
import subprocess


ARMS = ("off", "always", "gated")
SEEDS = (3, 4, 5)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def distribution(values):
    values = list(values)
    if not values:
        return {"n": 0}
    return dict(n=len(values), minimum=min(values), median=statistics.median(values),
                mean=statistics.mean(values), maximum=max(values))


def margin(row, index):
    logits = row["logits"][index]
    label = row["labels"][index]
    return logits[label] - max(value for j, value in enumerate(logits) if j != label)


def group_facts(row):
    fields = ("target_group", "target_occurrences", "incorrect", "gcn_correct_after_attack",
              "node_budget", "edge_budget", "actual_undirected_injected_edges",
              "mean_distance", "max_distance", "mean_tolerance", "active_nodes",
              "outside_tolerance_nodes", "injected_nodes", "alignment_penalty",
              "base_attack_loss", "total_loss", "physical_degree_min", "physical_degree_max",
              "fallback_nodes")
    result = {key: row[key] for key in fields}
    result["correct_class_logit_margin"] = distribution(margin(row, i) for i in range(len(row["node_ids"])))
    result["active_fraction"] = row["active_nodes"] / row["injected_nodes"]
    result["outside_fraction"] = row["outside_tolerance_nodes"] / row["injected_nodes"]
    return result


def transitions(left, right, include_private_nodes=False):
    assert left["target_group"] == right["target_group"]
    assert left["node_ids"] == right["node_ids"] and left["labels"] == right["labels"]
    result = collections.Counter()
    changed = []
    for i, (label, before, after) in enumerate(zip(left["labels"], left["predictions"], right["predictions"])):
        old_correct, new_correct = before == label, after == label
        name = ("correct" if old_correct else "wrong") + "_to_" + ("correct" if new_correct else "wrong")
        result[name] += 1
        if before != after:
            item = dict(occurrence_index=i, transition=name,
                        off_margin=margin(left, i), candidate_margin=margin(right, i))
            if include_private_nodes:
                item["private_node_id"] = left["node_ids"][i]
            changed.append(item)
    for key in ("correct_to_correct", "correct_to_wrong", "wrong_to_correct", "wrong_to_wrong"):
        result.setdefault(key, 0)
    result["net_extra_residual_correct"] = result["wrong_to_correct"] - result["correct_to_wrong"]
    assert result["net_extra_residual_correct"] == right["gcn_correct_after_attack"] - left["gcn_correct_after_attack"]
    return dict(result), changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=pathlib.Path, required=True)
    parser.add_argument("--raw-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    source = pathlib.Path(__file__).resolve()
    repo = source.parents[2]
    relative_source = source.relative_to(repo).as_posix()
    def git(*command):
        return subprocess.check_output(["git", "-C", str(repo), *command], text=True).strip()
    head = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--", relative_source):
        raise RuntimeError("Analysis source must be committed and unchanged before the recorded recount.")
    if git("hash-object", str(source)) != git("rev-parse", "HEAD:" + relative_source):
        raise RuntimeError("Working analysis source does not match its committed Git object.")
    snapshot, raw_root = args.snapshot.resolve(), args.raw_root.resolve()
    output = args.output.resolve()
    if output == repo or repo in output.parents:
        raise RuntimeError("Private output must be outside the public repository.")
    results, manifest, metadata = {}, {}, {}
    def remember(path):
        manifest[str(path)] = digest(path)
        return load(path)
    for arm in ARMS:
        for seed in SEEDS:
            job = "%s_seed%d" % (arm, seed)
            result = remember(snapshot / job / "result.json")
            registration = remember(snapshot / job / "registered.json")
            assert result["status"] == "completed" and not result["test_used"] and not result["victim_loaded"]
            results[arm, seed] = result
            metadata[job] = dict(best_epoch=result["best_epoch"], completed_epochs=result["completed_epochs"],
                                 original_source_commit=registration["source_commit"],
                                 original_protocol_sha256=registration["protocol_sha256"],
                                 checkpoint_sha256=result["checkpoint_sha256"])
            for epoch in range(1, result["completed_epochs"] + 1):
                stem = "validation_epoch_%03d" % epoch
                remember(snapshot / job / (stem + ".json"))
                remember(snapshot / job / (stem + "_groups.json"))
                raw = raw_root / job / (stem + ".jsonl")
                manifest[str(raw)] = digest(raw)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "registered.json", dict(
        analysis_type="POST-HOC saved validation-record recount; not preregistered scientific hypothesis",
        registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        analysis_source_commit=head, analysis_source_sha256=digest(source),
        inputs=manifest, jobs=metadata,
        ranking="Sum wrong-to-correct under attack across seeds, descending; numeric group ID ascending on ties",
        success_ranking="Sum correct-to-wrong under attack across seeds, descending; numeric group ID ascending on ties",
        scope="All retained validation groups/occurrences, selected-best and every common epoch; no clean/test/model calls",
        independence="Same group/node identity does not imply identical weights, generated features, endpoints, or RNG across arms"))
    records, epoch_checks = {}, []
    for (arm, seed), result in results.items():
        job = "%s_seed%d" % (arm, seed)
        for epoch in range(1, result["completed_epochs"] + 1):
            stem = "validation_epoch_%03d" % epoch
            summary = load(snapshot / job / (stem + ".json"))
            public = load(snapshot / job / (stem + "_groups.json"))
            raw_path = raw_root / job / (stem + ".jsonl")
            assert digest(raw_path) == summary["raw_prediction_sha256"]
            raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert len(raw) == len(public) == summary["groups"] == 128
            groups = {}
            for private, row in zip(raw, public):
                assert all(private[key] == value for key, value in row.items())
                group = row["target_group"]
                assert group not in groups
                groups[group] = private
                n = row["target_occurrences"]
                assert len(private["node_ids"]) == len(set(private["node_ids"])) == n
                assert len(private["labels"]) == len(private["predictions"]) == len(private["logits"]) == n
                assert all(label in (0, 1) for label in private["labels"])
                for logits, prediction in zip(private["logits"], private["predictions"]):
                    assert len(logits) == 2 and all(math.isfinite(x) for x in logits)
                    assert prediction == max(range(len(logits)), key=lambda j: logits[j])
                wrong = sum(label != prediction for label, prediction in zip(private["labels"], private["predictions"]))
                assert wrong == row["incorrect"] and n - wrong == row["gcn_correct_after_attack"]
                assert row["actual_undirected_injected_edges"] == row["edge_budget"]
                assert row["injected_nodes"] == row["node_budget"] > 0
            occurrences = sum(x["target_occurrences"] for x in raw)
            wrong = sum(x["incorrect"] for x in raw)
            assert occurrences == summary["target_occurrences"] == 1091 and wrong == summary["incorrect"]
            assert math.isclose(summary["mr_percent"], 100 * wrong / occurrences, abs_tol=1e-12)
            records[arm, seed, epoch] = groups
            epoch_checks.append(dict(arm=arm, seed=seed, epoch=epoch, groups=len(raw), occurrences=occurrences,
                                     incorrect=wrong, residual_correct=occurrences-wrong))
    selected, same_epoch, details = [], [], {}
    for arm in ("always", "gated"):
        for seed in SEEDS:
            off_epoch = results["off", seed]["best_epoch"]
            candidate_epoch = results[arm, seed]["best_epoch"]
            for kind, left_epoch, right_epoch in [("selected_best", off_epoch, candidate_epoch)] + [
                    ("same_epoch", epoch, epoch) for epoch in range(1, min(results["off", seed]["completed_epochs"], results[arm, seed]["completed_epochs"]) + 1)]:
                left, right = records["off", seed, left_epoch], records[arm, seed, right_epoch]
                assert set(left) == set(right)
                totals = collections.Counter()
                rows = []
                for group in sorted(left):
                    counts, changed = transitions(left[group], right[group], include_private_nodes=True)
                    totals.update(counts)
                    rows.append(dict(private_group_id=group, **counts,
                                     off=group_facts(left[group]), candidate=group_facts(right[group]),
                                     changed_occurrences=changed))
                summary = dict(kind=kind, candidate_arm=arm, seed=seed, off_epoch=left_epoch,
                               candidate_epoch=right_epoch, groups=len(left), target_occurrences=1091,
                               off_residual_correct=sum(x["gcn_correct_after_attack"] for x in left.values()),
                               candidate_residual_correct=sum(x["gcn_correct_after_attack"] for x in right.values()), **totals)
                (selected if kind == "selected_best" else same_epoch).append(summary)
                details[arm, seed, kind, right_epoch] = rows
    ranked, case_studies = {}, {}
    for arm in ("always", "gated"):
        accum = collections.defaultdict(collections.Counter)
        for seed in SEEDS:
            epoch = results[arm, seed]["best_epoch"]
            for row in details[arm, seed, "selected_best", epoch]:
                accum[row["private_group_id"]].update({key: row[key] for key in (
                    "wrong_to_correct", "correct_to_wrong", "net_extra_residual_correct")})
        failure_order = sorted(accum, key=lambda group: (-accum[group]["wrong_to_correct"], group))
        success_order = sorted(accum, key=lambda group: (-accum[group]["correct_to_wrong"], group))
        ranked[arm] = [dict(private_group_id=group, **accum[group]) for group in failure_order]
        wanted = [("failure_rank_%d" % (i+1), group) for i, group in enumerate(failure_order[:3])]
        wanted += [("success_counterexample", success_order[0])]
        case_studies[arm] = []
        for role, group in wanted:
            per_seed = []
            for seed in SEEDS:
                epoch = results[arm, seed]["best_epoch"]
                row = next(row for row in details[arm, seed, "selected_best", epoch] if row["private_group_id"] == group)
                per_seed.append(dict(seed=seed, off_epoch=results["off", seed]["best_epoch"],
                                     candidate_epoch=epoch, **row))
            case_studies[arm].append(dict(role=role, private_group_id=group, aggregate=dict(accum[group]), per_seed=per_seed))
    first = records["off", SEEDS[0], 1]
    all_ids = [node for row in first.values() for node in row["node_ids"]]
    write(output / "failure_analysis_private.json", dict(
        status="passed", scientific_status="Exploratory post-hoc validation analysis; no new efficacy experiment",
        epoch_checks=epoch_checks, selected_best=selected, same_epoch=same_epoch,
        group_rankings=ranked, case_studies=case_studies,
        coverage=dict(groups=len(first), occurrences=len(all_ids), unique_node_ids=len(set(all_ids)),
                      repeated_occurrences=len(all_ids)-len(set(all_ids)),
                      labels=sorted({label for row in first.values() for label in row["labels"]})),
        limitations=["Residual correct under attack is not a clean-to-attacked flip count.",
                     "No clean baseline was loaded; no intrinsic difficulty or clean correctness claim is made.",
                     "Group-overlapping target IDs are separate attack scenarios and not independent news samples.",
                     "Selected-best checkpoints and equal epoch comparisons still differ in trained weights, features and RNG trajectories.",
                     "D, tolerance, activation and margin associations do not identify a causal mechanism.",
                     "Retained group alignment summaries do not expose per-injected-node D/tolerance pairing or gradients."]))
    write(output / "completion.json", dict(status="passed", epoch_records=len(epoch_checks),
          input_files=len(manifest), output_sha256=digest(output / "failure_analysis_private.json"),
          new_model_calls=0, api_calls=0, new_training_steps=0, test_predictions_read=False))
    print(json.dumps(dict(status="passed", output=str(output), epoch_records=len(epoch_checks),
                          groups=128, occurrences=1091, new_model_calls=0, api_calls=0)))


if __name__ == "__main__":
    main()
