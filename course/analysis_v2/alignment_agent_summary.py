"""Summarize a COMPLETE saved alignment-agent matrix using the standard library.

This script never imports project/ML code, loads tensor/checkpoint files, makes
network requests, or opens test predictions. Run only after all 18 jobs finish.
It recounts saved validation vectors; it is not a replacement for the separate
strict provenance audit. Output files contain aggregates, group IDs and hashes,
but no raw node IDs, labels, prediction vectors or private absolute paths.

Example (paths supplied by the root agent):
  python root_saved_summary.py --matrix-root MATRIX --protocol FROZEN_JSON \
      --output NEW_SUMMARY_DIR [--bridge-root LOCAL_BRIDGE_DIR] \
      [--resume-dir FIRST_RESUME_DIR --resume-dir SECOND_RESUME_DIR]
"""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import statistics


ARMS = ("legacy_hinge_off", "smooth_off", "smooth_threshold", "smooth_rule",
        "smooth_agent", "smooth_random")
SEEDS = (3, 4, 5)
EXPORT_GROUPS = (957, 1185, 1198, 1249, 1267, 1285, 1328, 1415)
PAIRS = (
    ("smooth_minus_legacy", "smooth_off", "legacy_hinge_off",
     "Attack-loss change within the fresh fixed-eight-epoch harness."),
    ("threshold_minus_smooth", "smooth_threshold", "smooth_off",
     "Unprotected nodewise-q90 alignment after the common warm-up."),
    ("protection_minus_threshold", "smooth_rule", "smooth_threshold",
     "Combined blockwise projection and norm cap, not their separate effects."),
    ("agent_minus_protected_rule", "smooth_agent", "smooth_rule",
     "Agent scheduling versus protected ON every epoch2..8; ON counts may differ."),
    ("agent_minus_matched_random", "smooth_agent", "smooth_random",
     "Agent timing versus a permutation preserving enabled-epoch count, not node activation or cost."),
)
VIEWS = ("raw_predictions", "filtered_predictions", "clean_raw_predictions",
         "clean_filtered_predictions")
TRANSITION_KEYS = ("target_occurrences", "clean_wrong", "attacked_wrong", "new_wrong",
                   "recoveries", "retained_correct", "retained_wrong")
PAIR_METRICS = (
    "raw_mr_percent", "filtered_mr_percent", "raw_new_wrong", "raw_recoveries",
    "filtered_new_wrong", "filtered_recoveries", "group_equal_mean_distance",
    "outside_fraction", "flagged_injected_fraction", "clean_raw_mr_percent",
    "clean_filtered_mr_percent", "clean_filter_new_wrong", "clean_filter_recoveries",
    "utility_raw_accuracy_percent",
    "utility_filtered_accuracy_percent", "utility_raw_benign_fpr_percent",
    "utility_filtered_benign_fpr_percent", "utility_raw_fraud_recall_percent",
    "utility_filtered_fraud_recall_percent", "outer_on_epochs",
    "gradient_calls", "queue_charged_gpu_wall_hours",
)
LIMITS = [
    "Exploratory GCN-surrogate VALID pilot: 128 fixed groups, 1091 target occurrences and 374 unique targets per job; overlapping groups are not independent samples.",
    "Three paired training seeds provide descriptive differences, not a significance test, causal proof or a selected winner.",
    "All arms use final epoch8 without validation checkpoint selection. No GAGA/victim or test efficacy is established.",
    "The directly optimized relationship moments are not an independent camouflage measure or full distribution equality.",
    "The unoptimized anomaly filter is one diagnostic defense; its clean harm and clean-to-attacked transitions must accompany filtered MR.",
    "No learned neural-policy control is included, so the matrix cannot establish LLM superiority over an ordinary learned controller.",
    "Equal optimizer updates do not imply equal compute, wall time or API cost. Matched random preserves enabled epochs, not activated nodes or gradient magnitude.",
    "The executed agent arm may mix remote decisions and registered fallbacks. Infrastructure recovery does not retroactively replace applied actions; a hybrid policy is not pure LLM scheduling.",
    "Every preselected exported group is reported; individual cases were fixed before this matrix and are illustrative, not representative effect estimates.",
    "This saved-record recount does not independently establish the provenance or correctness of all upstream model execution.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate key in saved JSON.")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("Nonfinite JSON constant in a saved record.")


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=no_duplicates, parse_constant=reject_constant)


def read_jsonl(path):
    rows = []
    with pathlib.Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            require(bool(line.strip()), "Blank row in a saved JSONL file.")
            rows.append(json.loads(line, object_pairs_hook=no_duplicates,
                                   parse_constant=reject_constant))
    return rows


def same_number(left, right):
    return (isinstance(left, (int, float)) and isinstance(right, (int, float))
            and math.isfinite(left) and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10))


def mean(values):
    return statistics.mean(values)


def descriptive(values):
    require(len(values) == 3, "Descriptive summaries require all three seeds.")
    require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in values),
            "A summary metric is missing or nonfinite.")
    return dict(n_seeds=3, values=list(values), mean=mean(values),
                sample_sd=statistics.stdev(values), minimum=min(values), maximum=max(values))


def count_transitions(labels, clean, attacked):
    require(len(labels) == len(clean) == len(attacked) and len(labels) > 0,
            "Saved transition denominators differ.")
    result = dict.fromkeys(TRANSITION_KEYS, 0)
    result["target_occurrences"] = len(labels)
    for truth, before, after in zip(labels, clean, attacked):
        cw, aw = before != truth, after != truth
        result["clean_wrong"] += int(cw)
        result["attacked_wrong"] += int(aw)
        key = ("retained_wrong" if cw and aw else "recoveries" if cw else
               "new_wrong" if aw else "retained_correct")
        result[key] += 1
    require(result["attacked_wrong"] - result["clean_wrong"] ==
            result["new_wrong"] - result["recoveries"], "Transition identity failed.")
    return result


def binary_metrics(labels, predictions):
    require(len(labels) == len(predictions) == 128, "Clean utility is not 128 saved cases.")
    require(labels.count(0) == labels.count(1) == 64, "Clean utility classes are not balanced64/64.")
    require(all(value in (0, 1) for value in predictions), "Invalid utility predictions.")
    return dict(cases=128, correct=sum(a == b for a, b in zip(labels, predictions)),
                accuracy=sum(a == b for a, b in zip(labels, predictions)) / 128,
                benign_count=64, fraud_count=64,
                benign_false_positive_rate=sum(p == 1 for y, p in zip(labels, predictions) if y == 0) / 64,
                fraud_recall=sum(p == 1 for y, p in zip(labels, predictions) if y == 1) / 64)


def check_metrics(recorded, recounted, context):
    for key, value in recounted.items():
        require(key in recorded and same_number(recorded[key], value),
                context + " does not reconcile with saved vectors.")


def pct_fraction(numerator, denominator):
    require(denominator > 0, "Empty fraction denominator.")
    return numerator / denominator


def margin_summary(logits):
    require(all(isinstance(row, list) and len(row) == 2 and
                all(isinstance(v, (int, float)) and math.isfinite(v) for v in row)
                for row in logits), "Invalid saved binary logits.")
    values = [row[0] - row[1] for row in logits]
    return dict(mean=mean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values))


def add_inventory(inventory, path, role):
    inventory.append(dict(role=role, sha256=digest_file(path), bytes=pathlib.Path(path).stat().st_size))


def vector_hash(cases, view):
    return canonical_hash([cases[key][view] for key in sorted(cases)])


def check_record_identity(row, arm, seed, protocol_sha):
    require(row.get("arm") == arm and row.get("seed") == seed,
            "Saved method/seed identity changed.")
    if "protocol_sha256" in row:
        require(row["protocol_sha256"] == protocol_sha, "Saved protocol identity changed.")


def load_job(matrix, protocol, protocol_sha, arm, seed, queue_row, inventory):
    name = "%s_seed%d" % (arm, seed)
    job = matrix / name
    required = ("result.json", "registered.json", "initial_model_state.json", "target_groups.json",
                "final_validation.json", "final_validation_groups.json", "final_attack_exports.json",
                "outer_actions.json", "applied_actions.json", "clean_utility.json",
                "raw/final_validation_predictions.jsonl", "raw/clean_utility_predictions.json",
                "raw/clean_utility_manifest.json")
    for relative in required:
        require((job / relative).is_file(), "A required terminal saved artifact is missing.")
        add_inventory(inventory, job / relative, name + "/" + relative)
    result, registered = read_json(job / "result.json"), read_json(job / "registered.json")
    check_record_identity(result, arm, seed, protocol_sha)
    check_record_identity(registered, arm, seed, protocol_sha)
    require(result.get("status") == "completed", "A terminal job did not complete.")
    require(result.get("test_used") is False and result.get("gaga_forwards") == 0,
            "A job reports prohibited test/victim work.")
    require(registered.get("victim_loaded") is False and registered.get("test_used") is False,
            "A registered job reports prohibited test/victim access.")
    require(registered["source_commit"] == protocol["source_commit"] and
            registered["provenance"]["protocol_sha256"] == protocol_sha,
            "A terminal job has a different source/protocol freeze.")
    outcome = result["outcome"]
    for key, expected in (("generator_forwards", 2176), ("gcn_forwards", 2562), ("optimizer_steps", 2048)):
        require(result.get(key) == expected and outcome.get(key) == expected,
                "A terminal generation/GCN/update count differs.")
    require(result.get("completed_train_epochs") == 8 and outcome.get("final_epoch") == 8,
            "The primary terminal epoch is not8.")
    initial = read_json(job / "initial_model_state.json")
    check_record_identity(initial, arm, seed, protocol_sha)
    state_hash = initial["initial_model_state_sha256"]
    require(state_hash == result["initial_model_state_sha256"] == outcome["initial_model_state_sha256"],
            "Initial model hashes do not reconcile.")
    cohorts = read_json(job / "target_groups.json")
    require(cohorts["train"] == protocol["train_groups"] and cohorts["valid"] == protocol["valid_groups"]
            and cohorts["export"] == protocol["export_valid_groups"], "A saved cohort changed.")
    final = read_json(job / "final_validation.json")
    require(final == outcome["final_validation"], "Result and saved terminal validation differ.")
    require(final.get("epoch") == 8 and final.get("groups") == 128 and
            final.get("target_occurrences") == 1091, "Terminal validation denominator differs.")
    group_rows = read_json(job / "final_validation_groups.json")
    predictions = read_jsonl(job / "raw/final_validation_predictions.jsonl")
    require(len(group_rows) == len(predictions) == 128, "Terminal validation omits a group.")
    require([row["target_group"] for row in predictions] == protocol["valid_groups"],
            "Saved group order/membership changed.")
    public_by_group = {row["target_group"]: row for row in group_rows}
    require(len(public_by_group) == 128, "Terminal group summaries are duplicated.")
    cases, labels_by_case, group_ids, case_rows = {}, {}, {}, []
    totals = {condition: collections.Counter() for condition in ("raw", "filtered")}
    clean_filter_totals = collections.Counter()
    for row in predictions:
        group = row["target_group"]
        require(type(group) is int and group in public_by_group, "Invalid saved validation group.")
        ids, labels = row["node_ids"], row["labels"]
        n = len(ids)
        require(n > 0 and len(set(ids)) == n and len(labels) == n and
                row["target_occurrences"] == n, "Invalid within-group saved target IDs.")
        require(all(type(v) is int for v in ids) and all(v == 1 for v in labels),
                "Target partition/labels differ from the registered fraud cohort.")
        for view in VIEWS:
            require(len(row[view]) == n and all(type(v) is int and v in (0, 1) for v in row[view]),
                    "Invalid saved prediction vector.")
        for field in ("raw_logits", "filtered_logits", "clean_raw_logits", "clean_filtered_logits"):
            require(len(row[field]) == n, "Saved logits have a different denominator.")
            margin_summary(row[field])
            predictions_key = field.replace("logits", "predictions")
            require([0 if pair[0] >= pair[1] else 1 for pair in row[field]] == row[predictions_key],
                    "Saved logits and argmax classes disagree.")
        raw = count_transitions(labels, row["clean_raw_predictions"], row["raw_predictions"])
        filtered = count_transitions(labels, row["clean_filtered_predictions"], row["filtered_predictions"])
        clean_filter_changes = count_transitions(labels, row["clean_raw_predictions"],
                                                row["clean_filtered_predictions"])
        clean_filter_totals.update(clean_filter_changes)
        require(row["raw_transitions"] == raw and row["filtered_transitions"] == filtered,
                "Saved group transitions do not reconcile.")
        for condition, recounted in (("raw", raw), ("filtered", filtered)):
            totals[condition].update(recounted)
        require(row["raw_incorrect"] == raw["attacked_wrong"] and
                row["filtered_incorrect"] == filtered["attacked_wrong"], "Saved wrong counts differ.")
        for key, value in public_by_group[group].items():
            require(row.get(key) == value, "Public and private saved group summaries disagree.")
        group_ids[group] = tuple(ids)
        for index, node in enumerate(ids):
            key = (group, node)
            require(key not in cases, "Repeated within-case coordinate.")
            cases[key] = {view: row[view][index] for view in VIEWS}
            labels_by_case[key] = labels[index]
        if group in EXPORT_GROUPS:
            alignment, audit = row["alignment"], row["anomaly_filter"]
            record = dict(arm=arm, seed=seed, target_group=group, target_occurrences=n,
                          raw_incorrect=raw["attacked_wrong"], filtered_incorrect=filtered["attacked_wrong"],
                          raw_residual_correct=n-raw["attacked_wrong"],
                          filtered_residual_correct=n-filtered["attacked_wrong"],
                          node_budget=alignment["node_budget"], edge_budget=alignment["edge_budget"],
                          actual_injected_edges=alignment["actual_undirected_injected_edges"],
                          distance=alignment["mean_distance"], injected_nodes=alignment["injected_nodes"],
                          outside_nodes=alignment["outside_tolerance_nodes"],
                          flagged_injected_nodes=audit["flagged_injected_nodes"],
                          flagged_original_nodes=audit["flagged_original_nodes"],
                          removed_original_original_arcs=audit["removed_original_original_nonself_arcs"],
                          clean_filter_new_wrong=clean_filter_changes["new_wrong"],
                          clean_filter_recoveries=clean_filter_changes["recoveries"],
                          raw_attack_margin_mean=margin_summary(row["raw_logits"])["mean"],
                          filtered_attack_margin_mean=margin_summary(row["filtered_logits"])["mean"],
                          case_selection="All eight preselected export groups; no outcome selection.")
            for condition, values in (("raw", raw), ("filtered", filtered)):
                for field, value in values.items():
                    if field != "target_occurrences":
                        record[condition + "_" + field] = value
            case_rows.append(record)
    require(len(cases) == 1091 and len({node for _, node in cases}) == 374,
            "Full saved case/unique-target denominator differs.")
    require(set(record["target_group"] for record in case_rows) == set(EXPORT_GROUPS),
            "A preselected case is missing.")
    for condition in ("raw", "filtered"):
        require(dict(totals[condition]) == final[condition + "_transitions"],
                "Terminal transition totals do not reconcile.")
        require(same_number(final[condition + "_mr_percent"],
                            100*totals[condition]["attacked_wrong"]/1091), "Terminal MR does not reconcile.")
        require(same_number(final["clean_" + condition + "_mr_percent"],
                            100*totals[condition]["clean_wrong"]/1091), "Terminal clean MR does not reconcile.")
    require(final["raw_predictions_sha256"] == digest_file(job / "raw/final_validation_predictions.jsonl"),
            "Terminal saved prediction checksum differs.")
    exports = read_json(job / "final_attack_exports.json")
    require(len(exports) == 8 and {row["target_group"] for row in exports} == set(EXPORT_GROUPS),
            "The exact eight-group export index is incomplete.")
    for row in exports:
        require(row["arm"] == arm and row["epoch"] == 8 and row["phase"] == "validation"
                and row["protocol_sha256"] == protocol_sha, "An export has another method/phase identity.")
    utility = read_json(job / "clean_utility.json")
    saved_utility = read_json(job / "raw/clean_utility_predictions.json")
    utility_manifest = read_json(job / "raw/clean_utility_manifest.json")
    utility_nodes = utility_manifest["nodes"]
    require(saved_utility["nodes"] == utility_nodes and len({r["node_id"] for r in utility_nodes}) == 128,
            "Utility node coordinates differ.")
    utility_labels = [row["label"] for row in utility_nodes]
    for condition in ("raw", "filtered"):
        recounted = binary_metrics(utility_labels, saved_utility[condition + "_predictions"])
        check_metrics(utility[condition], recounted, "Clean utility")
    require(utility == outcome["clean_utility"] and utility.get("agent_feedback") is False,
            "Utility was changed or reports controller feedback.")
    actions = read_json(job / "outer_actions.json")
    check_record_identity(actions, arm, seed, protocol_sha)
    flags = actions["actions"]
    require(len(flags) == 8 and all(type(flag) is bool for flag in flags) and not flags[0],
            "Applied epoch actions are incomplete or warm-up changed.")
    require(flags == outcome["actions"], "Saved applied action sequences differ.")
    if arm in ("legacy_hinge_off", "smooth_off"):
        require(not any(flags), "An OFF arm enabled alignment.")
    if arm in ("smooth_threshold", "smooth_rule"):
        require(flags == [False] + [True]*7, "A fixed-ON control schedule changed.")
    decisions = read_json(job / "applied_actions.json")
    require(decisions["actions"] == flags and len(decisions["decisions"]) == 8,
            "Applied decision receipts do not reconcile.")
    fallback_count, wait_seconds, sources = 0, 0.0, collections.Counter()
    decision_rows = []
    for epoch, receipt in enumerate(decisions["decisions"], 1):
        require(receipt["epoch"] == epoch and receipt["decision"]["align"] == flags[epoch-1],
                "An applied epoch decision differs.")
        decision = receipt["decision"]
        if arm == "smooth_agent" and epoch > 1:
            controller_path = job / ("applied_controller_epoch_%02d.json" % epoch)
            applied_controller = read_json(controller_path)
            add_inventory(inventory, controller_path,
                          name + "/applied_controller_epoch_%02d.json" % epoch)
            require(applied_controller == decision,
                    "Server-applied controller receipt differs from the terminal action record.")
        fallback_count += int(decision["fallback"])
        wait_seconds += decision.get("wait_seconds", 0.0)
        source = decision.get("decision_source", "not_recorded")
        require(isinstance(source, str) and source.replace("_", "").isalnum(),
                "Unsafe/unexpected decision-source value.")
        sources[source] += 1
        decision_rows.append(dict(arm=arm, seed=seed, epoch=epoch, align=flags[epoch-1],
                                  fallback=decision["fallback"], decision_source=source,
                                  wait_seconds=decision.get("wait_seconds", 0.0)))
    expected_gradients = 128 + (496*sum(flags) if arm in ("smooth_rule", "smooth_agent", "smooth_random") else 0)
    require(result["gradient_calls"] == outcome["gradient_calls"] == expected_gradients,
            "Actual gradient call count differs from its registered action-dependent rule.")
    injected = sum(row["alignment"]["injected_nodes"] for row in group_rows)
    outside = sum(row["alignment"]["outside_tolerance_nodes"] for row in group_rows)
    flagged = sum(row["anomaly_filter"]["flagged_injected_nodes"] for row in group_rows)
    require(injected == final["injected_nodes"] and outside == final["outside_tolerance_nodes"]
            and flagged == final["flagged_injected_nodes"], "Saved alignment/filter totals differ.")
    require(same_number(final["group_equal_mean_distance"],
                        mean([row["alignment"]["mean_distance"] for row in group_rows])),
            "Group-equal distance mean differs.")
    metrics = dict(arm=arm, seed=seed, groups=128, target_occurrences=1091,
                   unique_targets=374, repeated_occurrences=717,
                   raw_mr_percent=final["raw_mr_percent"], filtered_mr_percent=final["filtered_mr_percent"],
                   clean_raw_mr_percent=final["clean_raw_mr_percent"],
                   clean_filtered_mr_percent=final["clean_filtered_mr_percent"],
                   clean_filter_net_extra_wrong=(totals["filtered"]["clean_wrong"]-totals["raw"]["clean_wrong"]),
                   clean_filter_new_wrong=clean_filter_totals["new_wrong"],
                   clean_filter_recoveries=clean_filter_totals["recoveries"],
                   group_equal_mean_distance=final["group_equal_mean_distance"],
                   injected_nodes=injected, outside_nodes=outside, outside_fraction=pct_fraction(outside, injected),
                   flagged_injected_nodes=flagged, flagged_injected_fraction=pct_fraction(flagged, injected),
                   flagged_original_node_exposures=final["flagged_original_node_exposures"],
                   removed_original_original_arcs=final["removed_original_original_nonself_arcs"],
                   outer_on_epochs=sum(flags), controller_fallback_epochs=fallback_count,
                   controller_wait_seconds=wait_seconds, gradient_calls=result["gradient_calls"],
                   generator_forwards=result["generator_forwards"], gcn_forwards=result["gcn_forwards"],
                   optimizer_steps=result["optimizer_steps"], job_elapsed_seconds=result["elapsed_seconds"],
                   inner_charged_gpu_wall_hours=result["charged_gpu_wall_hours"],
                   queue_elapsed_seconds=queue_row["elapsed_process_seconds"],
                   queue_charged_gpu_wall_hours=queue_row["charged_gpu_wall_hours"],
                   peak_reserved_bytes=result.get("peak_reserved_bytes"),
                   initial_model_state_sha256=state_hash,
                   applied_actions="".join("1" if value else "0" for value in flags),
                   decision_source_counts=json.dumps(dict(sources), sort_keys=True, separators=(",", ":")))
    for condition in ("raw", "filtered"):
        for field in TRANSITION_KEYS:
            if field != "target_occurrences":
                metrics[condition + "_" + field] = totals[condition][field]
        for field, short in (("accuracy", "accuracy"), ("benign_false_positive_rate", "benign_fpr"),
                             ("fraud_recall", "fraud_recall")):
            metrics["utility_" + condition + "_" + short + "_percent"] = 100*utility[condition][field]
    return dict(metrics=metrics, cases=cases, labels=labels_by_case, group_ids=group_ids,
                utility_nodes=utility_nodes, case_rows=case_rows, decision_rows=decision_rows,
                actions=flags, prediction_vector_hashes={view: vector_hash(cases, view) for view in VIEWS})


def summarize_api(bridge, ledger_arg, resume_args, protocol_sha, inventory, applied_agent_decisions):
    applied_sources = collections.Counter(row["decision_source"] for row in applied_agent_decisions)
    applied_summary = dict(
        slots=len(applied_agent_decisions),
        source_counts=dict(applied_sources),
        fallback_epochs=sum(row["fallback"] for row in applied_agent_decisions),
        enabled_epochs=sum(row["align"] for row in applied_agent_decisions),
        external_agent_epochs=sum(row["decision_source"] == "external_agent" and not row["fallback"]
                                  for row in applied_agent_decisions),
        authority="Actual server applied_controller receipts for epochs2..8; API completion/publication does not prove application.")
    require(applied_summary["slots"] == 21, "Applied agent decision slots are incomplete.")
    if bridge is None and ledger_arg is None:
        require(not resume_args, "Resume receipts require their original bridge directory.")
        return dict(status="not_provided", physical_attempts_recorded=None,
                    charged_usd_upper_bound=None,
                    applied_policy=applied_summary,
                    limitation="API cost was not supplied; missing cost is not zero.")
    require(bridge is not None or not resume_args, "Resume receipts require their original bridge directory.")
    result = dict(status="saved_shared_ledger_recount", applied_policy=applied_summary)
    ledger = pathlib.Path(ledger_arg) if ledger_arg else bridge / "api_attempts.jsonl"
    if bridge is not None:
        require(ledger.resolve() == (bridge / "api_attempts.jsonl").resolve(),
                "Recovery accounting must use the same original bridge ledger.")
        if resume_args:
            resume_dirs = [pathlib.Path(value).resolve() for value in resume_args]
        else:
            # Recovery outputs are private siblings of api_attempt1 by default.
            resume_dirs = [path.resolve() for path in bridge.parent.glob("resume_attempt*")
                           if path.is_dir() and path.name[len("resume_attempt"):].isdigit()]
        require(len(set(resume_dirs)) == len(resume_dirs) and bridge not in resume_dirs,
                "A bridge/recovery directory was supplied twice.")
        attempts = []
        for directory in [bridge] + resume_dirs:
            registered = read_json(directory / "registered.json")
            terminal = read_json(directory / "bridge_complete.json")
            require(registered["protocol_sha256"] == terminal["protocol_sha256"] == protocol_sha,
                    "Saved bridge/recovery protocol differs.")
            started = registered.get("started_utc", registered.get("registered_utc"))
            require(isinstance(started, str), "Bridge/recovery start timestamp is missing.")
            stamp = dt.datetime.fromisoformat(started.replace("Z", "+00:00"))
            require(stamp.tzinfo is not None, "Bridge/recovery timestamp is not timezone-aware.")
            status = terminal["status"]
            require(isinstance(status, str) and status.replace("_", "").isalnum(),
                    "Unsafe/unexpected bridge/recovery status.")
            require(terminal.get("unique_epoch_slots", 0) <= 21, "API logical slot cap exceeded.")
            attempts.append((stamp, directory, registered, terminal))
        attempts.sort(key=lambda value: value[0])
        require(attempts[0][1] == bridge, "A recovery predates the original bridge registration.")
        original_prompt = attempts[0][2]["prompt_sha256"]
        original_caps = attempts[0][2]["caps"]
        chronology, uncertain_attempt_bounds = [], []
        for index, (_, directory, registered, terminal) in enumerate(attempts):
            role = "api_bridge_original" if directory == bridge else "api_bridge_recovery_%d" % index
            require(registered["prompt_sha256"] == original_prompt and registered["caps"] == original_caps,
                    "A recovery changed the prompt or original aggregate caps.")
            add_inventory(inventory, directory / "registered.json", role + "/registered.json")
            add_inventory(inventory, directory / "bridge_complete.json", role + "/bridge_complete.json")
            amendment = directory / "infrastructure_amendment.json"
            if amendment.is_file():
                add_inventory(inventory, amendment, role + "/infrastructure_amendment.json")
            usage = terminal.get("usage", {})
            allowed_usage = ("physical_attempts_recorded", "physical_attempts_combined",
                             "charged_prompt_tokens", "charged_prompt_tokens_combined",
                             "charged_completion_tokens", "charged_completion_tokens_combined",
                             "charged_usd_upper_bound", "charged_usd_upper_bound_combined",
                             "unrecorded_attempt_upper_bound")
            snapshot = {key: usage[key] for key in allowed_usage if key in usage}
            require(all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
                        for value in snapshot.values()), "Invalid bridge/recovery usage snapshot.")
            if "unrecorded_attempt_upper_bound" in snapshot:
                uncertain_attempt_bounds.append(snapshot["unrecorded_attempt_upper_bound"])
            chronology.append(dict(attempt=index, role=role, status=terminal["status"],
                started_utc=registered.get("started_utc", registered.get("registered_utc")),
                completed_utc=terminal.get("completed_utc"),
                processed_requests=terminal.get("processed_requests"),
                unique_epoch_slots=terminal.get("unique_epoch_slots"),
                receipt_usage_snapshot_not_additive=snapshot))
        result.update(status="saved_shared_ledger_recount_with_recovery_history",
                      bridge_attempt_chronology=chronology,
                      final_bridge_attempt_status=chronology[-1]["status"],
                      recovery_attempts=len(chronology)-1, prompt_sha256=original_prompt,
                      unrecorded_attempt_upper_bound_summed_from_attempt_receipts=(
                          sum(uncertain_attempt_bounds) if uncertain_attempt_bounds else None),
                      chronology_accounting="Historical failed receipts remain separate. Attempt snapshots refer to the same evolving ledger and are never summed as physical usage.")
    require(ledger.is_file(), "The supplied API ledger is missing.")
    add_inventory(inventory, ledger, "api_bridge/api_attempts.jsonl")
    rows = read_jsonl(ledger)
    attempted = [row for row in rows if row.get("attempted") is True]
    for row in rows:
        if "protocol_sha256" in row:
            require(row["protocol_sha256"] == protocol_sha, "An API ledger record has another protocol.")
    result.update(physical_attempts_recorded=len(attempted),
                  charged_prompt_tokens=sum(row.get("charged_prompt_tokens", 0) for row in attempted),
                  charged_completion_tokens=sum(row.get("charged_completion_tokens", 0) for row in attempted),
                  charged_usd_upper_bound=sum(row.get("charged_usd_upper_bound", 0) for row in attempted),
                  accounting="Physical usage is recounted once from the original shared ledger. Late/fallback responses remain charged; bridge completion is not proof of application; no credentials or response text copied.")
    return result


def write_json(path, value):
    with pathlib.Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    require(bool(rows), "A public CSV table is empty.")
    fields = list(rows[0])
    require(all(set(row) == set(fields) for row in rows), "CSV row fields differ.")
    with pathlib.Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bridge-root")
    parser.add_argument("--api-ledger")
    parser.add_argument("--resume-dir", action="append", default=[],
                        help="Repeat for preserved recovery outputs; default discovers sibling resume_attemptN directories.")
    args = parser.parse_args()
    matrix, protocol_path, output = (pathlib.Path(value).resolve()
                                     for value in (args.matrix_root, args.protocol, args.output))
    require(not output.exists(), "The output must be a new exclusive directory.")
    inventory = []
    protocol = read_json(protocol_path)
    protocol_sha = digest_file(protocol_path)
    add_inventory(inventory, protocol_path, "execution_protocol")
    require(protocol.get("status") == "frozen" and protocol.get("owner_execution_authorized") is True,
            "The input is not the frozen authorized execution protocol.")
    require(tuple(protocol["arms"]) == ARMS and tuple(protocol["seeds"]) == SEEDS
            and protocol["epochs"] == 8, "The frozen matrix/terminal epoch differs.")
    require(len(protocol["valid_groups"]) == 128 and len(set(protocol["valid_groups"])) == 128
            and len(protocol["train_groups"]) == 256, "The frozen cohort sizes differ.")
    require(len(protocol["export_valid_groups"]) == 8 and set(protocol["export_valid_groups"]) == set(EXPORT_GROUPS),
            "The preregistered eight illustrative groups differ.")
    complete_path = matrix / "queue_complete.json"
    require(complete_path.is_file(), "The matrix has no terminal complete queue receipt; do not summarize partial jobs.")
    queue = read_json(complete_path)
    add_inventory(inventory, complete_path, "queue_complete.json")
    require(queue.get("status") == "completed" and queue.get("protocol_sha256") == protocol_sha,
            "The queue did not complete this exact protocol.")
    require(queue.get("test_queries") == 0 and queue.get("gaga_forwards") == 0,
            "The terminal queue reports prohibited test/victim work.")
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    queue_rows = {(row["arm"], row["seed"]): row for row in queue["jobs"]}
    require(len(queue["jobs"]) == len(queue_rows) == 18 and set(queue_rows) == expected
            and all(row["exit_code"] == 0 for row in queue_rows.values()),
            "The queue is not exactly eighteen successful arm/seed jobs.")
    jobs = {}
    for seed in SEEDS:
        for arm in ARMS:
            jobs[(arm, seed)] = load_job(matrix, protocol, protocol_sha, arm, seed,
                                          queue_rows[(arm, seed)], inventory)
    baseline = jobs[(ARMS[0], SEEDS[0])]
    for key, job in jobs.items():
        require(job["labels"] == baseline["labels"] and job["group_ids"] == baseline["group_ids"],
                "Case coordinates/labels or within-group node order differ across matrix jobs.")
        require(job["utility_nodes"] == baseline["utility_nodes"], "Utility coordinates differ across jobs.")
        for view in ("clean_raw_predictions", "clean_filtered_predictions"):
            require(all(job["cases"][case][view] == baseline["cases"][case][view]
                        for case in baseline["cases"]), "Fixed clean reference predictions differ across jobs.")
    for seed in SEEDS:
        require(len({jobs[(arm, seed)]["metrics"]["initial_model_state_sha256"] for arm in ARMS}) == 1,
                "The six same-seed arms did not start from identical model weights.")
        require(sum(jobs[("smooth_agent", seed)]["actions"]) ==
                sum(jobs[("smooth_random", seed)]["actions"]), "Random control does not match enabled epochs.")
        schedule_path = matrix / ("matched_schedule_seed%d.json" % seed)
        schedule = read_json(schedule_path)
        add_inventory(inventory, schedule_path, "matched_schedule_seed%d.json" % seed)
        require(schedule["actions"] == jobs[("smooth_random", seed)]["actions"] and
                schedule["source_enabled_epochs"] == sum(jobs[("smooth_agent", seed)]["actions"])
                and schedule["protocol_sha256"] == protocol_sha,
                "Derived random schedule and actual applications differ.")
    paired_rows, paired_summaries, identity_rows = [], [], []
    for name, treatment, control, interpretation in PAIRS:
        differences = {metric: [] for metric in PAIR_METRICS}
        for seed in SEEDS:
            treated, reference = jobs[(treatment, seed)], jobs[(control, seed)]
            row = dict(comparison=name, treatment=treatment, control=control, seed=seed,
                       interpretation=interpretation)
            for metric in PAIR_METRICS:
                delta = treated["metrics"][metric] - reference["metrics"][metric]
                differences[metric].append(delta)
                row[metric + "_difference"] = delta
            paired_rows.append(row)
            keys = sorted(reference["cases"])
            require(keys == sorted(treated["cases"]), "Paired prediction case coordinates differ.")
            for view in VIEWS:
                differing = [case for case in keys if treated["cases"][case][view] != reference["cases"][case][view]]
                differing_groups = {group for group, _ in differing}
                wrong_t = sum(treated["cases"][case][view] != treated["labels"][case] for case in keys)
                wrong_c = sum(reference["cases"][case][view] != reference["labels"][case] for case in keys)
                identity_rows.append(dict(comparison=name, seed=seed, view=view,
                    case_coordinates_identical=True, target_occurrences=1091, groups=128,
                    identical_occurrences=1091-len(differing), differing_occurrences=len(differing),
                    identical_groups=128-len(differing_groups), differing_groups=len(differing_groups),
                    complete_prediction_vector_identical=not differing,
                    aggregate_wrong_counts_equal=wrong_t == wrong_c,
                    treatment_vector_sha256=treated["prediction_vector_hashes"][view],
                    control_vector_sha256=reference["prediction_vector_hashes"][view],
                    ordering="Sorted private (target_group,node_id) coordinates; no IDs are released."))
        paired_summaries.append(dict(comparison=name, treatment=treatment, control=control,
                                     interpretation=interpretation,
                                     differences={metric: descriptive(values) for metric, values in differences.items()}))
    bridge = pathlib.Path(args.bridge_root).resolve() if args.bridge_root else None
    applied_agent_decisions = [row for seed in SEEDS for row in jobs[("smooth_agent", seed)]["decision_rows"]
                               if row["epoch"] > 1]
    api = summarize_api(bridge, args.api_ledger, args.resume_dir, protocol_sha, inventory,
                        applied_agent_decisions)
    job_rows = [jobs[(arm, seed)]["metrics"] for arm in ARMS for seed in SEEDS]
    case_rows = sorted([row for job in jobs.values() for row in job["case_rows"]],
                       key=lambda row: (ARMS.index(row["arm"]), row["seed"], row["target_group"]))
    decision_rows = sorted([row for job in jobs.values() for row in job["decision_rows"]],
                           key=lambda row: (ARMS.index(row["arm"]), row["seed"], row["epoch"]))
    arm_summaries = [dict(arm=arm, metrics={metric: descriptive([jobs[(arm, seed)]["metrics"][metric]
                        for seed in SEEDS]) for metric in PAIR_METRICS}) for arm in ARMS]
    cost = dict(queue_gpu_hours_including_diagnostic=queue["charged_gpu_hours_including_diagnostic"],
                recorded_job_gpu_hours=sum(row["charged_gpu_wall_hours"] for row in queue_rows.values()),
                registered_diagnostic_gpu_hours=protocol.get("diagnostic_gpu_hours"),
                registered_phase_gpu_ceiling=protocol["phase_gpu_hours"], api=api,
                accounting="Queue uses conservative whole-process GPU wall time including waits; API usage is separate.")
    require(cost["queue_gpu_hours_including_diagnostic"] <= protocol["phase_gpu_hours"] + 1e-10,
            "The reported phase GPU ceiling was exceeded.")
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(status="complete_saved_record_summary", created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   scope="Exploratory GCN-surrogate VALID only; no new inference, training, API call, GAGA/victim or test evaluation.",
                   source_commit=protocol["source_commit"], protocol_sha256=protocol_sha,
                   summary_script_sha256=digest_file(__file__), jobs=18, arms=list(ARMS), seeds=list(SEEDS),
                   groups_per_job=128, target_occurrences_per_job=1091, unique_targets_per_job=374,
                   repeated_occurrences_per_job=717, preselected_export_groups=list(EXPORT_GROUPS),
                   distance_weighting="Equal validation-group means; outside/flagged fractions use injected-node exposures.",
                   paired_difference_direction="Treatment minus control; MR and utility percentages are percentage-point differences.",
                   transition_interpretation="Each raw attack is compared with the original clean graph; each filtered attack with the once-filtered original graph. Attacked_wrong-clean_wrong=new_wrong-recoveries.",
                   prediction_identity="Class vectors are compared at identical private (target_group,node_id) coordinates, never inferred from equal aggregate counts.",
                   arm_summaries=arm_summaries, paired_descriptive_differences=paired_summaries,
                   cost=cost, limitations=LIMITS, public_safe=True)
    write_json(output / "summary.json", summary)
    write_csv(output / "jobs.csv", job_rows)
    write_csv(output / "paired_seed_differences.csv", paired_rows)
    write_csv(output / "prediction_vector_identity.csv", identity_rows)
    write_csv(output / "preselected_cases.csv", case_rows)
    write_csv(output / "applied_epoch_actions.csv", decision_rows)
    write_json(output / "input_hashes.json", dict(protocol_sha256=protocol_sha,
                summary_script_sha256=digest_file(__file__), inputs=inventory,
                privacy="Only logical roles, hashes and byte counts; no private paths, raw IDs or response content."))
    output_hashes = [dict(file=path.name, sha256=digest_file(path), bytes=path.stat().st_size)
                     for path in sorted(output.iterdir()) if path.is_file()]
    write_json(output / "output_hashes.json", output_hashes)
    print("Complete saved-record summary written: 18 jobs; 128 groups/1091 occurrences/374 unique targets each.")


if __name__ == "__main__":
    main()
