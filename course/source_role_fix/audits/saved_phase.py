"""Independent saved-record audit of the six-job source-role comparison.

Standard library only. Never deserialize tensors, datasets, graphs or models;
checkpoint/export files are checked as bytes. The auditor did not author the
model, training, queue or evaluation modules. The auditor DID author the optional
saved-teacher companion, whose receipt consistency is not an independent review
of its implementation. Freeze this file's SHA256 before any new outcome read.
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import pathlib
import statistics
import sys
import time
import traceback


ARMS = ("legacy", "source_fixed")
SEEDS = (3, 4, 5)
SOURCE = "dad856c9d2697d887bc6eeea5dd26fab432fe682"
PREPARATION_SHA = "b8246361e67cc20cc5d889988271bd03c3ad6e5408d84d45b7d2be1e08a8f8ff"
TRANSITION_KEYS = ("occurrences", "clean_wrong", "attacked_wrong", "new_wrong", "recoveries")
TCRIT_DF2 = math.sqrt(2) * .95 / math.sqrt(1 - .95 ** 2)


def digest(path):
    result = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    with pathlib.Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


class Audit:
    def __init__(self, root):
        self.root = root
        self.inputs = {}
        self.checks = collections.Counter()
        self.warnings = []

    def path(self, value, directory=False):
        raw = pathlib.Path(value)
        path = raw.resolve() if raw.is_absolute() else (self.root / raw).resolve()
        self.check(self.root in path.parents, "repository_path_boundary")
        self.check(path.is_dir() if directory else path.is_file(), "input_exists")
        return path

    def check(self, condition, name):
        self.checks[name] += 1
        if not condition:
            raise AssertionError("Audit check failed: " + name)

    def bytes(self, value, wanted=None):
        path = self.path(value)
        measured = digest(path)
        previous = self.inputs.setdefault(str(path.relative_to(self.root)), measured)
        self.check(previous == measured, "input_unchanged_during_audit")
        if wanted is not None:
            self.check(measured == wanted, "recorded_byte_sha256")
        return path

    def json(self, value, wanted=None):
        path = self.bytes(value, wanted)
        return json.loads(path.read_text(encoding="utf-8"))

    def jsonl(self, value):
        path = self.bytes(value)
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    def close(self, actual, expected, name, atol=1e-9):
        self.check(math.isfinite(actual) and math.isfinite(expected)
                   and abs(actual - expected) <= atol * max(1., abs(expected)), name)


def case_key(seed, phase, epoch, group):
    text = "%d|%s|%d|%d|dual-surrogate-v1" % (seed, phase, epoch, group)
    return int(hashlib.sha256(text.encode("ascii")).hexdigest()[:8], 16)


def classify(audit, logits, count):
    audit.check(len(logits) == count, "raw_logit_denominator")
    for values in logits:
        audit.check(len(values) == 2 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in values),
                    "finite_binary_raw_logits")
    return [0 if values[0] >= values[1] else 1 for values in logits]


def transitions(labels, clean, attacked):
    return dict(occurrences=len(labels),
                clean_wrong=sum(a != y for y, a in zip(labels, clean)),
                attacked_wrong=sum(b != y for y, b in zip(labels, attacked)),
                new_wrong=sum(a == y and b != y for y, a, b in zip(labels, clean, attacked)),
                recoveries=sum(a != y and b == y for y, a, b in zip(labels, clean, attacked)))


def index_groups(audit, rows, groups, name):
    result = {int(row["group"]): row for row in rows}
    audit.check(len(result) == len(rows) == len(groups) and set(result) == set(groups), name)
    return result


def audit_protocol(audit, stage, preparation_sha, expected_source):
    preparation = audit.json(stage / "execution_protocol_v1.json", preparation_sha)
    experiment_path = stage / "experiment_protocol_v1.json"
    protocol = audit.json(experiment_path)
    delta = dict(protocol)
    preflight_ref = delta.pop("preflight_ref")
    audit.check(delta == preparation, "only_preflight_reference_added")
    audit.check(protocol["status"] == "frozen" and protocol["owner_execution_authorized"] is True,
                "frozen_owner_execution_authorization")
    audit.check(protocol["source_commit"] == expected_source, "execution_source_commit")
    audit.check(tuple(protocol["arms"]) == ARMS and tuple(protocol["seeds"]) == SEEDS,
                "fixed_arms_and_seeds")
    audit.check(protocol["epochs"] == 8 and protocol["batch_size"] == 1 and protocol["phase_gpu_hours"] == 2,
                "fixed_epochs_batch_phase_cap")
    audit.check(len(protocol["train_groups"]) == len(set(protocol["train_groups"])) == 256
                and len(protocol["valid_groups"]) == len(set(protocol["valid_groups"])) == 128
                and not set(protocol["train_groups"]) & set(protocol["valid_groups"]), "fixed_disjoint_group_cohorts")
    audit.check(protocol["final_test_allowed"] is False and protocol["LLM_API_calls"] == 0, "no_test_or_api_authorization")
    for name, wanted in protocol["source_files"].items():
        audit.bytes(name, wanted)
    for row in protocol["inputs"] + protocol.get("retained_inputs", []):
        audit.bytes(row["path"], row["sha256"])
    preflight = audit.json(preflight_ref["path"], preflight_ref["sha256"])
    audit.check(preflight["status"] == "passed" and preflight["optimizer_steps"] == 0
                and preflight["sealed_victim_loaded"] is False and preflight["sealed_victim_forwards"] == 0
                and preflight["test_queries"] == 0, "passed_zero_update_real_train_preflight")
    audit.check(preflight["source_commit"] == expected_source
                and preflight["protocol_sha256"] == preparation_sha, "preflight_preparation_binding")
    audit.check(preflight["generator_forwards"] == 4 and preflight["autograd_calls"] == 4
                and preflight["classifier_forwards"] == dict(gcn_base=4, gcn_new=0, gaga_new=0, gaga_victim=0),
                "real_preflight_call_counts")
    audit.check(len(preflight["cases"]) == 4, "real_preflight_case_count")
    for arm in ARMS:
        cases = [row for row in preflight["cases"] if row["arm"] == arm]
        audit.check({row["group"] for row in cases} == set(protocol["preflight"]["train_groups"]), "preflight_train_groups")
        for key in ("feature_gradient_norm", "edge_gradient_norm", "feature_head_gradient_norm", "edge_head_gradient_norm"):
            audit.check(all(math.isfinite(row[key]) and row[key] >= 0 for row in cases)
                        and any(row[key] > 0 for row in cases), "recorded_preflight_finite_nonzero_gradient")
        for row in cases:
            audit.check(row["actual_injected_nodes"] == row["node_budget"]
                        and row["actual_injected_edges"] == row["edge_budget"], "preflight_recorded_budgets")
    return protocol, dict(source_commit=expected_source, protocol_sha256=digest(experiment_path))


def audit_training(audit, matrix, protocol, provenance):
    train_registry, initial_by_seed, teacher_states, paired_orders = {}, {}, set(), {}
    train_results, epoch_count, group_rows = {}, 0, 0
    for seed in SEEDS:
        initial_by_seed[seed] = set()
        for arm in ARMS:
            directory = matrix / (arm + "_seed" + str(seed))
            registration = audit.json(directory / "registered.json")
            initial = audit.json(directory / "initial_state.json")
            result = audit.json(directory / "result.json")
            audit.check(registration["provenance"] == provenance and registration["arm"] == arm
                        and registration["seed"] == seed and registration["source_role"] == arm,
                        "train_registration_source_arm_seed")
            audit.check(registration["counter_caps"] == dict(generator_forwards=2048, optimizer_steps=2048)
                        and registration["epochs"] == 8 and registration["evaluation_performed"] is False
                        and registration["sealed_victim_deserialized"] is False, "train_registered_information_boundary")
            audit.check(initial["arm"] == arm and initial["seed"] == seed
                        and initial["captured_before_training"] is True and initial["optimizer_steps"] == 0,
                        "initial_state_capture_receipt")
            audit.check(result["status"] == "completed" and result["arm"] == arm and result["seed"] == seed
                        and result["source_mode"] == arm and result["feedback_arm"] == "gcn_only"
                        and result["consistent_training"] is True, "completed_semantic_training_arm")
            audit.check(result["epochs"] == 8 and result["optimizer_steps"] == result["generator_forwards"] == 2048
                        and result["classifier_forwards"] == dict(gcn_base=2048, gcn_new=0, gaga_new=0, gaga_victim=0),
                        "terminal_update_and_forward_counts")
            audit.check(result["victim_loaded"] is False and result["validation_performed"] is False
                        and result["test_queries"] == 0, "train_no_victim_validation_or_test")
            audit.check(result["initial_model_state_sha256"] == initial["initial_state_sha256"], "initial_hash_receipt_agreement")
            initial_by_seed[seed].add(result["initial_model_state_sha256"])
            audit.check(set(result["teacher_initial_hashes"]) == {"gcn_base"}
                        and result["teacher_initial_hashes"] == result["teacher_final_hashes"], "recorded_gcn_teacher_unchanged")
            teacher_states.add(result["teacher_initial_hashes"]["gcn_base"])
            audit.bytes(directory / "final_model.pt", result["final_model_sha256"])
            audit.bytes(directory / "final_optimizer.pt", result["final_optimizer_sha256"])
            audit.check(result["final_model_state_sha256"] != result["initial_model_state_sha256"], "recorded_generator_content_changed")
            for epoch in range(1, 9):
                rows = audit.jsonl(directory / "raw" / ("train_epoch_%03d.jsonl" % epoch))
                indexed = index_groups(audit, rows, protocol["train_groups"], "all_256_train_groups_each_epoch")
                order = [row["group"] for row in rows]
                previous_order = paired_orders.setdefault((seed, epoch), order)
                audit.check(order == previous_order, "paired_train_loader_order")
                epoch_count += 1
                group_rows += len(rows)
                audit.check(sum(row["targets"] for row in rows) == 3141, "fixed_train_occurrences_per_epoch")
                for group, row in indexed.items():
                    identity = (row["targets"], row["node_budget"], row["edge_budget"])
                    audit.check(all(type(value) is int and value > 0 for value in identity), "positive_train_counts_and_budgets")
                    previous = train_registry.setdefault(group, identity)
                    audit.check(previous == identity, "train_group_target_and_budget_parity")
                    audit.check(row["actual_injected_nodes"] == row["node_budget"]
                                and row["actual_injected_edges"] == row["edge_budget"], "recorded_train_actual_budget_parity")
                    audit.check(row["generation_seed"] == case_key(seed, "train", epoch, group), "train_case_rng_key")
                    audit.check(set(row["losses"]) == {"gcn_base"}, "single_original_gcn_feedback_only")
                    values = row["losses"]["gcn_base"]
                    audit.check(math.isfinite(values["loss"]) and values["loss"] >= 0
                                and math.isfinite(values["mean_attack_margin"])
                                and type(values["wrong"]) is int and 0 <= values["wrong"] <= row["targets"],
                                "finite_train_loss_margin_and_count")
                    audit.close(row["total_loss"], values["loss"], "single_teacher_total_loss")
                receipt = audit.json(directory / ("epoch_%03d.json" % epoch))
                audit.check(receipt["epoch"] == epoch and receipt["groups"] == 256
                            and receipt["optimizer_steps"] == receipt["generator_forwards"] == epoch * 256,
                            "epoch_cumulative_update_counts")
                audit.check(receipt["classifier_forwards"] == dict(gcn_base=epoch*256, gcn_new=0, gaga_new=0, gaga_victim=0)
                            and receipt["validation_performed"] is False and receipt["victim_loaded"] is False,
                            "epoch_teacher_and_information_boundary")
                audit.close(receipt["mean_loss"], statistics.fmean(row["total_loss"] for row in rows), "epoch_mean_loss_recount")
            train_results[arm, seed] = result
        audit.check(len(initial_by_seed[seed]) == 1, "paired_initial_generator_content")
    audit.check(len(teacher_states) == 1, "same_frozen_gcn_across_six_runs")
    audit.check(epoch_count == 48 and group_rows == 12288, "complete_training_matrix_raw_rows")
    return train_results, dict(completed_jobs=6, epochs=epoch_count, group_optimizer_rows=group_rows,
                              generator_forwards=12288, original_gcn_forwards=12288,
                              initial_pairing_checked=True, teacher_content_receipts_consistent=True)


def audit_evaluation(audit, stage, protocol, provenance, train_results):
    evaluation = stage / "evaluation_attempt1"
    result = audit.json(evaluation / "result.json")
    registration = audit.json(evaluation / "registered.json")
    audit.check(result["status"] == "completed" and result["source_commit"] == provenance["source_commit"]
                and result["protocol_sha256"] == provenance["protocol_sha256"] and result["test_queries"] == 0,
                "completed_evaluation_source_binding")
    audit.check(registration["provenance"] == provenance and registration["all6_trainings_complete_before_victim_load"] is True
                and registration["original_valid_groups"] == protocol["valid_groups"] and registration["final_test_used"] is False,
                "evaluation_gated_information_boundary_receipt")
    audit.check(result["counts"] == dict(gcn_base=896, gaga_victim=896, generations=768), "evaluation_model_counts")
    clean = index_groups(audit, audit.jsonl(evaluation / "clean_predictions.jsonl"), protocol["valid_groups"], "clean_128_groups")
    ids_seen = set()
    clean_predictions = {}
    for group, row in clean.items():
        labels, ids = row["labels"], row["node_ids"]
        audit.check(len(labels) == len(ids) > 0 and len(set(ids)) == len(ids)
                    and all(type(value) is int and value >= 0 for value in ids)
                    and all(value == 1 for value in labels), "clean_fraud_target_ids_and_labels")
        ids_seen.update(ids)
        for detector in ("gcn", "gaga"):
            clean_predictions[group, detector] = classify(audit, row[detector + "_logits"], len(ids))
    audit.check(sum(len(row["labels"]) for row in clean.values()) == 1091 and len(ids_seen) == 374,
                "clean_occurrence_unique_target_denominators")
    summaries = {(row["arm"], row["seed"]): row for row in result["summaries"]}
    audit.check(len(result["summaries"]) == len(summaries) == 6
                and set(summaries) == {(arm, seed) for arm in ARMS for seed in SEEDS}, "six_evaluation_summaries")
    rows_by_run, metrics, export_count = {}, [], 0
    for seed in SEEDS:
        for arm in ARMS:
            identity = arm + "_seed" + str(seed)
            rows = index_groups(audit, audit.jsonl(evaluation / (identity + "_predictions.jsonl")),
                                protocol["valid_groups"], "all_128_attacked_groups")
            groups_receipt = index_groups(audit, audit.json(evaluation / (identity + "_groups.json")),
                                         protocol["valid_groups"], "all_128_group_receipts")
            summary = summaries[arm, seed]
            audit.check(summary["groups"] == 128 and summary["checkpoint_sha256"] == train_results[arm, seed]["final_model_sha256"]
                        and registration["checkpoint_sha256s"][arm + "_s" + str(seed)] == summary["checkpoint_sha256"],
                        "evaluated_terminal_checkpoint_binding")
            audit.check(sum(row["actual_injected_nodes"] for row in rows.values()) == summary["injected_nodes"] == 7776
                        and sum(row["actual_injected_edges"] for row in rows.values()) == summary["injected_edges"] == 358491,
                        "recorded_evaluation_physical_budget_totals")
            local = {name: collections.Counter() for name in ("gcn", "gaga")}
            macro = {name: [] for name in ("gcn", "gaga")}
            for group, row in rows.items():
                base = clean[group]
                audit.check(row["node_ids"] == base["node_ids"] and row["labels"] == base["labels"], "clean_attack_identity_order")
                audit.check(row["node_budget"] == row["actual_injected_nodes"]
                            and row["edge_budget"] == row["actual_injected_edges"], "recorded_evaluation_budget_caps")
                stripped = {key: value for key, value in row.items() if key not in ("node_ids", "labels", "gcn_logits", "gaga_logits")}
                audit.check(groups_receipt[group] == stripped, "raw_prediction_and_group_receipt_parity")
                for detector in ("gcn", "gaga"):
                    prediction = classify(audit, row[detector + "_logits"], len(base["labels"]))
                    counted = transitions(base["labels"], clean_predictions[group, detector], prediction)
                    audit.check(row[detector + "_transitions"] == counted, "raw_argmax_transition_recount")
                    local[detector].update(counted)
                    macro[detector].append(100 * counted["attacked_wrong"] / counted["occurrences"])
            for detector in ("gcn", "gaga"):
                totals = {key: local[detector][key] for key in TRANSITION_KEYS}
                audit.check(totals == summary["totals"][detector] and totals["occurrences"] == 1091,
                            "raw_recount_matches_evaluation_totals")
                mr = 100 * totals["attacked_wrong"] / 1091
                audit.close(summary[detector + "_mr_percent"], mr, "raw_recount_matches_recorded_mr")
                metrics.append(dict(arm=arm, seed=seed, detector=detector, **totals,
                                    mr_percent=mr, group_macro_mr_percent=statistics.fmean(macro[detector])))
            audit.check(len(summary["exports"]) == 8 and {row["group"] for row in summary["exports"]} == set(protocol["export_valid_groups"]),
                        "eight_preselected_export_groups")
            for export in summary["exports"]:
                audit.bytes(evaluation / export["path"], export["sha256"])
                export_count += 1
            rows_by_run[arm, seed] = rows
    paired, transitions_all = [], collections.Counter()
    strata = {"one_node_one_edge": collections.Counter(), "one_node_multiple_edges": collections.Counter(),
              "multiple_nodes": collections.Counter()}
    group_cases = []
    for seed in SEEDS:
        for detector in ("gaga", "gcn"):
            legacy_metric = next(row for row in metrics if row["arm"] == "legacy" and row["seed"] == seed and row["detector"] == detector)
            fixed_metric = next(row for row in metrics if row["arm"] == "source_fixed" and row["seed"] == seed and row["detector"] == detector)
            paired.append(dict(seed=seed, detector=detector, delta_mr_pp=fixed_metric["mr_percent"] - legacy_metric["mr_percent"],
                               delta_group_macro_pp=fixed_metric["group_macro_mr_percent"] - legacy_metric["group_macro_mr_percent"]))
        for group in protocol["valid_groups"]:
            a, b, base = rows_by_run["legacy", seed][group], rows_by_run["source_fixed", seed][group], clean[group]
            audit.check(all(a[key] == b[key] for key in ("node_ids", "labels", "node_budget", "edge_budget")),
                        "paired_attack_id_order_budget_parity")
            ac = classify(audit, a["gaga_logits"], len(base["labels"]))
            bc = classify(audit, b["gaga_logits"], len(base["labels"]))
            cc = clean_predictions[group, "gaga"]
            counts = collections.Counter(occurrences=len(base["labels"]), group_seed_cases=1)
            for y, ap, bp, cp in zip(base["labels"], ac, bc, cc):
                aw, bw, cw = ap != y, bp != y, cp != y
                counts["legacy_wrong"] += aw
                counts["fixed_wrong"] += bw
                if cw:
                    counts["additional_recovery"] += aw and not bw
                    counts["previous_recovery_lost"] += bw and not aw
                else:
                    counts["legacy_new_success_lost"] += aw and not bw
                    counts["new_success_gained"] += bw and not aw
                    counts["shared_new_wrong"] += aw and bw
            delta = counts["fixed_wrong"] - counts["legacy_wrong"]
            audit.check(delta == counts["new_success_gained"] - counts["legacy_new_success_lost"]
                        - counts["additional_recovery"] + counts["previous_recovery_lost"], "paired_transition_algebra")
            counts["delta_wrong"] = delta
            counts["group_wins" if delta > 0 else "group_losses" if delta < 0 else "group_ties"] = 1
            transitions_all.update(counts)
            key = ("multiple_nodes" if a["node_budget"] > 1 else
                   "one_node_one_edge" if a["edge_budget"] == 1 else "one_node_multiple_edges")
            strata[key].update(counts)
            group_cases.append(dict(seed=seed, group=group, **dict(counts)))
    audit.check(export_count == 48 and len(group_cases) == 384 and transitions_all["occurrences"] == 3273,
                "all_export_and_paired_cohort_counts")
    values = [row["delta_mr_pp"] for row in paired if row["detector"] == "gaga"]
    mean, sd = statistics.fmean(values), statistics.stdev(values)
    half = TCRIT_DF2 * sd / math.sqrt(3)
    aggregates = {}
    for arm in ARMS:
        values_arm = [row["mr_percent"] for row in metrics if row["arm"] == arm and row["detector"] == "gaga"]
        aggregates[arm] = dict(mean_mr_percent=statistics.fmean(values_arm), sample_sd_pp=statistics.stdev(values_arm))
    return dict(groups=128, unique_targets=374, occurrences_per_run=1091, recorded_exports_byte_checked=export_count,
                metrics=metrics, gaga_aggregate=aggregates, paired_seed_differences=paired,
                mean_paired_delta_mr_pp=mean, paired_t95_pp=[mean-half, mean+half], attack_seeds=3, degrees_of_freedom=2,
                consistent_development_gain=(mean >= 1 and all(value > 0 for value in values)),
                paired_transition_counts=dict(transitions_all),
                budget_strata={key: dict(value, delta_mr_pp=100*value["delta_wrong"]/value["occurrences"])
                               for key, value in strata.items()}), group_cases


def audit_queue_and_cost(audit, stage, matrix, protocol, preparation_sha, provenance):
    complete = audit.json(stage / "queue_complete.json")
    matrix_complete = audit.json(matrix / "queue_complete.json")
    progress = audit.json(matrix / "queue_progress.json")
    marker = audit.json(stage / "all6_trainings_complete.json")
    audit.check(complete == matrix_complete and complete["status"] == "completed", "queue_completion_receipts_equal")
    audit.check(complete["scientific_source"] == provenance["source_commit"]
                and complete["preparation_protocol_sha256"] == preparation_sha
                and complete["experiment_protocol_sha256"] == provenance["protocol_sha256"], "queue_source_protocol_binding")
    audit.check(complete["test_queries"] == complete["LLM_API_calls"] == 0, "queue_no_test_or_api")
    audit.bytes(stage / "evaluation_attempt1" / "result.json", complete["evaluation_result_sha256"])
    expected = ["real_train_preflight", "saved_teacher_diagnostic"] + [arm + "_seed" + str(seed) for seed in SEEDS for arm in ARMS] + ["development_evaluation"]
    jobs = complete["jobs"]
    audit.check([row["name"] for row in jobs] == expected and len(jobs) == 9, "exclusive_queue_order_no_duplicate_jobs")
    audit.check(progress["jobs"] == jobs and marker["status"] == "completed" and marker["jobs"] == jobs[:-1],
                "all_training_completion_marker_precedes_evaluation")
    charge = 0.
    previous_end = None
    guardian_rows = []
    for job in jobs:
        name = job["name"]
        audit.check(job["exit_code"] == 0 and job["outer_seconds"] > 0 and job["pid"] > 0, "successful_job_exit_and_outer_time")
        guardian = audit.json(matrix / (name + "_guardian.json"), job["guardian_sha256"])
        audit.check(guardian["status"] == "completed" and guardian.get("exit_code", 0) == 0
                    and guardian["module"] == job["module"], "guardian_completion_module")
        audit.check(guardian["numeric_visibility"] == protocol["cuda_visible_devices"]
                    and guardian["own_physical_uuids"] == [protocol["physical_gpu_uuid"]]
                    and guardian["torch_visible_count"] == guardian["native_visible_count"] == 1,
                    "recorded_guardian_device_identity")
        audit.close(guardian["charged_gpu_wall_hours"], guardian["elapsed_seconds"] / 3600, "guardian_charge_equals_lifetime")
        expected_charge = max(job["outer_seconds"] / 3600, guardian["charged_gpu_wall_hours"])
        audit.close(job["charged_gpu_wall_hours"], expected_charge, "maximum_outer_guardian_charge")
        audit.check(guardian["elapsed_seconds"] <= guardian["limit_seconds"] + 15,
                    "guardian_within_registered_operational_bound")
        started = dt.datetime.fromisoformat(guardian["started_utc"])
        if previous_end is not None:
            if started.timestamp() < previous_end - 1:
                audit.warnings.append("Guardian wall-clock timestamps overlap by more than one second; monotonic queue lifetimes and serial source remain authoritative.")
        previous_end = started.timestamp() + guardian["elapsed_seconds"]
        charge += expected_charge
        guardian_rows.append(dict(name=name, outer_seconds=job["outer_seconds"], guardian_seconds=guardian["elapsed_seconds"],
                                  charged_gpu_wall_hours=expected_charge))
    audit.close(complete["charged_gpu_hours"], charge, "complete_total_charge_reconciled")
    audit.close(progress["charged_gpu_hours"], charge, "progress_total_charge_reconciled")
    audit.close(marker["charged_gpu_hours"], sum(row["charged_gpu_wall_hours"] for row in jobs[:-1]), "training_marker_charge_reconciled")
    audit.check(charge <= protocol["phase_gpu_hours"], "two_gpu_hour_phase_limit")
    synthetic = audit.json(matrix / "synthetic_receipt.json")
    audit.check(synthetic["status"] == "passed" and synthetic["exit_code"] == 0 and synthetic["GPU_hours"] == 0
                and synthetic["source_commit"] == provenance["source_commit"], "cpu_synthetic_receipt_passed")
    audit.bytes(matrix / synthetic["stdout_log"], synthetic["stdout_sha256"])
    companion = audit.json(stage / "saved_teacher_diagnostic_attempt1" / "result.json")
    audit.check(companion["status"] == "completed" and companion["official_victim_loaded"] is False,
                "companion_receipt_completed_without_victim")
    audit.check(companion["provenance"] == provenance, "companion_receipt_source_binding")
    audit.check(companion["counters"] == dict(gaga_teacher_dispatches=104, gaga_teacher_completed_forwards=104,
                    attack_graphs_reconstructed=96, generator_forwards=0, victim_forwards=0,
                    optimizer_steps=0, API_calls=0, TEST_queries=0), "companion_recorded_104_forward_cap")
    for name, wanted in companion["output_hashes"].items():
        audit.bytes(stage / "saved_teacher_diagnostic_attempt1" / name, wanted)
    return dict(phase_charged_gpu_wall_hours=charge, phase_limit_gpu_hours=2, guardians=guardian_rows,
                successful_guardian_jobs=9, companion_receipt_scope="Author self-authored companion: counter/output-byte consistency only, not independent implementation review.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-source-sha256", required=True)
    parser.add_argument("--preparation-protocol-sha256", default=PREPARATION_SHA)
    parser.add_argument("--execution-source", default=SOURCE)
    args = parser.parse_args()
    root = pathlib.Path(args.root).resolve()
    output = pathlib.Path(args.output)
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    if (root / "runs").resolve() not in output.parents:
        raise RuntimeError("Audit output must remain in a fresh ignored runs directory.")
    output.mkdir(parents=True, exist_ok=False)
    started, cpu = time.monotonic(), time.process_time()
    auditor = Audit(root)
    result = dict(status="failed", scope="Standard-library saved-record audit only", model_calls=0,
                  tensor_deserializations=0, dataset_loads=0, graph_loads=0, API_calls=0, GPU_hours=0,
                  authorship="Independent of model/training/evaluation/queue authorship; authored the optional companion diagnostic.")
    try:
        own_sha = digest(__file__)
        auditor.check(own_sha == args.audit_source_sha256, "external_before_outcome_source_freeze")
        write(output / "registered.json", dict(audit_source_sha256=own_sha,
              expected_preparation_protocol_sha256=args.preparation_protocol_sha256,
              expected_execution_source=args.execution_source,
              registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
              outcome_reads_started=False, no_model_or_tensor_loads=True))
        stage, matrix = auditor.path(args.stage, directory=True), auditor.path(args.matrix, directory=True)
        protocol, provenance = audit_protocol(auditor, stage, args.preparation_protocol_sha256, args.execution_source)
        train_results, training = audit_training(auditor, matrix, protocol, provenance)
        evaluation, group_cases = audit_evaluation(auditor, stage, protocol, provenance, train_results)
        cost = audit_queue_and_cost(auditor, stage, matrix, protocol, args.preparation_protocol_sha256, provenance)
        write(output / "private_paired_group_recount.json", group_cases)
        result.update(status="passed", provenance=provenance, training=training, evaluation=evaluation, cost=cost,
                      limitations=["Recorded actual graph budgets/counters are reconciled; tensor graph topology is not deserialized or independently reconstructed.",
                          "Initial/final normalized state hashes and unchanged teacher states are reconciled from receipts; checkpoint byte hashes are independently measured.",
                          "TRAIN group counts/budgets/case keys are checked against the frozen cohort and across jobs; no new dataset IDs/labels are read.",
                          "This audit does not replay models or prove numerical/device determinism, generalization, novelty or human verification.",
                          "Repeated development-cohort target memberships are correlated; the t interval is over three attack seeds, conditional on fixed teachers/cohort.",
                          "The auditor authored the companion module; its 104-forward receipt check is not independent of implementation authorship."])
    except BaseException as error:
        traceback.print_exc()
        result.update(error_type=type(error).__name__, error=repr(error), traceback=traceback.format_exc())
    finally:
        result.update(check_counts=dict(auditor.checks), unique_input_files_hashed=len(auditor.inputs),
                      warnings=auditor.warnings, process_cpu_seconds=time.process_time()-cpu,
                      wall_seconds=time.monotonic()-started)
        write(output / "input_hashes.json", [dict(path=name, sha256=wanted) for name, wanted in sorted(auditor.inputs.items())])
        write(output / "result.json", result)
        print(json.dumps(result, allow_nan=False), flush=True)
    if result["status"] != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
