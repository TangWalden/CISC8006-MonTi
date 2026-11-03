"""POSTHOC saved-JSON-only diagnosis, frozen after final VALID outcomes were known.

No ML imports, tensor/checkpoint deserialization, network/model/API calls or
training. TRAIN logs are read once. TRAIN and VALID are different cohorts and
their rates are never subtracted as a paired gap. The illustrative failure rule
is fixed here: seed3, lowest group ID with fewer joint than GCN-only GAGA errors.
That outcome-selected example is descriptive, not a population estimate/cause.
"""
import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics
import time

ARMS = ("gcn_only", "gaga_only", "two_gcn", "gcn_gaga")
ROLES = {"gcn_only": ("gcn_base",), "gaga_only": ("gaga_new",),
         "two_gcn": ("gcn_base", "gcn_new"), "gcn_gaga": ("gcn_base", "gaga_new")}
SEEDS = (3, 4, 5)


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def load(path):
    check(pathlib.Path(path).suffix == ".json", "Only JSON input is permitted.")
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with pathlib.Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n")


def table(path, rows):
    if not rows:
        return
    with pathlib.Path(path).open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def jsonl_once(path, inventory, role, deadline):
    check(path.suffix == ".jsonl", "Only saved JSONL inputs are permitted.")
    h, rows = hashlib.sha256(), []
    with path.open("rb") as f:
        for raw in f:
            check(time.monotonic() < deadline, "The300-second CPU deadline expired.")
            h.update(raw)
            rows.append(json.loads(raw.decode("utf-8")))
    inventory.append(dict(role=role, sha256=h.hexdigest(), rows=len(rows)))
    return rows


def cls(values):
    check(len(values) == 2 and all(math.isfinite(v) for v in values), "Invalid saved binary logits.")
    return 0 if values[0] >= values[1] else 1


def transition(labels, clean, attacked):
    values = dict(occurrences=len(labels), clean_wrong=0, attacked_wrong=0,
                  new_wrong=0, recoveries=0, retained_correct=0, retained_wrong=0)
    for y, c, a in zip(labels, clean, attacked):
        cw, aw = c != y, a != y
        values["clean_wrong"] += cw; values["attacked_wrong"] += aw
        values["retained_wrong" if cw and aw else "recoveries" if cw else
               "new_wrong" if aw else "retained_correct"] += 1
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix-root", required=True); p.add_argument("--protocol", required=True)
    p.add_argument("--evaluation-root", required=True); p.add_argument("--output", required=True)
    args = p.parse_args()
    start = time.monotonic(); deadline = start+300
    matrix, protocol_path, evaluation, out = map(pathlib.Path,
        (args.matrix_root, args.protocol, args.evaluation_root, args.output))
    out.mkdir(parents=True, exist_ok=False)
    source_hash = sha(__file__)
    save(out/"registered_before_train_log_reads.json", dict(
        status="POSTHOC_after_final_outcomes_known", script_sha256=source_hash,
        known_outcomes="Joint GAGA VALID MR38.9245%, GCN-only51.3596%; negative in all3 seeds.",
        train_scope="All12 primary completed trainings; all8 saved epochs; one pass; fixed256 TRAIN groups.",
        cpu_wall_limit_seconds=300, ml_imports=False, tensor_deserialization=False,
        model_queries=0, API_calls=0, optimizer_steps=0, GPU_hours=0,
        example_rule="Seed3; smallest groupID with joint GAGA attacked-wrong below GCN-only, after matching all128 group/node coordinates."))
    protocol = load(protocol_path)
    groups = set(protocol["train_groups"])
    check(len(groups) == 256 and len(protocol["train_groups"]) == 256, "TRAIN cohort is not fixed256.")
    inventory = [dict(role="execution_protocol", sha256=sha(protocol_path))]
    all_epochs, first_last, sizes = [], [], {}
    for arm in ARMS:
        for seed in SEEDS:
            job = matrix/("%s_seed%d" % (arm, seed))
            result = load(job/"result.json")
            check(result["status"] == "completed" and result["epochs"] == 8
                  and result["optimizer_steps"] == 2048, "A primary training is incomplete.")
            inventory.append(dict(role="%s_seed%d/result.json" % (arm, seed), sha256=sha(job/"result.json")))
            records = {}
            for epoch in range(1, 9):
                filename = "train_epoch_%03d.jsonl" % epoch
                rows = jsonl_once(job/"raw"/filename, inventory,
                                  "%s_seed%d/raw/%s" % (arm, seed, filename), deadline)
                check(len(rows) == 256 and len({r["group"] for r in rows}) == 256
                      and {r["group"] for r in rows} == groups, "Saved TRAIN cohort is incomplete/different.")
                for row in rows:
                    check(type(row["targets"]) is int and row["targets"] > 0,
                          "Invalid saved TRAIN target denominator.")
                    g = row["group"]
                    if g not in sizes:
                        sizes[g] = row["targets"]
                    check(sizes[g] == row["targets"] and set(row["losses"]) == set(ROLES[arm]),
                          "TRAIN group size or teacher role changed.")
                occurrences = sum(r["targets"] for r in rows)
                for role in ROLES[arm]:
                    losses = [r["losses"][role] for r in rows]
                    check(all(type(v["wrong"]) is int and 0 <= v["wrong"] <= r["targets"]
                              and math.isfinite(v["loss"]) and math.isfinite(v["mean_attack_margin"])
                              for r, v in zip(rows, losses)), "Invalid saved TRAIN teacher metrics.")
                    wrong = sum(v["wrong"] for v in losses)
                    record = dict(arm=arm, seed=seed, epoch=epoch, teacher=role, groups=256,
                        target_occurrences=occurrences, teacher_predicted_benign=wrong,
                        occurrence_weighted_wrong_percent=100*wrong/occurrences,
                        occurrence_weighted_softplus= sum(r["targets"]*v["loss"] for r,v in zip(rows,losses))/occurrences,
                        occurrence_weighted_attack_margin=sum(r["targets"]*v["mean_attack_margin"] for r,v in zip(rows,losses))/occurrences,
                        group_equal_softplus=statistics.mean(v["loss"] for v in losses),
                        group_equal_attack_margin=statistics.mean(v["mean_attack_margin"] for v in losses))
                    all_epochs.append(record); records[(epoch, role)] = record
            for role in ROLES[arm]:
                first, last = records[(1, role)], records[(8, role)]
                first_last.append(dict(arm=arm, seed=seed, teacher=role, groups=256,
                    target_occurrences=first["target_occurrences"],
                    epoch1_wrong_percent=first["occurrence_weighted_wrong_percent"],
                    epoch8_wrong_percent=last["occurrence_weighted_wrong_percent"],
                    wrong_change_pp=last["occurrence_weighted_wrong_percent"]-first["occurrence_weighted_wrong_percent"],
                    epoch1_softplus=first["occurrence_weighted_softplus"], epoch8_softplus=last["occurrence_weighted_softplus"],
                    epoch1_attack_margin=first["occurrence_weighted_attack_margin"],
                    epoch8_attack_margin=last["occurrence_weighted_attack_margin"]))
    check(len(all_epochs) == 144 and len(first_last) == 18, "Teacher/epoch coverage changed.")
    eval_path = evaluation/"result.json"; ev = load(eval_path)
    check(ev["status"] == "completed" and ev["test_queries"] == 0 and len(ev["summaries"]) == 12,
          "Saved development evaluation is incomplete.")
    inventory.append(dict(role="evaluation/result.json", sha256=sha(eval_path)))
    validation = [dict(arm=r["arm"], seed=r["seed"], groups=r["groups"],
        occurrences=r["totals"]["gaga"]["occurrences"],
        gcn_mr_percent=r["gcn_mr_percent"], sealed_gaga_mr_percent=r["gaga_mr_percent"],
        sealed_gaga_new_wrong=r["totals"]["gaga"]["new_wrong"],
        sealed_gaga_recoveries=r["totals"]["gaga"]["recoveries"]) for r in ev["summaries"]]
    clean_rows = jsonl_once(evaluation/"clean_predictions.jsonl", inventory, "evaluation/clean_predictions.jsonl", deadline)
    clean = {r["group"]:r for r in clean_rows}
    pair = {}
    for arm in ("gcn_only", "gcn_gaga"):
        pair[arm] = {r["group"]:r for r in jsonl_once(evaluation/(arm+"_seed3_predictions.jsonl"),
                    inventory, "evaluation/"+arm+"_seed3_predictions.jsonl", deadline)}
    valid = set(protocol["valid_groups"])
    check(len(clean_rows) == len(clean) == 128 and set(clean) == valid and
          all(len(rows)==128 and set(rows)==valid for rows in pair.values()), "VALID pair coverage differs.")
    examples = []
    for group in sorted(valid):
        baseline, joint, before = pair["gcn_only"][group], pair["gcn_gaga"][group], clean[group]
        keys = list(before["node_ids"])
        check(len(keys) == len(set(keys)), "Duplicate within-group saved node coordinate.")
        aligned = {}
        for name, row in (("clean",before),("baseline",baseline),("joint",joint)):
            check(len(row["node_ids"])==len(keys) and set(row["node_ids"])==set(keys),
                  "Saved VALID node coordinates do not match.")
            aligned[name] = {node:(y,cls(logits)) for node,y,logits in
                             zip(row["node_ids"],row["labels"],row["gaga_logits"])}
        check(all(aligned["clean"][node][0] == aligned["baseline"][node][0] ==
                  aligned["joint"][node][0] == 1 for node in keys), "Saved VALID truths differ.")
        transitions = {}
        for name, row in (("baseline",baseline),("joint",joint)):
            transitions[name] = transition([aligned["clean"][node][0] for node in keys],
                [aligned["clean"][node][1] for node in keys], [aligned[name][node][1] for node in keys])
            check(all(transitions[name][k] == v for k,v in row["gaga_transitions"].items()),
                  "Saved VALID transitions disagree with matched vectors.")
        if transitions["joint"]["attacked_wrong"] < transitions["baseline"]["attacked_wrong"]:
            check(all(baseline[k]==joint[k] for k in ("node_budget","edge_budget","actual_injected_nodes","actual_injected_edges")),
                  "Illustrative pair physical budgets differ.")
            examples.append(dict(seed=3, group=group, matched_node_coordinates=True,
                target_occurrences=len(keys), node_budget=baseline["node_budget"],edge_budget=baseline["edge_budget"],
                baseline=transitions["baseline"], joint=transitions["joint"]))
    arm_train = []
    for arm in ARMS:
        for role in ROLES[arm]:
            rows = [r for r in first_last if r["arm"]==arm and r["teacher"]==role]
            arm_train.append(dict(arm=arm,teacher=role,seeds=3,
                **{name:statistics.mean(r[name] for r in rows) for name in (
                    "epoch1_wrong_percent","epoch8_wrong_percent","wrong_change_pp",
                    "epoch1_softplus","epoch8_softplus","epoch1_attack_margin","epoch8_attack_margin")},
                seeds_with_wrong_rate_increase=sum(r["wrong_change_pp"]>0 for r in rows),
                seeds_with_softplus_decrease=sum(r["epoch8_softplus"]<r["epoch1_softplus"] for r in rows)))
    summary = dict(status="completed_POSTHOC_saved_JSON_diagnosis", script_sha256=source_hash,
        train_groups_per_epoch=256, train_target_occurrences_per_epoch=sum(sizes.values()),
        train_teacher_epoch_rows=144, primary_runs=12, epochs_per_run=8,
        train_first_last_by_arm_teacher=arm_train, validation_runs=validation,
        illustrative_failure=(examples[0] if examples else None),
        matching_validation_groups_checked=128,
        limits=["Final VALID outcomes were known before this diagnostic was frozen; exploratory post-hoc analysis.",
                "TRAIN metrics are online pre-update observations along changing weights and generation noise, not frozen-checkpoint re-evaluations.",
                "Occurrence weighting pools saved group means; no per-target TRAIN logits, margins or clean predictions were opened.",
                "Saved wrong counts mean predicted benign for targets treated as fraud by the executed source; this analysis does not reload dataset labels.",
                "TRAIN256 and VALID128 have different targets/contexts. No rate subtraction is a paired generalization gap.",
                "Success against an independent training teacher does not prove useful transfer to the sealed GAGA victim or explain a causal mechanism.",
                "The smallest seed3 negative group is an outcome-selected illustration, not a representative population or independent experiment.",
                "No model, tensor file, test prediction, API or new training was used."],
        CPU_seconds=time.monotonic()-start,GPU_hours=0,model_queries=0,API_calls=0,optimizer_steps=0)
    table(out/"train_teacher_epochs.csv",all_epochs); table(out/"train_first_last.csv",first_last)
    table(out/"validation_runs.csv",validation); save(out/"input_hashes.json",inventory)
    save(out/"summary.json",summary)
    print(json.dumps(summary,allow_nan=False))


if __name__ == "__main__":
    main()
