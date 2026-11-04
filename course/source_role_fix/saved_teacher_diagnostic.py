"""Measure an existing fresh GAGA teacher on 96 retained attack exports.

This companion diagnostic never constructs a MonTi generator, loads the official
victim, updates parameters, changes group masks, or selects a new training arm.
The victim channel is joined exclusively from immutable saved VALID JSONL
records. Outputs containing IDs/logits remain in a fresh ignored runs directory.
"""
import argparse
import json
import math
import os
import pathlib
import time
import traceback

from course.source_role_fix.common import ROOT, private_output, sha256, validate_protocol, write_json


OLD_ARMS = ("gcn_only", "gaga_only", "two_gcn", "gcn_gaga")
SEEDS = (3, 4, 5)


def _repository_file(name):
    relative = pathlib.Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Only repository-relative saved input paths are allowed.")
    path = (ROOT / relative).resolve()
    if ROOT.resolve() not in path.parents or not path.is_file():
        raise RuntimeError("A saved input is missing or resolves outside the repository: " + str(name))
    return path


def _checked_file(name, wanted):
    path = _repository_file(name)
    if sha256(path) != wanted:
        raise RuntimeError("Saved input bytes differ: " + str(name))
    return path


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _index_predictions(path, expected_groups):
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            group = int(row["group"])
            if group in rows:
                raise RuntimeError("Duplicate group in retained prediction file.")
            rows[group] = row
    if set(rows) != set(expected_groups):
        raise RuntimeError("Retained prediction cohort differs from its frozen protocol.")
    return rows


def collect_saved_inputs(protocol):
    """Hash every export before any tensor, graph, or classifier is loaded."""
    spec = protocol["saved_teacher_diagnostic"]
    if spec["max_forwards"] != 104 or not (0 < spec["wall_seconds"] <= 600):
        raise RuntimeError("Companion inference/time limits differ from the fixed 8+96 design.")
    old_path = _checked_file(spec["old_protocol_path"], spec["old_protocol_sha256"])
    old = _read_json(old_path)
    if old.get("status") != "frozen" or tuple(old["arms"]) != OLD_ARMS or tuple(old["seeds"]) != SEEDS:
        raise RuntimeError("Retained experiment identity differs.")
    groups = [int(group) for group in old["export_valid_groups"]]
    if len(groups) != 8 or len(set(groups)) != 8 or not set(groups) <= set(old["valid_groups"]):
        raise RuntimeError("Exactly the eight originally exported VALID groups are required.")
    evaluation = _repository_file(spec["evaluation_path"] + "/result.json").parent
    result_path = _checked_file(
        (evaluation / "result.json").relative_to(ROOT).as_posix(),
        spec["evaluation_result_sha256"],
    )
    result = _read_json(result_path)
    if result.get("status") != "completed" or result.get("test_queries") != 0:
        raise RuntimeError("Retained evaluation must have completed without TEST queries.")
    clean_path = _checked_file(
        (evaluation / "clean_predictions.jsonl").relative_to(ROOT).as_posix(),
        spec["clean_predictions_sha256"],
    )
    clean = _index_predictions(clean_path, old["valid_groups"])
    summaries = {(row["arm"], int(row["seed"])): row for row in result["summaries"]}
    expected_runs = {(arm, seed) for arm in OLD_ARMS for seed in SEEDS}
    if len(result["summaries"]) != 12 or set(summaries) != expected_runs:
        raise RuntimeError("Retained evaluation does not contain exactly twelve old runs.")
    exports, seen_paths, predictions = [], set(), {}
    input_receipts = [dict(path=str(path.relative_to(ROOT)), sha256=sha256(path))
                      for path in (old_path, result_path, clean_path)]
    if set(spec["prediction_files"]) != {arm + "_seed" + str(seed) for arm, seed in expected_runs}:
        raise RuntimeError("Freeze all twelve retained prediction JSONL files.")
    for arm in OLD_ARMS:
        for seed in SEEDS:
            run_name = arm + "_seed" + str(seed)
            reference = spec["prediction_files"][run_name]
            prediction_path = _checked_file(reference["path"], reference["sha256"])
            if prediction_path.parent != evaluation:
                raise RuntimeError("Prediction file belongs to another evaluation directory.")
            predictions[arm, seed] = _index_predictions(prediction_path, old["valid_groups"])
            input_receipts.append(dict(path=str(prediction_path.relative_to(ROOT)), sha256=sha256(prediction_path)))
            records = summaries[arm, seed]["exports"]
            if len(records) != 8 or {int(row["group"]) for row in records} != set(groups):
                raise RuntimeError("An old run lacks one of its eight exports.")
            by_group = {int(row["group"]): row for row in records}
            for group in groups:
                reference = by_group[group]
                relative = pathlib.Path(reference["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError("Unsafe retained export path.")
                path = (evaluation / relative).resolve()
                if path.parent != evaluation or path in seen_paths or sha256(path) != reference["sha256"]:
                    raise RuntimeError("A retained export is duplicated, moved, or changed.")
                seen_paths.add(path)
                exports.append(dict(arm=arm, seed=seed, group=group, path=path,
                                    sha256=reference["sha256"]))
                input_receipts.append(dict(path=str(path.relative_to(ROOT)), sha256=reference["sha256"]))
    if len(exports) != 96:
        raise RuntimeError("The companion must retain all 96 exported attacks.")
    if "export_files" in spec:
        explicit = {row["path"]: row["sha256"] for row in spec["export_files"]}
        measured = {row["path"].relative_to(ROOT).as_posix(): row["sha256"] for row in exports}
        if len(spec["export_files"]) != 96 or explicit != measured:
            raise RuntimeError("Explicit companion export freeze differs from the old result manifest.")
    return spec, old, groups, clean, predictions, exports, input_receipts


def reconstruct(data, payload, saved, clean):
    """Use the frozen ExperimentHandler physical endpoint/self-loop mapping."""
    import dgl
    import torch
    from utils.utils import get_n_ids

    base, device = data.graph, data.graph.device
    original = base.num_nodes()
    if set(payload) != {"x_atk", "adj_attack", "target_ids", "candidate_ids"}:
        raise RuntimeError("Retained capture schema differs from the original exporter.")
    feature = payload["x_atk"].detach().to(device)
    adjacency = payload["adj_attack"].detach().to(device)
    integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if any(payload[name].ndim != 1 or payload[name].dtype not in integer_types for name in ("target_ids", "candidate_ids")):
        raise RuntimeError("Retained node IDs must be one-dimensional integer tensors.")
    target_ids = payload["target_ids"].detach().to(device=device, dtype=base.idtype)
    candidate_ids = payload["candidate_ids"].detach().to(device=device, dtype=base.idtype)
    wanted_ids = torch.tensor(get_n_ids(data.target_assignment, int(saved["group"])), device=device, dtype=base.idtype)
    if (not torch.equal(target_ids, wanted_ids) or target_ids.cpu().tolist() != saved["node_ids"]
            or saved["node_ids"] != clean["node_ids"] or saved["labels"] != clean["labels"]):
        raise RuntimeError("Capture, assignment, clean, and attacked target order/labels differ.")
    labels = data.labels[target_ids].detach().cpu().tolist()
    if labels != saved["labels"] or not labels or any(value != 1 for value in labels):
        raise RuntimeError("The retained fraud-target cohort changed.")
    node_budget, edge_budget = int(saved["node_budget"]), int(saved["edge_budget"])
    group = int(saved["group"])
    if node_budget != int(data.node_budgets[group]) or edge_budget != int(data.edge_budgets[group]):
        raise RuntimeError("Saved budgets differ from the original implementation budgets.")
    if feature.shape != (node_budget, data.feat_dim) or feature.dtype != base.ndata["x"].dtype:
        raise RuntimeError("Retained feature shape/dtype differs.")
    originals = torch.cat((target_ids, candidate_ids))
    if (len(originals) != len(torch.unique(originals))
            or not bool(((originals >= 0) & (originals < original)).all())):
        raise RuntimeError("Retained target/candidate IDs overlap or are invalid.")
    if adjacency.shape != (node_budget, len(originals) + node_budget):
        raise RuntimeError("Retained attack adjacency has another shape.")
    if not bool(torch.isfinite(feature).all() and torch.isfinite(adjacency).all()):
        raise RuntimeError("Nonfinite retained attack tensors.")
    if not bool(((adjacency == 0) | (adjacency == 1)).all()):
        raise RuntimeError("Retained adjacency is not binary.")
    legal = torch.tril(torch.ones_like(adjacency, dtype=torch.bool), diagonal=len(originals) - 1)
    if bool((adjacency[~legal] != 0).any()) or int(adjacency.count_nonzero()) != edge_budget:
        raise RuntimeError("Retained attack violates its original legal mask/budget.")
    if not bool((adjacency[:, :len(target_ids)].sum(dim=1) >= 1).all()):
        raise RuntimeError("An injected node lacks its original mandatory target attachment.")
    graph = dgl.add_nodes(base, node_budget, {"x": feature})
    injected = torch.arange(original, original + node_budget, device=device, dtype=base.idtype)
    indices = torch.cat((originals, injected))
    src, dst = adjacency.nonzero(as_tuple=True)
    physical_src, physical_dst = src.to(base.idtype) + original, indices[dst]
    if bool((physical_src == physical_dst).any()):
        raise RuntimeError("Retained attack contains a non-legal injected self edge.")
    logical = torch.stack((torch.minimum(physical_src, physical_dst),
                           torch.maximum(physical_src, physical_dst)), dim=1)
    if len(torch.unique(logical, dim=0)) != edge_budget:
        raise RuntimeError("Retained logical undirected edges are duplicated.")
    graph.add_edges(physical_src, physical_dst)
    graph.add_edges(physical_dst, physical_src)
    graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
    actual_edges = (graph.num_edges() - base.num_edges() - node_budget) // 2
    if (graph.num_nodes() != original + node_budget
            or graph.num_edges() != base.num_edges() + node_budget + 2 * edge_budget
            or actual_edges != edge_budget
            or int(saved["actual_injected_nodes"]) != node_budget
            or int(saved["actual_injected_edges"]) != edge_budget):
        raise RuntimeError("Reconstructed physical counts differ from retained evaluation.")
    if not torch.equal(graph.ndata["x"][:original], base.ndata["x"]) or not torch.equal(graph.ndata["x"][original:], feature):
        raise RuntimeError("Reconstruction changed original or injected features.")
    # Every injected incident arc is checked against the capture, including arcs
    # to earlier injected nodes allowed by the original lower-triangular mask.
    all_src, all_dst = graph.edges(order="eid")
    keep = (all_src != all_dst) & ((all_src >= original) | (all_dst >= original))
    actual_codes = torch.sort(all_src[keep].long() * graph.num_nodes() + all_dst[keep].long()).values
    expected_codes = torch.sort(torch.cat((physical_src.long() * graph.num_nodes() + physical_dst.long(),
                                          physical_dst.long() * graph.num_nodes() + physical_src.long()))).values
    if not torch.equal(actual_codes, expected_codes):
        raise RuntimeError("Reconstructed injected incident arcs differ from the capture.")
    return graph, target_ids, candidate_ids, injected


def _prediction_list(logits):
    if not logits or any(len(values) != 2 or not all(math.isfinite(float(value)) for value in values) for values in logits):
        raise RuntimeError("Retained/new logits must be finite two-class rows.")
    return [0 if values[0] >= values[1] else 1 for values in logits]


def _transitions(labels, clean, attacked):
    if not (len(labels) == len(clean) == len(attacked)):
        raise RuntimeError("A transition denominator differs.")
    return dict(occurrences=len(labels), clean_wrong=sum(a != y for y, a in zip(labels, clean)),
                attacked_wrong=sum(b != y for y, b in zip(labels, attacked)),
                new_wrong=sum(a == y and b != y for y, a, b in zip(labels, clean, attacked)),
                recoveries=sum(a != y and b == y for y, a, b in zip(labels, clean, attacked)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = private_output(args.output, create=True)
    started, cpu_started = time.monotonic(), time.process_time()
    counters = dict(gaga_teacher_dispatches=0, gaga_teacher_completed_forwards=0,
                    attack_graphs_reconstructed=0, generator_forwards=0,
                    victim_forwards=0, optimizer_steps=0, API_calls=0, TEST_queries=0)
    result = dict(status="failed", official_victim_loaded=False, counters=counters)
    try:
        protocol, provenance = validate_protocol(args.protocol)
        spec, old, groups, clean, predictions, exports, input_receipts = collect_saved_inputs(protocol)
        deadline = started + spec["wall_seconds"]
        write_json(out / "registered.json", dict(provenance=provenance, own_source_sha256=sha256(__file__),
                   retained_input_hashes=input_receipts, saved_graphs=96, original_exported_groups=8,
                   max_fresh_teacher_forwards=104, official_victim_access="saved VALID logits only",
                   training_adaptation_allowed=False, crossed_mask_inference=False), exclusive=True)
        # Only after all 96 export byte hashes pass do any ML/tensor loads begin.
        import numpy as np
        import torch
        import dgl
        from utils.utils import set_seed
        from course.dual_surrogate.runtime import load_data, load_fit_view, load_classifier, checked_logits, model_hash
        from course.dual_surrogate.sequence_adapter import updated_sequences

        if os.environ.get("CUDA_VISIBLE_DEVICES") != protocol["cuda_visible_devices"] or torch.cuda.device_count() != 1:
            raise RuntimeError("Companion device visibility differs from the frozen phase.")
        torch.set_num_threads(8)
        set_seed(20261104)
        dgl.seed(20261104)
        dgl.random.seed(20261104)
        data = load_data(old, 20261104)
        teacher_directory = _repository_file(old["teacher_directory"] + "/teacher_artifacts.json").parent
        fit_view, teacher_manifest = load_fit_view(data, teacher_directory, old)
        teacher = load_classifier("gaga_new", data, teacher_directory)
        teacher_before = model_hash(teacher)
        sequence_reference = teacher_manifest["cache"]["files"]["sequences"]
        sequence_path = _checked_file((teacher_directory / sequence_reference["path"]).relative_to(ROOT).as_posix(), sequence_reference["sha256"])
        fit_sequences = np.load(sequence_path, allow_pickle=False, mmap_mode="r")
        if fit_sequences.shape != (data.graph.num_nodes(), 7, data.feat_dim):
            raise RuntimeError("Retained FIT sequence cache shape differs.")

        def forward(sequence, count):
            if time.monotonic() >= deadline:
                raise TimeoutError("Companion frozen wall deadline reached.")
            if counters["gaga_teacher_dispatches"] >= spec["max_forwards"]:
                raise RuntimeError("Companion fresh-teacher forward cap exhausted.")
            counters["gaga_teacher_dispatches"] += 1
            value = checked_logits(teacher(sequence), count)
            counters["gaga_teacher_completed_forwards"] += 1
            return value.detach().cpu().tolist()

        clean_teacher = {}
        summaries = {}
        with torch.no_grad(), (out / "clean_predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for group in groups:
                saved = clean[group]
                ids = saved["node_ids"]
                from utils.utils import get_n_ids
                if ids != [int(value) for value in get_n_ids(data.target_assignment, group)]:
                    raise RuntimeError("Retained clean target order differs from its assignment.")
                selected = np.array(fit_sequences[ids], dtype=np.float32, copy=True)
                logits = forward(torch.from_numpy(selected).to(data.device), len(ids))
                clean_teacher[group] = _prediction_list(logits)
                stream.write(json.dumps(dict(group=group, node_ids=ids, labels=saved["labels"],
                              fresh_teacher_logits=logits, saved_victim_logits=saved["gaga_logits"]), allow_nan=False) + "\n")
                stream.flush()
        with torch.no_grad(), (out / "attacked_predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for export in exports:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Companion frozen wall deadline reached before reconstruction.")
                arm, seed, group = export["arm"], export["seed"], export["group"]
                saved = predictions[arm, seed][group]
                if sha256(export["path"]) != export["sha256"]:
                    raise RuntimeError("Retained export changed after initial hash checks.")
                payload = torch.load(export["path"], map_location="cpu")
                graph, targets, candidates, injected = reconstruct(data, payload, saved, clean[group])
                counters["attack_graphs_reconstructed"] += 1
                sequence = updated_sequences(fit_view, graph, targets, candidates, injected, training=False)
                logits = forward(sequence, len(targets))
                teacher_pred = _prediction_list(logits)
                victim_pred = _prediction_list(saved["gaga_logits"])
                victim_clean = _prediction_list(clean[group]["gaga_logits"])
                labels = saved["labels"]
                a = _transitions(labels, clean_teacher[group], teacher_pred)
                b = _transitions(labels, victim_clean, victim_pred)
                for key, value in b.items():
                    if int(saved["gaga_transitions"][key]) != value:
                        raise RuntimeError("Saved official-victim transitions do not recount.")
                agreements = sum(x == y for x, y in zip(teacher_pred, victim_pred))
                both_wrong = sum(x != y and z != y for y, x, z in zip(labels, teacher_pred, victim_pred))
                row = dict(arm=arm, seed=seed, group=group, node_ids=saved["node_ids"], labels=labels,
                           node_budget=int(saved["node_budget"]), edge_budget=int(saved["edge_budget"]),
                           export_sha256=export["sha256"], fresh_teacher_logits=logits,
                           saved_victim_logits=saved["gaga_logits"], teacher_transitions=a,
                           saved_victim_transitions=b, prediction_agreements=agreements,
                           teacher_and_victim_wrong=both_wrong)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                key = (arm, seed)
                summary = summaries.setdefault(key, dict(arm=arm, seed=seed, graphs=0,
                    teacher={name: 0 for name in a}, saved_victim={name: 0 for name in b},
                    prediction_agreements=0, teacher_and_victim_wrong=0))
                summary["graphs"] += 1
                for name, value in a.items(): summary["teacher"][name] += value
                for name, value in b.items(): summary["saved_victim"][name] += value
                summary["prediction_agreements"] += agreements
                summary["teacher_and_victim_wrong"] += both_wrong
                del graph, sequence, payload, targets, candidates, injected
        if (counters["gaga_teacher_dispatches"] != 104 or counters["gaga_teacher_completed_forwards"] != 104
                or counters["attack_graphs_reconstructed"] != 96 or len(summaries) != 12
                or any(row["graphs"] != 8 for row in summaries.values())):
            raise RuntimeError("The fixed companion inference/cohort count differs.")
        if model_hash(teacher) != teacher_before or teacher.training or any(parameter.grad is not None for parameter in teacher.parameters()):
            raise RuntimeError("Companion inference changed the frozen teacher.")
        for summary in summaries.values():
            for name in ("teacher", "saved_victim"):
                channel = summary[name]
                channel["clean_mr_percent"] = 100 * channel["clean_wrong"] / channel["occurrences"]
                channel["attacked_mr_percent"] = 100 * channel["attacked_wrong"] / channel["occurrences"]
        result.update(status="completed", provenance=provenance, original_exported_groups=8,
                      saved_attack_graphs=96, fresh_teacher_mask_scope="Original FIT80 only; unchanged training convention",
                      official_victim_channel="Immutable saved VALID logits; zero victim loads/forwards",
                      teacher_state_sha256=teacher_before, summaries=list(summaries.values()),
                      output_hashes={name: sha256(out / name) for name in ("clean_predictions.jsonl", "attacked_predictions.jsonl")},
                      limitations=["Eight predetermined exported groups, not the full 128-group development cohort.",
                          "Repeated targets, seeds and graph contexts are correlated; no IID confidence claim.",
                          "Teacher/victim weights and group-mask conventions both differ; this does not isolate their causal effects.",
                          "No crossed-mask inference, training adaptation, generator search, API or TEST evaluation."])
    except BaseException as error:
        traceback.print_exc()
        result.update(error_type=type(error).__name__, error=repr(error), traceback=traceback.format_exc())
    finally:
        result.update(counters=counters, wall_seconds=time.monotonic() - started,
                      process_cpu_seconds=time.process_time() - cpu_started)
        write_json(out / "result.json", result, exclusive=True)
        print("SAVED_TEACHER_DIAGNOSTIC_END", json.dumps(result, allow_nan=False), flush=True)
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
