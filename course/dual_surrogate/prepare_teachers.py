"""Fresh TRAIN-only teachers and private FIT80-only GAGA caches.

No ML imports or execution at import time. The original victim and its labelled
sequence caches are never loaded. Non-TRAIN features/topology are transductive
inputs; only original TRAIN labels are indexed, and only FIT labels enter GAGA
group masks. Internal STOP labels select a checkpoint and never enter its input.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time
import traceback


ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE_SCHEMA = "monti_fit_only_gaga_cache_v1"
CACHE_FILES = dict(sequences="fit_sequences.npy", feat_sum="fit_feat_sum.npy", d="fit_degrees.npy",
                   group_masks="fit_group_masks.pt", adj="original_adjacency.npz", manifest="fit_cache_manifest.json")
SETTINGS = dict(seed=20261103, split_seed=20261103, fit_fraction=.8, max_epochs=80,
                patience=10, batch_size=512, lr=.001, weight_decay=.0001,
                wall_seconds=900, max_forwards=5000)
QUALITY_GATE = dict(auc_min=.75, macro_f1_min=.60, accuracy_min=.60,
                    benign_recall_min=.20, fraud_recall_min=.20)


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""): digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path, value):
    with pathlib.Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def stratified_train_split(train_ids, train_labels, seed=20261103):
    ids, labels = list(train_ids), list(train_labels)
    if len(ids) != len(labels) or len(set(ids)) != len(ids) or any(type(n) is not int or n < 0 for n in ids):
        raise ValueError("TRAIN identities must be unique nonnegative integers with parallel labels")
    if any(type(y) is not int or y not in (0, 1) for y in labels): raise ValueError("Binary TRAIN labels required")
    fit, stop = [], []
    for label in (0, 1):
        pool = [node for node, truth in zip(ids, labels) if truth == label]
        if len(pool) < 2: raise ValueError("Each class needs FIT and internal STOP members")
        pool.sort(key=lambda node: (hashlib.sha256(("%d|teacher-fit80|%d|%d" % (seed, label, node)).encode()).hexdigest(), node))
        count = int(.8*len(pool))
        fit.extend(pool[:count]); stop.extend(pool[count:])
    if set(fit) & set(stop) or set(fit) | set(stop) != set(ids): raise RuntimeError("Internal split coverage failed")
    return tuple(sorted(fit)), tuple(sorted(stop))


def fit_group_masks(num_nodes, fit_ids, fit_labels):
    ids, labels = list(fit_ids), list(fit_labels)
    if type(num_nodes) is not int or num_nodes < 1 or len(ids) != len(labels) or len(set(ids)) != len(ids):
        raise ValueError("Invalid FIT mask dimensions")
    if any(type(n) is not int or not 0 <= n < num_nodes for n in ids) or any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError("Invalid FIT identities or labels")
    masks = dict(train_neg=[0]*num_nodes, train_pos=[0]*num_nodes, train_masked=[1]*num_nodes)
    for node, label in zip(ids, labels):
        masks["train_masked"][node] = 0
        masks["train_pos" if label else "train_neg"][node] = 1
    return masks


def binary_metrics(labels, scores, predictions=None):
    labels, scores = list(labels), list(scores)
    if not labels or len(labels) != len(scores) or set(labels) != {0, 1} or any(type(y) is not int for y in labels):
        raise ValueError("Both classes and parallel internal STOP scores required")
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
        raise ValueError("Finite probability scores required")
    predictions = [int(p > .5) for p in scores] if predictions is None else list(predictions)
    if len(predictions) != len(labels) or any(type(y) is not int or y not in (0, 1) for y in predictions):
        raise ValueError("Binary predictions required")
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions)); fn = len(labels)-tp-tn-fp
    ordered = sorted(zip(scores, labels)); rank_sum = 0.; index = 0
    while index < len(ordered):
        end = index+1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]: end += 1
        rank_sum += ((index+1+end)/2)*sum(y for _, y in ordered[index:end]); index = end
    positives, negatives = tp+fn, tn+fp
    auc = (rank_sum-positives*(positives+1)/2)/(positives*negatives)
    f1_pos = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    f1_neg = 2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
    return dict(cases=len(labels), accuracy=(tp+tn)/len(labels), macro_f1=(f1_pos+f1_neg)/2,
                auc=auc, benign_recall=tn/negatives, fraud_recall=tp/positives,
                tp=tp, tn=tn, fp=fp, fn=fn, class_counts=[negatives, positives])


def improves_checkpoint(current_macro_f1, best_macro_f1):
    if not math.isfinite(current_macro_f1): raise ValueError("Nonfinite checkpoint score")
    return best_macro_f1 is None or current_macro_f1 > best_macro_f1


def passes_quality(metrics):
    return all(metrics[key[:-4]] >= value for key, value in QUALITY_GATE.items()) and min(metrics["class_counts"]) > 0


class Meter:
    def __init__(self):
        self.started = time.monotonic(); self.cpu_started = time.process_time(); self.forwards = 0; self.optimizer_steps = 0
        self.by_model_phase = {}; self.cpu_prep_seconds = 0.; self.prep_wall_seconds = 0.
    def check(self):
        if time.monotonic()-self.started >= 900: raise TimeoutError("Frozen teacher900second deadline reached")
    def forward(self, model, phase):
        self.check()
        if self.forwards >= 5000: raise RuntimeError("Frozen teacher5000forward ceiling reached")
        self.forwards += 1
        key = model+"_"+phase; self.by_model_phase[key] = self.by_model_phase.get(key, 0)+1
    def report(self):
        return dict(forwards=self.forwards, optimizer_steps=self.optimizer_steps, forward_counts=self.by_model_phase,
                    elapsed_seconds=time.monotonic()-self.started, cpu_prep_seconds=self.cpu_prep_seconds,
                    prep_wall_seconds=self.prep_wall_seconds, cpu_process_seconds=time.process_time()-self.cpu_started,
                    cpu_prep_scope="FIT-mask/cache construction, serialization and cache hashing; whole-process CPU/wall also reported",
                    charged_gpu_wall_hours=(time.monotonic()-self.started)/3600)


def build_sequence_arrays(features, adjacency, adj2, masks, check=lambda: None, chunk_size=512):
    """Sparse CPU aggregation matching upstream masked-self seven-token inputs."""
    import numpy as np
    n, width = features.shape
    sums = np.zeros((n, 7, width), dtype=np.float32); degrees = np.zeros((n, 7, 1), dtype=np.float32)
    sums[:, 0] = features; degrees[:, 0] = 1
    groups = [np.flatnonzero(np.asarray(masks[name])) for name in ("train_neg", "train_pos", "train_masked")]
    for begin in range(0, n, chunk_size):
        check(); rows = np.arange(begin, min(begin+chunk_size, n))
        for hop, source in enumerate((adjacency, adj2)):
            block = source[rows].astype(np.float32)
            for group, columns in enumerate(groups):
                check(); token = 1+hop*3+group
                part = block[:, columns]
                sums[rows, token] = part @ features[columns]
                degrees[rows, token, 0] = np.asarray(part.sum(axis=1)).reshape(-1)
            # Upstream removes the diagonal from A and bool(A@A), then always
            # places the center's own raw feature in the unknown token per hop.
            sums[rows, 3+hop*3] += features[rows]
            degrees[rows, 3+hop*3, 0] += 1
    sequences = sums/np.maximum(degrees, 1)
    if not np.isfinite(sequences).all(): raise RuntimeError("Nonfinite FIT-only sequence cache")
    return sequences, sums, degrees


def _prepare_cache(graph, train_ids, train_labels, output, protocol, meter):
    import numpy as np
    import scipy.sparse as sp
    import torch
    from course.dual_surrogate.verify import verify_teacher_partition, verify_gaga_label_masks
    started, cpu = time.monotonic(), time.process_time()
    fit, stop = stratified_train_split(train_ids, train_labels)
    by_id = dict(zip(train_ids, train_labels))
    masks = fit_group_masks(graph.num_nodes(), fit, [by_id[node] for node in fit])
    partition = verify_teacher_partition(train_ids, list(fit), list(stop))
    mask_receipt = verify_gaga_label_masks(list(fit), list(stop),
        [node for node,value in enumerate(masks["train_neg"]) if value],
        [node for node,value in enumerate(masks["train_pos"]) if value])
    adjacency = sp.csr_matrix(graph.adj_external(scipy_fmt="coo")).T.tocsr()
    adjacency.setdiag(0); adjacency.eliminate_zeros()
    if adjacency.data.size and (adjacency.data.min() != 1 or adjacency.data.max() != 1):
        raise RuntimeError("Expected original binary topology without parallel-edge multiplicity")
    adj2_path = ROOT/protocol["data"]["adj2_cache"]
    adj2 = sp.load_npz(adj2_path).tocsr()
    if adj2.shape != adjacency.shape or adj2.dtype.kind not in "biu" or (adj2.data.size and (adj2.data.min() < 0 or adj2.data.max() > 1)):
        raise RuntimeError("Frozen two-hop cache must be binary topology only")
    if np.any(adj2.diagonal()): raise RuntimeError("Frozen two-hop diagonal must be zero")
    adj2.eliminate_zeros()
    features = graph.ndata["x"].detach().cpu().numpy().astype(np.float32, copy=False)
    sequences, sums, degrees = build_sequence_arrays(features, adjacency, adj2, masks, meter.check)
    for key, value in (("sequences", sequences), ("feat_sum", sums), ("d", degrees)):
        with (output/CACHE_FILES[key]).open("xb") as stream: np.save(stream, value, allow_pickle=False)
    with (output/CACHE_FILES["adj"]).open("xb") as stream: sp.save_npz(stream, adjacency)
    tensor_masks = {key:torch.tensor(value, dtype=torch.float32) for key,value in masks.items()}
    with (output/CACHE_FILES["group_masks"]).open("xb") as stream: torch.save(tensor_masks, stream)
    files = {key:dict(path=name, sha256=sha256(output/name)) for key,name in CACHE_FILES.items() if key != "manifest"}
    manifest = dict(schema=CACHE_SCHEMA, num_nodes=graph.num_nodes(), feat_dim=features.shape[1],
        sequence_shape=list(sequences.shape), files=files,
        fit_group_masks_sha256=files["group_masks"]["sha256"],
        partition=partition, mask_membership_receipt=mask_receipt,
        adj2_reference=dict(path=protocol["data"]["adj2_cache"], sha256=sha256(adj2_path), semantics="support(A@A),zero diagonal; topology-only immutable input"),
        label_input="FIT80 TRAIN labels only; STOP20/VALID/test are unknown in every group input",
        self_mask="Center excluded from label groups; raw center included in unknown token of each hop",
        fit_count=len(fit), stop_count=len(stop), cached_full_train_sequences_used=False)
    write_json(output/CACHE_FILES["manifest"], manifest)
    write_json(output/"train_split_private.json", dict(seed=20261103, fit_nodes=list(fit), stop_nodes=list(stop),
        original_train_nodes=train_ids, fit_label_counts=[sum(by_id[node]==label for node in fit) for label in (0,1)],
        stop_label_counts=[sum(by_id[node]==label for node in stop) for label in (0,1)],
        rule="Per-class floor(.8*n) by SHA256(seed|teacher-fit80|class|node), earliest digest then node; no outputs used"))
    meter.cpu_prep_seconds = time.process_time()-cpu; meter.prep_wall_seconds = time.monotonic()-started
    del sums, degrees, adj2
    return fit, stop, by_id, torch.from_numpy(sequences), manifest


def _train_one(name, graph_cpu, sequences, fit, stop, by_id, output, meter):
    import torch
    import torch.nn.functional as F
    from models.victim_model import GAGA, GCN
    from utils.utils import set_seed
    from course.dual_surrogate.runtime import model_hash
    from course.dual_surrogate.verify import node_ids_sha256
    seed = 20261103 if name == "gaga" else 20261104
    before_steps = meter.optimizer_steps
    set_seed(seed)
    width = int(graph_cpu.ndata["x"].shape[1]); device = torch.device("cuda:0")
    constructor = (dict(feat_dim=width, emb_dim=64, num_classes=2, num_relations=1, n_hops=2,
                        n_heads=4, ff_dim=128, n_layers=3, dropout=.1) if name == "gaga" else
                   dict(feature_dim=width, emb_dim=[64,64], num_classes=2))
    model = (GAGA(**constructor) if name == "gaga" else GCN(**constructor)).to(device)
    initial_state_sha256 = model_hash(model)  # Before optimizer construction or any forward/update.
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    counts = [sum(by_id[node] == label for node in fit) for label in (0,1)]
    weights = torch.tensor([len(fit)/(2*count) for count in counts], dtype=torch.float32, device=device)
    label_tensor = torch.full((graph_cpu.num_nodes(),), -1, dtype=torch.long)
    for node, truth in by_id.items(): label_tensor[node] = truth
    fit_ids = torch.tensor(fit, dtype=torch.long); stop_ids = torch.tensor(stop, dtype=torch.long)
    graph = graph_cpu.to(device) if name == "gcn2" else None
    def forward(ids, phase):
        meter.forward(name, phase)
        value = model(sequences[ids].to(device)) if name == "gaga" else model([graph,graph])
        if name == "gcn2": value = value[ids.to(device)]
        if value.ndim == 1 and len(ids) == 1: value = value.unsqueeze(0)
        if value.shape != (len(ids), 2) or not bool(torch.isfinite(value).all()): raise RuntimeError("Invalid teacher logits")
        return value
    best_score = best_state = best_metrics = best_state_sha256 = None; best_epoch = None; stale = 0; history = []
    for epoch in range(1, 81):
        meter.check(); model.train(); batches = [fit_ids] if name == "gcn2" else fit_ids[torch.randperm(len(fit_ids))].split(512)
        for ids in batches:
            meter.check(); optimizer.zero_grad()
            logits = forward(ids, "fit")
            loss = F.cross_entropy(logits, label_tensor[ids].to(device), weight=weights)
            if not bool(torch.isfinite(loss)): raise RuntimeError("Nonfinite teacher fit loss")
            loss.backward(); optimizer.step(); meter.optimizer_steps += 1
        model.eval(); scores, predictions, raw_logits = [], [], []
        with torch.no_grad():
            evaluation_batches = [stop_ids] if name == "gcn2" else stop_ids.split(512)
            for ids in evaluation_batches:
                logits = forward(ids, "internal_stop")
                scores.extend(torch.softmax(logits, 1)[:,1].cpu().tolist())
                predictions.extend(logits.argmax(1).cpu().tolist()); raw_logits.extend(logits.cpu().tolist())
        truths = [by_id[node] for node in stop]
        metrics = binary_metrics(truths, scores, predictions)
        improved = improves_checkpoint(metrics["macro_f1"], best_score)
        if improved:
            best_score, best_epoch, best_metrics, stale = metrics["macro_f1"], epoch, metrics, 0
            best_state = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            best_state_sha256 = model_hash(model)
        else: stale += 1
        row = dict(epoch=epoch, internal_stop=metrics, selected_if_improved=improved, best_epoch=best_epoch, stale_epochs=stale)
        history.append(row); write_json(output/("%s_epoch_%03d.json" % (name, epoch)), row)
        write_json(output/"raw"/("%s_internal_stop_epoch_%03d.json" % (name, epoch)), dict(node_ids=list(stop), labels=truths, scores=scores, predictions=predictions, logits=raw_logits))
        print("TEACHER_EPOCH", name, json.dumps(row), flush=True)
        if stale >= 10: break
    checkpoint = output/("gaga_teacher.pt" if name == "gaga" else "gcn2_teacher.pt")
    if best_state_sha256 == initial_state_sha256:
        raise RuntimeError("Fresh teacher checkpoint content did not change from initialization")
    with checkpoint.open("xb") as stream: torch.save(best_state, stream)
    training_receipt = dict(seed=seed, optimizer_steps=meter.optimizer_steps-before_steps,
        initial_state_sha256=initial_state_sha256, final_state_sha256=best_state_sha256,
        initial_captured_before_optimizer=True,
        final_state_semantics="Selected earliest-max-macroF1 checkpoint; last optimization epoch may differ",
        normalized_hash_rule="runtime.model_hash: sorted state names, dtype, shape and contiguous raw tensor bytes",
        fit_ids_sha256=node_ids_sha256(list(fit)), tune_ids_sha256=node_ids_sha256(list(stop)),
        original_valid_used=False, test_used=False, sealed_victim_loaded=False,
        checkpoint_selection="internal_tune_only",
        aggregation_label_scope="internal_fit_only" if name == "gaga" else "no_group_label_inputs",
        aggregation_label_ids_sha256=node_ids_sha256(list(fit)) if name == "gaga" else None)
    result = dict(model="GAGA" if name == "gaga" else "GCN", seed=seed, constructor=constructor,
        checkpoint=checkpoint.name, checkpoint_sha256=sha256(checkpoint), epochs=len(history), selected_epoch=best_epoch,
        checkpoint_rule="Earliest strict maximum internal STOP macro-F1; no other split or metric selects the checkpoint",
        internal_stop=best_metrics, quality_gate=QUALITY_GATE, quality_passed=passes_quality(best_metrics),
        fit_class_weights=weights.detach().cpu().tolist(), optimizer="Adam lr.001 weight_decay.0001",
        fit_batch="full graph one optimizer step/epoch; loss FIT nodes only" if name == "gcn2" else "shuffled FIT nodes, batch512",
        inference_semantics="full homogeneous graph normalization; audit sampled-block equivalence before changing inference path" if name == "gcn2" else "FIT80-only seven-token group inputs; masks remain fixed during attacker training",
        forward_counts={phase:meter.by_model_phase.get(name+"_"+phase,0) for phase in ("fit","internal_stop")},
        optimizer_steps=meter.optimizer_steps-before_steps,
        initial_state_sha256=initial_state_sha256, final_state_sha256=best_state_sha256,
        training_receipt=training_receipt,
        prediction_scope="Only FIT loss and internal STOP predictions are indexed/scored; GCN uses full-graph feature propagation",
        labels_outside_original_train_used=False, original_victim_loaded=False, test_evaluation=False)
    write_json(output/(name+"_teacher.json"), result)
    del optimizer, model, graph; torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args(); meter = Meter()
    output = pathlib.Path(args.output).resolve()
    if (ROOT/"runs").resolve() not in output.parents or args.gpu != 0: raise RuntimeError("Use a fresh ignored runs output and registered GPU0")
    output.mkdir(parents=True, exist_ok=False); (output/"raw").mkdir()
    result = dict(status="failed", original_victim_loaded=False, original_victim_forwards=0, test_used=False)
    try:
        from course.dual_surrogate.common import validate_protocol
        validated = validate_protocol(args.protocol)
        protocol, provenance = validated if isinstance(validated, tuple) else (validated, {})
        settings = protocol["teachers"]
        if any(settings.get(key) != value for key,value in SETTINGS.items()) or settings.get("quality_gate") != QUALITY_GATE:
            raise RuntimeError("Frozen teacher settings/quality gate differ from implemented constants")
        sealed_entry = protocol.get("sealed_victim", {})
        sealed_sha = protocol.get("sealed_victim_sha256", sealed_entry.get("sha256") if type(sealed_entry) is dict else None)
        if type(sealed_sha) is not str or not re.fullmatch(r"[0-9a-f]{64}", sealed_sha):
            raise RuntimeError("An externally frozen sealed-victim SHA is required; victim bytes are not loaded here")
        write_json(output/"registered.json", dict(protocol_sha256=sha256(args.protocol), source_commit=protocol["source_commit"],
            provenance=provenance, source_sha256=sha256(__file__), command=[sys.executable]+sys.argv,
            registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(), settings=SETTINGS, quality_gate=QUALITY_GATE,
            inputs=protocol["inputs"], data=protocol["data"], gcn2_seed=20261104,
            cache_schema=CACHE_SCHEMA, sealed_victim_access="none; only its externally supplied digest may be compared"))
        import numpy as np
        import dgl
        import torch
        import scipy
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or torch.cuda.device_count() != 1: raise RuntimeError("Exactly physical GPU0 must be exposed")
        torch.set_num_threads(8); torch.empty(1, device="cuda:0")
        rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"], text=True)
        own = {line.split(",")[1].strip() for line in rows.splitlines() if line.split(",")[0].strip() == str(os.getpid())}
        if own != {protocol["physical_gpu_uuid"]}: raise RuntimeError("Own-PID physical GPU identity differs")
        torch.cuda.reset_peak_memory_stats(0)
        runtime = dict(torch=torch.__version__, dgl=dgl.__version__, numpy=np.__version__, scipy=scipy.__version__, own_gpu_uuids=sorted(own))
        locked = json.loads((ROOT/"environment/runtime.json").read_text(encoding="utf-8"))
        if any(runtime[key] != locked[key] for key in ("torch","dgl","numpy","scipy")):
            raise RuntimeError("The frozen ML runtime changed")
        write_json(output/"runtime.json", runtime)
        graphs, _ = dgl.load_graphs(str(ROOT/protocol["data"]["graph"]))
        graph = graphs[0].to("cpu")
        indices = np.load(ROOT/protocol["data"]["train_indices"], allow_pickle=False)
        train_ids = [int(node) for node in indices.tolist()]
        train_tensor = torch.tensor(train_ids, dtype=torch.long)
        # The serialized graph contains labels. Index ONLY original TRAIN, then
        # discard the global label field before constructing either model input.
        all_labels = graph.ndata.pop("y")
        train_labels = [int(value) for value in all_labels[train_tensor].tolist()]
        del all_labels, graphs
        graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
        graph.ndata["x"] = graph.ndata["x"].to(torch.float32)
        fit, stop, by_id, sequences, cache = _prepare_cache(graph, train_ids, train_labels, output, protocol, meter)
        artifacts = {name:_train_one(name, graph, sequences, fit, stop, by_id, output, meter) for name in ("gaga", "gcn2")}
        if any(item["checkpoint_sha256"] == sealed_sha for item in artifacts.values()): raise RuntimeError("Fresh teacher unexpectedly equals sealed victim digest")
        meter.check()
        aggregate = dict(schema="monti_independent_teacher_artifacts_v1", teachers=artifacts, cache=cache,
            partition=cache["partition"],
            cache_manifest=dict(path=CACHE_FILES["manifest"], sha256=sha256(output/CACHE_FILES["manifest"])),
            fit_masks_sha256=cache["fit_group_masks_sha256"], train_split_sha256=sha256(output/"train_split_private.json"),
            original_victim_loaded=False, labels_outside_original_train_used=False,
            both_quality_gates_passed=all(item["quality_passed"] for item in artifacts.values()), counters=meter.report())
        write_json(output/"teacher_artifacts.json", aggregate)
        result.update(status="completed" if aggregate["both_quality_gates_passed"] else "quality_gate_failed", teacher_artifacts_sha256=sha256(output/"teacher_artifacts.json"),
                      teachers={name:dict(selected_epoch=row["selected_epoch"],internal_stop=row["internal_stop"],quality_passed=row["quality_passed"]) for name,row in artifacts.items()}, peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=repr(error)); traceback.print_exc()
    finally:
        result.update(meter.report()); write_json(output/"result.json", result)
        print("TEACHER_PREPARATION_FINAL", json.dumps(result, allow_nan=False), flush=True)
    if result["status"] != "completed": raise SystemExit(1)


if __name__ == "__main__":
    main()
