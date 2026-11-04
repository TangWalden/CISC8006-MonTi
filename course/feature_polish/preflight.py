"""Four TRAIN-only fixed-graph affine/feature-gradient checks.

The official victim is never loaded. Existing seed-three legacy generators and
both frozen feedback classifiers are read only. No optimizer step is performed.
All finite differences use interior normalized injected features and the same
fixed hard graph; this does not validate a derivative of edge selection.
"""
import argparse
import json
import math
import time
import traceback

from course.feature_polish.common import ROOT, private_output, validate_protocol, write_json


GROUPS = (263, 238, 84, 125)
SEQUENCE_ATOL = SEQUENCE_RTOL = 2e-5
LOGIT_ATOL = LOGIT_RTOL = 2e-4
FD_H = 1e-3
FD_ATOL = 1e-3
FD_RTOL = .03
MAX_FORWARDS = {"gaga": 60, "gcn": 44}


def graph_with_features(graph, original_count, injected_features):
    """Retain every hard arc/order; replace only the appended raw features."""
    import torch
    if injected_features.shape != graph.ndata["x"][original_count:].shape:
        raise RuntimeError("Replacement injected feature dimensions differ.")
    copied = graph.local_var()
    copied.ndata["x"] = torch.cat((graph.ndata["x"][:original_count].detach(), injected_features), dim=0)
    return copied


def discrepancy(actual, expected, atol, rtol):
    import torch
    if actual.shape != expected.shape or not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise RuntimeError("Nonfinite or incompatible comparison tensors.")
    error = (actual - expected).detach().abs()
    return dict(max_absolute_error=float(error.max()),
                max_tolerance_fraction=float((error / (atol + rtol * expected.detach().abs())).max()),
                passed=bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = private_output(args.output, create=True)
    started, cpu_started = time.monotonic(), time.process_time()
    counts = dict(generator_forwards=0, gaga_forward_dispatches=0, gaga_completed_forwards=0,
                  gcn_forward_dispatches=0, gcn_completed_forwards=0, canonical_sequence_updates=0,
                  autograd_calls=0, finite_difference_gaga_forwards=0, finite_difference_gcn_forwards=0,
                  optimizer_steps=0, official_victim_forwards=0, API_calls=0, TEST_queries=0)
    result = dict(status="failed", official_victim_loaded=False, optimizer_steps=0,
                  counts=counts, cases=[])
    engine = None
    try:
        protocol, provenance = validate_protocol(args.protocol)
        if tuple(protocol["preflight"]["train_groups"]) != GROUPS:
            raise RuntimeError("The four frozen TRAIN preflight groups differ.")
        deadline = started + protocol["preflight_wall_seconds"]
        import dgl
        import torch
        import torch.nn.functional as F
        from utils.utils import get_n_ids, set_seed
        from course.dual_surrogate.runtime import load_data, load_original_view, checked_logits, gcn_logits, model_hash
        from course.dual_surrogate.sequence_adapter import updated_sequences
        from course.feature_polish.affine import build_affine, features_from_unit_box
        from course.feature_polish.refine import make_baseline_engine
        from course.feature_polish.teacher import load_teacher

        def check_deadline():
            if time.monotonic() >= deadline:
                raise TimeoutError("Fixed TRAIN preflight wall deadline reached.")

        torch.set_num_threads(8)
        set_seed(20261104)
        dgl.seed(20261104)
        dgl.random.seed(20261104)
        engine, data, baseline_receipt = make_baseline_engine(protocol, 3, output, deadline)
        view = load_original_view(data)
        teacher, manifest = load_teacher(protocol, data)
        engine.phase, engine.current_epoch = "preflight", 8
        original_count = data.graph.num_nodes()
        gcn = engine.teachers["gcn_base"]
        classifiers = {"gaga": teacher, "gcn": gcn}
        before = {name: model_hash(model) for name, model in classifiers.items()}
        generator_before = model_hash(engine.model)
        for model in classifiers.values():
            if model.training or any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("A feedback classifier is not frozen in eval mode.")
        minimum, maximum = data.features.min(dim=0).values, data.features.max(dim=0).values
        span = maximum - minimum
        if not bool(torch.isfinite(minimum).all() and torch.isfinite(maximum).all() and (span >= 0).all()):
            raise RuntimeError("Invalid original raw feature box.")
        zero_width = span == 0
        if not bool((span > 0).any()):
            raise RuntimeError("All original feature coordinates are constant.")

        def forward(name, graph=None, targets=None, sequence=None, finite_difference=False):
            check_deadline()
            if counts[name + "_forward_dispatches"] >= MAX_FORWARDS[name]:
                raise RuntimeError("Preflight classifier forward cap reached: " + name)
            counts[name + "_forward_dispatches"] += 1
            if finite_difference:
                counts["finite_difference_" + name + "_forwards"] += 1
            value = (checked_logits(teacher(sequence), len(targets)) if name == "gaga"
                     else gcn_logits(gcn, graph, targets, training=False))
            counts[name + "_completed_forwards"] += 1
            return value

        def loss_from_logits(logits):
            # Binary CE to class zero equals Softplus(z_fraud-z_benign).
            # This remains the actual PGD objective even if float32 Softplus happens
            # to underflow. Raw-margin functional checks below are separately
            # registered derivative checks, never a replacement PGD objective.
            return F.softplus(logits[:, 1] - logits[:, 0]).mean()

        def feature_gradient_check(name, graph, ids, affine, x0):
            u = torch.full_like(x0, .5, requires_grad=True)
            x = features_from_unit_box(u, minimum, maximum)
            value = (forward(name, targets=ids, sequence=affine.sequence(x)) if name == "gaga"
                     else forward(name, graph=graph_with_features(graph, original_count, x), targets=ids))
            loss = loss_from_logits(value)
            raw_margin = (value[:, 1] - value[:, 0]).mean()
            counts["autograd_calls"] += 1
            ce_gradient = torch.autograd.grad(loss, u, retain_graph=True, create_graph=False)[0]
            counts["autograd_calls"] += 1
            gradient = torch.autograd.grad(raw_margin, u, retain_graph=False, create_graph=False)[0]
            if not bool(torch.isfinite(loss) and torch.isfinite(raw_margin)
                        and torch.isfinite(gradient).all() and torch.isfinite(ce_gradient).all()):
                raise RuntimeError("A normalized-feature gradient is nonfinite: " + name)
            if bool((gradient[:, zero_width] != 0).any() or (ce_gradient[:, zero_width] != 0).any()):
                raise RuntimeError("A constant raw-feature coordinate has a nonzero normalized gradient.")
            flat = gradient.detach().reshape(-1).cpu().tolist()
            ce_flat = ce_gradient.detach().reshape(-1).cpu().tolist()
            ce_nonzero = any(value != 0 for value in ce_flat)
            indices = sorted((index for index, value in enumerate(flat) if value != 0), key=lambda index: (-abs(flat[index]), index))[:4]
            if len(indices) != 4:
                raise RuntimeError("Fewer than four nonzero raw-margin feature-gradient coordinates: " + name)
            measurements = []
            for index in indices:
                with torch.no_grad():
                    plus, minus = u.detach().clone(), u.detach().clone()
                    plus.reshape(-1)[index] += FD_H
                    minus.reshape(-1)[index] -= FD_H
                    plus_x = features_from_unit_box(plus, minimum, maximum)
                    minus_x = features_from_unit_box(minus, minimum, maximum)
                    if name == "gaga":
                        plus_logits = forward(name, targets=ids, sequence=affine.sequence(plus_x), finite_difference=True)
                        minus_logits = forward(name, targets=ids, sequence=affine.sequence(minus_x), finite_difference=True)
                    else:
                        plus_logits = forward(name, graph=graph_with_features(graph, original_count, plus_x), targets=ids, finite_difference=True)
                        minus_logits = forward(name, graph=graph_with_features(graph, original_count, minus_x), targets=ids, finite_difference=True)
                    plus_margin = float((plus_logits[:, 1]-plus_logits[:, 0]).mean())
                    minus_margin = float((minus_logits[:, 1]-minus_logits[:, 0]).mean())
                    plus_loss, minus_loss = float(loss_from_logits(plus_logits)), float(loss_from_logits(minus_logits))
                checks = {}
                objectives = [("raw_margin", flat[index], plus_margin, minus_margin)]
                if ce_nonzero:
                    objectives.append(("actual_softplus", ce_flat[index], plus_loss, minus_loss))
                for objective, automatic, positive, negative in objectives:
                    numerical = (positive-negative) / (2*FD_H)
                    error = abs(numerical-automatic)
                    tolerance = FD_ATOL+FD_RTOL*max(abs(numerical), abs(automatic))
                    passed = math.isfinite(numerical) and error <= tolerance
                    checks[objective] = dict(autograd=automatic, central_difference=numerical,
                                             absolute_error=error, tolerance=tolerance, passed=passed)
                    if not passed:
                        raise RuntimeError("Fixed feature finite-difference check failed for " + name + ": " + json.dumps(dict(flat_coordinate=index, objective=objective, **checks[objective])))
                measurements.append(dict(flat_coordinate=index, checks=checks))
            return dict(actual_objective="float32 mean Softplus(fraud-benign), exactly as the PGD implementation",
                        functional_objective="mean fraud-minus-benign raw margin; four highest absolute nonzero gradients",
                        normalized_input="u=.5; central perturbations remain inside [.499,.501]",
                        loss=float(loss.detach()), raw_margin=float(raw_margin.detach()),
                        raw_margin_gradient_l2=float(gradient.detach().double().norm()),
                        raw_margin_gradient_max_abs=max(abs(value) for value in flat),
                        actual_softplus_gradient_l2=float(ce_gradient.detach().double().norm()),
                        actual_softplus_gradient_all_zero=not ce_nonzero,
                        ce_finite_difference_checked_on_same_four_coordinates=ce_nonzero,
                        zero_width_gradients_exact_zero=True,
                        finite_difference_h=FD_H, coordinates=measurements)

        for group in GROUPS:
            check_deadline()
            engine.current_group = group
            ids = torch.tensor(get_n_ids(data.target_assignment, group), device=data.device, dtype=torch.long)
            nb, eb = int(data.node_budgets[group]), int(data.edge_budgets[group])
            with torch.no_grad():
                graph = engine.get_perturbed_graph(ids, nb, eb, False)
            counts["generator_forwards"] = engine.generator_forwards
            x0 = engine.capture["x_atk"].detach()
            candidates = engine.capture["candidate_ids"].detach()
            injected = torch.arange(original_count, graph.num_nodes(), device=data.device, dtype=graph.idtype)
            if (x0.shape != (nb, data.feat_dim) or graph.num_nodes() != original_count + nb
                    or graph.num_edges() != data.graph.num_edges() + nb + 2 * eb):
                raise RuntimeError("Baseline TRAIN graph violates its fixed physical budget.")
            if bool((x0 < minimum - 1e-6).any() or (x0 > maximum + 1e-6).any()):
                raise RuntimeError("Baseline injected features lie outside the original box.")
            with torch.no_grad():
                counts["canonical_sequence_updates"] += 1
                base_sequence = updated_sequences(view, graph, ids, candidates, injected, training=False)
                affine = build_affine(view, graph, ids, candidates, injected, x0, base_sequence=base_sequence)
                probes = [("baseline_x0", x0),
                          ("interior_u25", features_from_unit_box(torch.full_like(x0, .25), minimum, maximum)),
                          ("interior_u75", features_from_unit_box(torch.full_like(x0, .75), minimum, maximum))]
                comparisons = []
                for name, features in probes:
                    replaced = graph_with_features(graph, original_count, features)
                    if name == "baseline_x0":
                        canonical = base_sequence
                    else:
                        counts["canonical_sequence_updates"] += 1
                        canonical = updated_sequences(view, replaced, ids, candidates, injected, training=False)
                    linear = affine.sequence(features)
                    sequence_error = discrepancy(linear, canonical, SEQUENCE_ATOL, SEQUENCE_RTOL)
                    if not sequence_error["passed"]:
                        raise RuntimeError("Affine versus canonical hard-sequence mismatch: " + name)
                    canonical_logits = forward("gaga", targets=ids, sequence=canonical)
                    affine_logits = forward("gaga", targets=ids, sequence=linear)
                    logit_error = discrepancy(affine_logits, canonical_logits, LOGIT_ATOL, LOGIT_RTOL)
                    if not logit_error["passed"]:
                        raise RuntimeError("Affine versus canonical hard-logit mismatch: " + name)
                    comparisons.append(dict(probe=name, sequence=sequence_error, logits=logit_error))
                original_gcn = forward("gcn", graph=graph, targets=ids)
                copied_gcn = forward("gcn", graph=graph_with_features(graph, original_count, x0), targets=ids)
                gcn_parity = discrepancy(copied_gcn, original_gcn, LOGIT_ATOL, LOGIT_RTOL)
                if not gcn_parity["passed"]:
                    raise RuntimeError("Replacing x0 changed the frozen hard-graph GCN logits.")
            row = dict(group=group, targets=len(ids), node_budget=nb, edge_budget=eb,
                       actual_injected_nodes=graph.num_nodes()-original_count,
                       actual_nonself_undirected_edges=(graph.num_edges()-data.graph.num_edges()-nb)//2,
                       feature_dimension=data.feat_dim, frozen_zero_width_dimensions=int(zero_width.sum()),
                       canonical_affine_comparisons=comparisons, hard_gcn_x0_replacement=gcn_parity)
            # Append partial case information before derivative checks, retaining
            # useful progress if a strict finite-difference contract fails.
            result["cases"].append(row)
            row["gaga_feature_gradient"] = feature_gradient_check("gaga", graph, ids, affine, x0)
            row["gcn_feature_gradient"] = feature_gradient_check("gcn", graph, ids, affine, x0)
            engine.capture = engine.live_capture = None
            del graph, affine, base_sequence, x0, candidates, injected
        if (counts["generator_forwards"] != 4 or counts["gaga_completed_forwards"] != 60
                or counts["gcn_completed_forwards"] != 44 or counts["autograd_calls"] != 16
                or counts["canonical_sequence_updates"] != 12
                or counts["finite_difference_gaga_forwards"] != 32
                or counts["finite_difference_gcn_forwards"] != 32):
            raise RuntimeError("Preflight fixed case/forward/derivative counters differ.")
        for name, model in classifiers.items():
            if model_hash(model) != before[name] or model.training or any(parameter.grad is not None or parameter.requires_grad for parameter in model.parameters()):
                raise RuntimeError("A feedback classifier changed during preflight: " + name)
        if engine.optimizer_steps or model_hash(engine.model) != generator_before:
            raise RuntimeError("TRAIN preflight changed the frozen baseline generator.")
        result.update(status="passed", provenance=provenance, baseline_generator_seed=3,
                      baseline_generator_state_sha256=generator_before,
                      classifier_state_sha256=before,
                      feature_box="Original transductive per-coordinate min/max; zero-width dimensions frozen",
                      tolerances=dict(sequence_atol=SEQUENCE_ATOL, sequence_rtol=SEQUENCE_RTOL,
                                      logit_atol=LOGIT_ATOL, logit_rtol=LOGIT_RTOL,
                                      finite_difference_h=FD_H, finite_difference_atol=FD_ATOL,
                                      finite_difference_rtol=FD_RTOL),
                      gradient_scope="Exact injected-feature path on fixed hard graph; no edge-selection/topology derivative",
                      teacher_manifest_schema=manifest.get("schema"))
    except BaseException as error:
        traceback.print_exc()
        result.update(error_type=type(error).__name__, error=repr(error), traceback=traceback.format_exc())
    finally:
        if engine is not None:
            counts["generator_forwards"] = engine.generator_forwards
            counts["optimizer_steps"] = engine.optimizer_steps
        result.update(counts=counts, elapsed_seconds=time.monotonic()-started,
                      process_cpu_seconds=time.process_time()-cpu_started)
        write_json(output/"result.json", result, exclusive=True)
        print("FEATURE_POLISH_PREFLIGHT_END", json.dumps(result, allow_nan=False), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
