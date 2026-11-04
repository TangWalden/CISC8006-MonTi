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
GCN_FD_GRID = (1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5)
# Per TRAIN case: two native hard-graph parity calls, three fixed-block float32
# center calls, one float64 center call and 4*5*2 float64 differences =46.
MAX_FORWARDS = {"gaga": 60, "gcn": 184}


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
                  gcn_native_sample_block_calls=0, gcn_fixed_block_sample_calls=0,
                  gcn_fixed_float32_center_forwards=0, gcn_float64_center_forwards=0,
                  gcn_float64_finite_difference_forwards=0, gcn_float64_clones=0,
                  gcn_fixed_float32_autograd_calls=0, gcn_float64_autograd_calls=0,
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
        from course.feature_polish.refine import make_baseline_engine, tensor_sha256
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
            if name == "gaga":
                value = checked_logits(teacher(sequence), len(targets))
            else:
                counts["gcn_native_sample_block_calls"] += 1
                value = gcn_logits(gcn, graph, targets, training=False)
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

        def fixed_block_gcn_gradient_check(graph, ids, x0, report):
            """Check one fixed sampled operator without changing float32 PGD.

            The float64 model is a deepcopy of the existing frozen weights, not
            a new parameter initialization. Each forward gathers a differentiable
            complete feature stack using the same first-block source NID order.
            Every declared h is measured; only the finest TWO determine passing.
            """
            import copy
            report.update(status="incomplete", methodology="fixed sampled blocks; float64 functional finite differences",
                historical_failure="Attempt2 group238 float32/resampled h=.001 failure remains in its original receipt; not rerun or cleared.",
                actual_refinement_changed=False, finite_difference_grid=list(GCN_FD_GRID),
                acceptance="All-coordinate float32/float64 gradients meet original tolerances; finest two h pass both objectives and preserve both LeakyReLU sign patterns versus center.",
                finite_difference_atol=FD_ATOL, finite_difference_rtol=FD_RTOL,
                coordinates=[], float32_center_repeats=[])
            counts["gcn_fixed_block_sample_calls"] += 1
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(gcn.num_layers)
            _, ordered_targets, sampled = sampler.sample_blocks(graph, ids.to(graph.device))
            if not torch.equal(ordered_targets, ids.to(ordered_targets.device)):
                raise RuntimeError("Fixed-block GCN changed target ordering.")
            blocks = [block.to(graph.device) for block in sampled]
            if len(blocks) != 2:
                raise RuntimeError("The frozen GCN must retain its two sampled layers.")
            if not torch.equal(blocks[0].dstdata[dgl.NID], blocks[1].srcdata[dgl.NID]):
                raise RuntimeError("Frozen GCN block boundary NID order differs.")

            def block_identity(block):
                if dgl.EID not in block.edata:
                    raise RuntimeError("Fixed sampled GCN blocks lack their original EID order.")
                source, destination = block.edges(order="eid")
                return dict(source_nid_sha256=tensor_sha256(block.srcdata[dgl.NID]),
                    destination_nid_sha256=tensor_sha256(block.dstdata[dgl.NID]),
                    original_eid_sha256=tensor_sha256(block.edata[dgl.EID]),
                    local_arc_order_sha256=tensor_sha256(torch.stack((source, destination))),
                    source_nodes=block.num_src_nodes(), destination_nodes=block.num_dst_nodes(), arcs=block.num_edges())

            identities = [block_identity(block) for block in blocks]
            report["frozen_sampled_block_identity"] = identities
            source_ids = blocks[0].srcdata[dgl.NID].long()
            original32 = graph.ndata["x"][:original_count].detach()
            original64 = original32.to(torch.float64)
            double_gcn = copy.deepcopy(gcn).to(dtype=torch.float64).eval()
            counts["gcn_float64_clones"] += 1
            if (any(parameter.requires_grad or parameter.grad is not None for parameter in double_gcn.parameters())
                    or list(double_gcn.state_dict()) != list(gcn.state_dict())
                    or any(not torch.equal(value, double_gcn.state_dict()[name].to(value.dtype))
                           for name, value in gcn.state_dict().items())):
                raise RuntimeError("Float64 functional GCN clone changed its frozen parameter values.")
            report["float64_clone_state_sha256"] = model_hash(double_gcn)
            report["float64_clone_scope"] = "Same frozen float32 parameter values cast to float64; derivative diagnostic only."

            def fixed_forward(features, precision, finite_difference=False):
                check_deadline()
                if counts["gcn_forward_dispatches"] >= MAX_FORWARDS["gcn"]:
                    raise RuntimeError("Preflight GCN forward cap reached.")
                counts["gcn_forward_dispatches"] += 1
                if precision == "float32":
                    counts["gcn_fixed_float32_center_forwards"] += 1
                    classifier, originals = gcn, original32
                else:
                    classifier, originals = double_gcn, original64
                    key = "gcn_float64_finite_difference_forwards" if finite_difference else "gcn_float64_center_forwards"
                    counts[key] += 1
                    if finite_difference:
                        counts["finite_difference_gcn_forwards"] += 1
                full_features = torch.cat((originals, features), dim=0)
                local_blocks = [block.local_var() for block in blocks]
                local_blocks[0].srcdata["x"] = full_features[source_ids]
                signs = []
                hook = None
                if precision == "float64":
                    def capture_activation(module, values):
                        signs.append((values[0].detach() > 0).clone())
                    hook = double_gcn.activation.register_forward_pre_hook(capture_activation)
                try:
                    logits = checked_logits(classifier(local_blocks, edge_weight=False), len(ids))
                finally:
                    if hook is not None:
                        hook.remove()
                counts["gcn_completed_forwards"] += 1
                if precision == "float64" and len(signs) != 2:
                    raise RuntimeError("Float64 GCN must capture both LeakyReLU preactivation branches.")
                return logits, signs

            u32 = torch.full_like(x0, .5, requires_grad=True)
            value32, _ = fixed_forward(features_from_unit_box(u32, minimum, maximum), "float32")
            loss32 = loss_from_logits(value32)
            margin32 = (value32[:, 1]-value32[:, 0]).mean()
            counts["autograd_calls"] += 1;counts["gcn_fixed_float32_autograd_calls"] += 1
            gradient_loss32 = torch.autograd.grad(loss32, u32, retain_graph=True)[0]
            counts["autograd_calls"] += 1;counts["gcn_fixed_float32_autograd_calls"] += 1
            gradient_margin32 = torch.autograd.grad(margin32, u32)[0]
            for repeat in range(2):
                with torch.no_grad():
                    repeated, _ = fixed_forward(features_from_unit_box(u32.detach(), minimum, maximum), "float32")
                report["float32_center_repeats"].append(dict(repeat=repeat+1,
                    logits=discrepancy(repeated, value32.detach(), LOGIT_ATOL, LOGIT_RTOL),
                    raw_margin_difference=float(((repeated[:, 1]-repeated[:, 0]).mean()-margin32.detach()).abs()),
                    softplus_difference=float((loss_from_logits(repeated)-loss32.detach()).abs())))

            lower64, upper64 = minimum.to(torch.float64), maximum.to(torch.float64)
            u64 = torch.full(x0.shape, .5, dtype=torch.float64, device=x0.device, requires_grad=True)
            value64, center_signs = fixed_forward(features_from_unit_box(u64, lower64, upper64), "float64")
            loss64 = loss_from_logits(value64)
            margin64 = (value64[:, 1]-value64[:, 0]).mean()
            counts["autograd_calls"] += 1;counts["gcn_float64_autograd_calls"] += 1
            gradient_loss64 = torch.autograd.grad(loss64, u64, retain_graph=True)[0]
            counts["autograd_calls"] += 1;counts["gcn_float64_autograd_calls"] += 1
            gradient_margin64 = torch.autograd.grad(margin64, u64)[0]
            gradients = {"raw_margin": (gradient_margin32, gradient_margin64),
                         "actual_softplus": (gradient_loss32, gradient_loss64)}
            comparisons = {}
            for name, (actual, reference) in gradients.items():
                if not bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()):
                    raise RuntimeError("Nonfinite fixed-block GCN feature gradient.")
                if bool((actual[:, zero_width] != 0).any() or (reference[:, zero_width] != 0).any()):
                    raise RuntimeError("A fixed-block zero-width feature has a nonzero gradient.")
                error = (actual.double()-reference).abs()
                tolerance = FD_ATOL+FD_RTOL*torch.maximum(actual.double().abs(), reference.abs())
                comparisons[name] = dict(max_absolute_error=float(error.max()),
                    max_tolerance_fraction=float((error/tolerance).max()),
                    failing_coordinates=int((error > tolerance).sum()), passed=bool((error <= tolerance).all()),
                    float32_gradient_l2=float(actual.detach().double().norm()),
                    float64_gradient_l2=float(reference.detach().norm()))
            report.update(float32_vs_float64_gradients=comparisons,
                float32_center_raw_margin=float(margin32.detach()), float64_center_raw_margin=float(margin64.detach()),
                float32_center_softplus=float(loss32.detach()), float64_center_softplus=float(loss64.detach()),
                center_logit_max_absolute_difference=float((value32.detach().double()-value64.detach()).abs().max()),
                center_leaky_relu_sign_patterns=[dict(layer=layer+1,
                    sha256=tensor_sha256(signs), elements=signs.numel(), positive_count=int(signs.sum()))
                    for layer, signs in enumerate(center_signs)],
                zero_width_gradients_exact_zero=True)
            flat32 = gradient_margin32.detach().reshape(-1).cpu().tolist()
            indices = sorted((index for index, value in enumerate(flat32) if value != 0),
                             key=lambda index: (-abs(flat32[index]), index))[:4]
            if len(indices) != 4:
                raise RuntimeError("Fewer than four nonzero actual float32 raw-margin coordinates.")
            reference_gradients = {name:pair[1].detach().reshape(-1).cpu().tolist() for name, pair in gradients.items()}
            for index in indices:
                coordinate = dict(flat_coordinate=index, actual_float32_raw_margin_gradient=flat32[index], grid=[])
                report["coordinates"].append(coordinate)
                for h in GCN_FD_GRID:
                    with torch.no_grad():
                        plus, minus = u64.detach().clone(), u64.detach().clone()
                        plus.reshape(-1)[index] += h;minus.reshape(-1)[index] -= h
                        positive, positive_signs = fixed_forward(features_from_unit_box(plus, lower64, upper64), "float64", True)
                        negative, negative_signs = fixed_forward(features_from_unit_box(minus, lower64, upper64), "float64", True)
                        values = dict(raw_margin=(float((positive[:, 1]-positive[:, 0]).mean()),
                                                  float((negative[:, 1]-negative[:, 0]).mean())),
                                      actual_softplus=(float(loss_from_logits(positive)), float(loss_from_logits(negative))))
                    branch_changes = [dict(layer=layer+1,
                        plus_changed=int((positive_signs[layer] != center_signs[layer]).sum()),
                        minus_changed=int((negative_signs[layer] != center_signs[layer]).sum()),
                        center_sha256=tensor_sha256(center_signs[layer]),
                        plus_sha256=tensor_sha256(positive_signs[layer]),
                        minus_sha256=tensor_sha256(negative_signs[layer])) for layer in range(2)]
                    stable = all(row["plus_changed"] == row["minus_changed"] == 0 for row in branch_changes)
                    tests = {}
                    for name, (positive_value, negative_value) in values.items():
                        automatic = reference_gradients[name][index]
                        numerical = (positive_value-negative_value)/(2*h)
                        error = abs(numerical-automatic)
                        tolerance = FD_ATOL+FD_RTOL*max(abs(numerical), abs(automatic))
                        tests[name] = dict(autograd=automatic, plus_value=positive_value,
                            minus_value=negative_value, central_difference=numerical,
                            absolute_error=error, tolerance=tolerance,
                            passed=math.isfinite(numerical) and error <= tolerance)
                    coordinate["grid"].append(dict(h=h, objectives=tests,
                        leaky_relu_branch_changes=branch_changes, stable_branches=stable))
                coordinate["finest_two_passed"] = all(row["stable_branches"] and
                    all(test["passed"] for test in row["objectives"].values()) for row in coordinate["grid"][-2:])
            if [block_identity(block) for block in blocks] != identities:
                raise RuntimeError("Frozen sampled block identities changed during GCN derivative checks.")
            report["frozen_block_identity_retained"] = True
            report["finest_two_all_coordinates_passed"] = all(row["finest_two_passed"] for row in report["coordinates"])
            if (not all(row["passed"] for row in comparisons.values())
                    or not report["finest_two_all_coordinates_passed"]
                    or not all(row["logits"]["passed"] for row in report["float32_center_repeats"])):
                raise RuntimeError("Fixed-block float64 GCN functional check did not satisfy the unchanged gradient tolerances and finest-two branch-stability rule.")
            report["status"] = "passed"

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
            row["gcn_feature_gradient"] = {}
            fixed_block_gcn_gradient_check(graph, ids, x0, row["gcn_feature_gradient"])
            engine.capture = engine.live_capture = None
            del graph, affine, base_sequence, x0, candidates, injected
        if (counts["generator_forwards"] != 4 or counts["gaga_completed_forwards"] != 60
                or counts["gcn_completed_forwards"] != 184 or counts["autograd_calls"] != 24
                or counts["canonical_sequence_updates"] != 12
                or counts["finite_difference_gaga_forwards"] != 32
                or counts["finite_difference_gcn_forwards"] != 160
                or counts["gcn_native_sample_block_calls"] != 8 or counts["gcn_fixed_block_sample_calls"] != 4
                or counts["gcn_fixed_float32_center_forwards"] != 12 or counts["gcn_float64_center_forwards"] != 4
                or counts["gcn_float64_finite_difference_forwards"] != 160 or counts["gcn_float64_clones"] != 4
                or counts["gcn_fixed_float32_autograd_calls"] != 8 or counts["gcn_float64_autograd_calls"] != 8):
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
                                      finite_difference_rtol=FD_RTOL,
                                      gcn_functional_float64_grid=list(GCN_FD_GRID)),
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
