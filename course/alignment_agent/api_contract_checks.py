"""Synthetic stdlib checks only; no SSH, API, data, model or credential access."""
import copy
import json
import unittest

from course.alignment_agent.api_bridge import (ContractError, PROMPT, fallback_rule,
    prepare_request, resolve_decision, sha, strict_json, validate_decision,
    validate_envelope, validate_state)


def state():
    return dict(completed_epoch=1, train_groups=256, target_occurrences=2048,
        margin_mean=2.0, margin_median=1.0, margin_positive_fraction=.8,
        hinge_mean=.2, softplus_mean=.3, distance_mean=1.2, outside_fraction=.2,
        reference_quantile=.9, gradient_probe_groups=8, gradient_cosine_median=.1,
        gradient_norm_ratio_median=.5, gradient_missing_count=0, prior_actions=[False])


def envelope():
    return dict(schema="alignment_training_controller_v2", request_id="opaque_abc",
                protocol_sha256="a" * 64, seed=3, epoch=2, state=state())


def protocol():
    return dict(source_commit="b" * 40, source_files={"synthetic.py": "c" * 64},
                api=dict(model="deepseek-flash", model_version="declared-test-version"))


class ControllerContractChecks(unittest.TestCase):
    def test_known_fallback(self):
        self.assertTrue(fallback_rule(state()))

    def test_exact_ten_percent_boundary_is_off(self):
        s = state(); s["outside_fraction"] = .1
        self.assertFalse(fallback_rule(s))
        s["outside_fraction"] = .10000001
        self.assertTrue(fallback_rule(s))

    def test_missing_or_conflicting_gradient_is_not_safe(self):
        for value in (None, -.01):
            s = state(); s["gradient_cosine_median"] = value
            self.assertFalse(fallback_rule(s))
        s = state(); s["gradient_missing_count"] = 8
        self.assertFalse(fallback_rule(s))

    def test_zero_cosine_valid_boundary(self):
        s = state(); s["gradient_cosine_median"] = 0
        self.assertTrue(fallback_rule(s))

    def test_unknown_or_nested_state_fields_rejected(self):
        for key, value in (("node_ids", [1]), ("validation_accuracy", .9), ("margin_mean", {"label": 1})):
            s = state(); s[key] = value
            with self.assertRaises(ContractError):
                validate_state(s)

    def test_bad_numeric_types_ranges_and_nonfinite(self):
        for key, value in (("train_groups", True), ("outside_fraction", True), ("outside_fraction", 1.01),
                           ("gradient_cosine_median", float("nan")), ("distance_mean", -1),
                           ("gradient_norm_ratio_median", float("inf")), ("gradient_missing_count", 9)):
            s = state(); s[key] = value
            with self.assertRaises(ContractError):
                validate_state(s)

    def test_prior_actions_and_epoch_consistency(self):
        for actions in ([True], [], [0]):
            s = state(); s["prior_actions"] = actions
            with self.assertRaises(ContractError):
                validate_state(s)
        e = envelope(); e["epoch"] = 3
        with self.assertRaises(ContractError):
            validate_envelope(e, "a" * 64)

    def test_protocol_and_identity_envelope(self):
        for key, value in (("seed", True), ("seed", 9), ("request_id", "../unsafe"), ("protocol_sha256", "d" * 64)):
            e = envelope(); e[key] = value
            with self.assertRaises(ContractError):
                validate_envelope(e, "a" * 64)

    def test_no_local_identifiers_in_outbound_messages(self):
        request = prepare_request(envelope(), protocol(), "a" * 64)["request"]
        payload = json.loads(request["messages"][1]["content"])
        self.assertEqual(set(payload), set(state()))
        self.assertNotIn("opaque_abc", json.dumps(request["messages"]))
        self.assertNotIn("a" * 64, json.dumps(request["messages"]))
        self.assertEqual(request["max_tokens"], 512)

    def test_full_cache_identity(self):
        original = prepare_request(envelope(), protocol(), "a" * 64)["cache_key"]
        for change in ("state", "seed", "source", "model", "protocol", "epoch"):
            e, p, digest = envelope(), protocol(), "a" * 64
            if change == "state": e["state"]["margin_mean"] = 3
            if change == "seed": e["seed"] = 4
            if change == "source": p["source_files"]["synthetic.py"] = "f" * 64
            if change == "model": p["api"]["model_version"] = "other-declared-version"
            if change == "protocol": digest = "f" * 64; e["protocol_sha256"] = digest
            if change == "epoch": e["epoch"] = 3; e["state"]["completed_epoch"] = 2; e["state"]["prior_actions"].append(True)
            self.assertNotEqual(original, prepare_request(e, p, digest)["cache_key"])

    def test_decision_exact_schema_and_boolean(self):
        self.assertTrue(validate_decision('{"align":true,"reason":"TRAIN diagnostics support activation."}')["align"])
        for raw in ('{"align":1,"reason":"No."}', '{"align":true,"reason":"No.","extra":0}',
                    '{"align":true,"align":false,"reason":"No."}', '{"align":NaN,"reason":"No."}',
                    json.dumps(dict(align=True, reason=chr(233))), '{"align":true,"reason":"123"}'):
            with self.assertRaises(ContractError):
                validate_decision(raw)

    def test_fallback_preserves_each_valid_epoch(self):
        rows = [resolve_decision(envelope(), raw, "a" * 64) for raw in (None, "", "{}", "not JSON")]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["fallback"] and row["align"] for row in rows))
        self.assertTrue(all(row["request_id"] == "opaque_abc" for row in rows))

    def test_input_not_mutated(self):
        e = envelope(); before = copy.deepcopy(e)
        checked = validate_envelope(e, "a" * 64)
        checked["state"]["prior_actions"].append(True)
        self.assertEqual(e, before)

    def test_json_size_and_duplicates(self):
        for raw in ('{"a":1,"a":2}', '{"a":Infinity}', "[" + "0," * 10000 + "0]"):
            with self.assertRaises(ContractError):
                strict_json(raw)

    def test_invalid_input_rejected_before_fallback(self):
        e = envelope(); e["state"]["hidden_label"] = 1
        with self.assertRaises(ContractError):
            resolve_decision(e, None, "a" * 64)


if __name__ == "__main__":
    unittest.main()
