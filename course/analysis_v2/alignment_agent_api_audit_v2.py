"""Post-hoc saved-record recount; stdlib only, with no API/SSH/ML/secret access.

The author also implemented the bridge. This is a separate record recount from
root execution, not independent implementation review or human verification.
Use a completed local metadata snapshot. No dataset/checkpoint/test files read.
"""
import argparse
import ast
import collections
import datetime as dt
from decimal import Decimal
import hashlib
import json
import math
import pathlib
import random
import statistics
import subprocess
import time


ARMS = ("legacy_hinge_off", "smooth_off", "smooth_threshold", "smooth_rule", "smooth_agent", "smooth_random")
CAPS = dict(max_requests=120, prompt_tokens=200000, completion_tokens=64000, usd=.5)
EXPECTED_SLOTS = {(seed, epoch) for seed in (3, 4, 5) for epoch in range(2, 9)}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate saved JSON key")
            result[key] = value
        return result
    def bad(value):
        raise ValueError("Nonfinite JSON constant")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def distribution(values):
    values = list(values)
    return dict(n=len(values), total=sum(values), mean=statistics.mean(values),
                median=statistics.median(values), maximum=max(values)) if values else dict(n=0, total=0)


def iso(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def response_diagnostics(raw):
    """Describe format failures without accepting a rejected suggested action."""
    result = dict(json_valid=False, exact_schema=False, action_reason_types_valid=False,
                  suggested_align=None, reason_characters_stripped=None, valid=False, invalid_categories=[])
    bad = result["invalid_categories"]
    if type(raw) is not str:
        bad.append("missing_or_non_string_content"); return result
    try:
        encoded = raw.encode("utf-8")
    except UnicodeError:
        bad.append("invalid_unicode"); return result
    if len(encoded) > 4096: bad.append("response_exceeds_4096_bytes")
    try:
        parsed = strict(raw)
    except (ValueError, TypeError, RecursionError):
        bad.append("invalid_json_duplicate_nonfinite_or_syntax"); return result
    result["json_valid"] = True
    if type(parsed) is not dict:
        bad.append("json_not_object"); return result
    result["exact_schema"] = set(parsed) == {"align", "reason"}
    if not result["exact_schema"]: bad.append("extra_or_missing_properties")
    if type(parsed.get("align")) is bool:
        result["suggested_align"] = parsed["align"]
    else: bad.append("align_not_boolean")
    reason = parsed.get("reason")
    result["action_reason_types_valid"] = type(parsed.get("align")) is bool and type(reason) is str
    if type(reason) is not str:
        bad.append("reason_not_string")
    else:
        result["reason_characters_stripped"] = len(reason.strip())
        if not reason.strip(): bad.append("reason_empty")
        if len(reason.strip()) > 240: bad.append("reason_exceeds_240_characters")
        if any(ord(c) < 32 or ord(c) > 126 for c in reason): bad.append("reason_not_printable_ascii")
        if not any(c.isascii() and c.isalpha() for c in reason): bad.append("reason_without_ascii_letter")
    result["valid"] = not bad
    return result


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "protocol", "api-dir", "training-snapshot", "output"):
        parser.add_argument("--" + name, required=True, type=pathlib.Path)
    parser.add_argument("--resume-dir", type=pathlib.Path, action="append", default=[],
                        help="Repeat in chronological order for every recovery attempt")
    parser.add_argument("--request-snapshot", type=pathlib.Path,
                        help="Optional local stage requests directory, separate from job metadata")
    parser.add_argument("--audit-amendment", type=pathlib.Path,
                        help="Optional disclosed correction of a previous saved-record audit predicate")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    repo, protocol_path, api, training, output = (x.resolve() for x in
        (args.repo, args.protocol, args.api_dir, args.training_snapshot, args.output))
    resumes = [path.resolve() for path in args.resume_dir]
    resume = resumes[-1] if resumes else None
    request_snapshot = args.request_snapshot.resolve() if args.request_snapshot is not None else training / "requests"
    if output == repo or repo in output.parents:
        raise RuntimeError("Private audit output must stay outside the public repository")
    paths = {protocol_path}
    audit_amendment = args.audit_amendment.resolve() if args.audit_amendment is not None else None
    if audit_amendment is not None: paths.add(audit_amendment)
    for name in ("registered.json", "bridge_complete.json", "api_attempts.jsonl", "bridge_events.jsonl"):
        if (api / name).exists(): paths.add(api / name)
    for recovery in resumes:
        for name in ("registered.json", "infrastructure_amendment.json", "remote_claim_receipt.json", "bridge_complete.json", "bridge_events.jsonl"):
            if (recovery / name).exists(): paths.add(recovery / name)
    if resumes:
        for name in ("resume_bridge.py", "resume_bridge_v2.py"):
            if (api.parent / name).exists(): paths.add(api.parent / name)
    for name in ("requests", "responses", "cache"):
        paths.update((api / name).glob("*.json"))
    paths.update(request_snapshot.glob("*.json"))
    for arm in ARMS:
        for seed in (3, 4, 5):
            directory = training / ("%s_seed%d" % (arm, seed))
            for name in ("result.json", "outer_actions.json"):
                if (directory / name).exists(): paths.add(directory / name)
            if arm == "smooth_agent":
                paths.update(directory.glob("train_state_epoch_*.json"))
                paths.update(directory.glob("applied_controller_epoch_*.json"))
    paths.update(training.glob("matched_schedule_seed*.json"))
    if (training / "queue_complete.json").exists(): paths.add(training / "queue_complete.json")
    content = {path: path.read_bytes() for path in sorted(paths)}
    manifest = {str(path): sha(raw) for path, raw in content.items()}
    output.mkdir(parents=True, exist_ok=False)
    write(output / "registered.json", dict(audit_type="POST-HOC saved-record recount", source_sha256=sha(pathlib.Path(__file__).read_bytes()),
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), input_files=len(manifest),
        input_manifest_sha256=sha(canonical(manifest).encode()),
        audit_logic_amendment_sha256=manifest.get(str(audit_amendment)) if audit_amendment is not None else None,
        attribution="Same agent authored the API bridge; separate saved-record recount from root execution, not independent implementation review or human verification.",
        infrastructure_recovery_included=bool(resumes), recovery_attempts=len(resumes),
        allow_incomplete=args.allow_incomplete, model_calls=0, api_calls=0, ssh_calls=0))
    write(output / "input_manifest_private.json", manifest)
    failures, checks = [], []
    def check(name, condition):
        checks.append(dict(check=name, passed=bool(condition)))
        if not condition: failures.append(name)
    def read(path):
        return strict(content[path].decode("utf-8-sig"))
    def lines(path):
        return [strict(line) for line in content.get(path, b"").decode("utf-8-sig").splitlines()]
    protocol = read(protocol_path)
    protocol_sha = manifest[str(protocol_path)]
    commit = protocol["source_commit"]
    source_bytes = subprocess.check_output(["git", "-C", str(repo), "show", commit + ":course/alignment_agent/api_bridge.py"])
    source_ast = ast.parse(source_bytes.decode("utf-8"))
    assignments = {node.targets[0].id: node.value for node in source_ast.body
                   if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
    prompt = ast.literal_eval(assignments["PROMPT"])
    fields = set(ast.literal_eval(assignments["FIELDS"].args[0]))
    prepare = next(node for node in source_ast.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_request")
    identity_call = next(node.value for node in prepare.body if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == "identity" for target in node.targets))
    fallback_identity = ast.literal_eval(next(kw.value for kw in identity_call.keywords if kw.arg == "fallback"))
    frozen_sources = {}
    for name in ("course/alignment_agent/api_bridge.py", "course/alignment_agent/controller.py", "course/llm_fusion/invoke_batch.ps1"):
        raw = subprocess.check_output(["git", "-C", str(repo), "show", commit + ":" + name])
        frozen_sources[name] = sha(raw)
        check("frozen_source_" + name, sha(raw) == protocol["source_files"][name])
    check("protocol_frozen_authorized", protocol["status"] == "frozen" and protocol["owner_execution_authorized"] is True)
    check("new_stage_caps", protocol["api"]["caps"] == CAPS and protocol["api"]["max_logical_requests"] == 21)
    check("frozen_prompt_hash", sha(prompt.encode()) == protocol["api"]["prompt_sha256"])
    registration = read(api / "registered.json")
    check("bridge_registration_identity", registration["protocol_sha256"] == protocol_sha and registration["source_commit"] == commit
          and registration["prompt_sha256"] == sha(prompt.encode()) and registration["caps"] == CAPS)
    original_events = lines(api / "bridge_events.jsonl")
    recovery_events = []
    recovery_chain = []
    recovery_details = []
    original_complete = read(api / "bridge_complete.json") if (api / "bridge_complete.json") in content else None
    if original_complete is not None:
        original_usage = original_complete.get("usage", {})
        check("original_phase_has_no_unknown_attempts", original_usage.get("unrecorded_attempt_upper_bound", 0) == 0
              and original_usage.get("status") not in ("ledger_unreadable_preserved_for_audit", "ledger_unreadable_preserved_accounting_uncertain"))
    amendment = None
    original_prefix_rows = []
    for index, recovery in enumerate(resumes):
        amendment = read(recovery / "infrastructure_amendment.json")
        resume_registration = read(recovery / "registered.json")
        claim = read(recovery / "remote_claim_receipt.json")
        stage_events = lines(recovery / "bridge_events.jsonl")
        recovery_events.extend(stage_events)
        check("recovery_amendment_protocol_source", amendment["protocol_sha256"] == protocol_sha and amendment["scientific_source_commit"] == commit)
        check("recovery_original_failure_retained", original_complete is not None and original_complete["status"] == "bridge_failed"
              and amendment["old_completion_sha256"] == manifest[str(api / "bridge_complete.json")])
        source_name = "resume_bridge_v2.py" if amendment["schema"].endswith("_v2") else "resume_bridge.py"
        check("recovery_source_hash", amendment["recovery_source_sha256"] == manifest.get(str(api.parent / source_name)))
        check("recovery_registration_and_claim", resume_registration["protocol_sha256"] == claim["protocol_sha256"] == protocol_sha
              and resume_registration["amendment_sha256"] == claim["amendment_sha256"] == manifest[str(recovery / "infrastructure_amendment.json")]
              and resume_registration["recovery_nonce"] == claim["recovery_nonce"])
        if index:
            previous = resumes[index-1]
            previous_registration = read(previous / "registered.json")
            previous_complete = read(previous / "bridge_complete.json")
            check("prior_failed_recovery_retained", previous_complete["status"] == "resume_failed"
                  and previous_complete["protocol_sha256"] == protocol_sha)
            check("recovery_chain_prior_receipt_amendment_nonce", amendment["prior_recovery_completion_sha256"] == claim["prior_recovery_completion_sha256"] == manifest[str(previous / "bridge_complete.json")]
                  and amendment["prior_recovery_amendment_sha256"] == claim["prior_recovery_amendment_sha256"] == manifest[str(previous / "infrastructure_amendment.json")]
                  and amendment["prior_recovery_nonce"] == claim["prior_recovery_nonce"] == previous_registration["recovery_nonce"]
                  and claim["original_bridge_nonce"] == read(previous / "remote_claim_receipt.json")["original_bridge_nonce"])
            check("recovery_order_chronological", iso(amendment["registered_utc"]) >= iso(read(previous / "infrastructure_amendment.json")["registered_utc"]))
        prefix = b""
        prefix_rows = []
        for line in content.get(api / "api_attempts.jsonl", b"").splitlines(keepends=True):
            prefix += line
            if sha(prefix) == amendment["old_ledger_sha256"]:
                prefix_rows = [strict(item) for item in prefix.decode("utf-8-sig").splitlines()]
                break
        check("recovery_preserved_original_ledger_prefix", bool(prefix_rows))
        if not index: original_prefix_rows = prefix_rows
        recovery_details.append((recovery, amendment, stage_events, prefix_rows))
        stage_complete_path = recovery / "bridge_complete.json"
        if stage_complete_path in content:
            stage_usage = read(stage_complete_path).get("usage", {})
            check("recovery_phase_has_no_unknown_attempts", stage_usage.get("unrecorded_attempt_upper_bound", 0) == 0
                  and stage_usage.get("status") not in ("ledger_unreadable_preserved_for_audit", "ledger_unreadable_preserved_accounting_uncertain"))
        recovery_chain.append(dict(attempt=index+1, schema=amendment["schema"],
            status=read(stage_complete_path)["status"] if stage_complete_path in content else "incomplete",
            amendment_sha256=manifest[str(recovery / "infrastructure_amendment.json")],
            completion_sha256=manifest.get(str(stage_complete_path)), recovery_source_sha256=amendment["recovery_source_sha256"],
            physical_attempts_existing_at_registration=sum(row.get("attempted") is True for row in prefix_rows)))
    envelopes, requests, private_ids = {}, {}, set()
    numeric_ranges = {"margin_positive_fraction": (0, 1), "outside_fraction": (0, 1),
        "reference_quantile": (.9, .9), "gradient_cosine_median": (-1, 1),
        "gradient_norm_ratio_median": (0, 1e12), "hinge_mean": (0, 1e6), "softplus_mean": (0, 1e6), "distance_mean": (0, 1e6)}
    integer_fields = {"completed_epoch", "train_groups", "target_occurrences", "gradient_probe_groups", "gradient_missing_count"}
    envelope_paths = sorted((api / "requests").glob("*.json")) + sorted(request_snapshot.glob("*.json"))
    envelope_filenames = {}
    for path in envelope_paths:
        if path.name.startswith("api_"): continue
        envelope = read(path)
        check("envelope_exact_schema", set(envelope) == {"schema", "request_id", "protocol_sha256", "seed", "epoch", "state"})
        check("envelope_filename_identity", type(envelope["request_id"]) is str and path.name == envelope["request_id"] + ".json")
        if path.name in envelope_filenames:
            check("duplicate_filename_envelope_identical", envelope == envelope_filenames[path.name])
        else:
            envelope_filenames[path.name] = envelope
        state, seed, epoch = envelope["state"], envelope["seed"], envelope["epoch"]
        if (seed, epoch) in envelopes:
            check("duplicate_snapshot_envelope_identical", envelope == envelopes[seed, epoch])
            continue
        check("allowlisted_train_only_state", type(state) is dict and set(state) == fields)
        check("slot_and_protocol", envelope["schema"] == "alignment_training_controller_v2" and envelope["protocol_sha256"] == protocol_sha
              and type(seed) is int and type(epoch) is int and (seed, epoch) in EXPECTED_SLOTS and state["completed_epoch"] == epoch - 1)
        check("state_integer_types", all(type(state[key]) is int for key in integer_fields))
        check("state_prior_actions", len(state["prior_actions"]) == epoch - 1 and all(type(x) is bool for x in state["prior_actions"]) and state["prior_actions"][0] is False)
        for key in fields - integer_fields - {"prior_actions"}:
            value = state[key]
            nullable = key in ("gradient_cosine_median", "gradient_norm_ratio_median")
            okay = value is None and nullable
            if type(value) in (int, float) and math.isfinite(value):
                bounds = numeric_ranges.get(key, (-1e6, 1e6))
                okay = bounds[0] <= value <= bounds[1]
            check("numeric_state_" + key, okay)
        check("probe_count_consistency", 0 <= state["gradient_missing_count"] <= state["gradient_probe_groups"] <= 8)
        check("training_state_saved_match", read(training / ("smooth_agent_seed%d" % seed) / ("train_state_epoch_%03d.json" % (epoch - 1))) == state)
        slot = (seed, epoch)
        check("unique_logical_slot", slot not in envelopes)
        envelopes[slot] = envelope
        private_ids.add(envelope["request_id"])
        identity = dict(state=state, seed=seed, next_epoch=epoch, source_commit=commit,
            source_files=protocol["source_files"], model=protocol["api"]["model"], model_version=protocol["api"]["model_version"],
            protocol_sha256=protocol_sha, prompt_sha256=sha(prompt.encode()),
            settings=dict(temperature=0, thinking="disabled", max_tokens=512), fallback=fallback_identity)
        key = sha(canonical(identity).encode())
        expected = dict(request_id="controller_" + key, max_tokens=512,
            messages=[dict(role="system", content=prompt), dict(role="user", content=canonical(state))])
        api_request_path = api / "requests" / ("api_" + path.name)
        if api_request_path in content:
            check("prepared_request_exact_no_local_identifiers", read(api_request_path) == expected)
        requests[expected["request_id"]] = (slot, expected, key)
    check("logical_slot_ceiling", set(envelopes) <= EXPECTED_SLOTS)
    if not args.allow_incomplete: check("all_twenty_one_logical_slots", set(envelopes) == EXPECTED_SLOTS)
    ledger = lines(api / "api_attempts.jsonl")
    physical, grouped, latency = [], collections.defaultdict(list), []
    prompt_total = completion_total = 0
    fatal_seen = False
    cache_hit = cache_miss = 0
    for row in ledger:
        check("ledger_protocol_identity", row.get("protocol_sha256") == protocol_sha)
        check("ledger_known_request", row.get("request_id") in requests)
        if row.get("attempted") is not True: continue
        check("no_calls_after_auth_or_reservation_fatal", not fatal_seen)
        request = requests[row["request_id"]][1]
        expected_body = dict(model="deepseek-flash", thinking=dict(type="disabled"), temperature=0,
            stream=False, response_format=dict(type="json_object"), max_tokens=512, messages=request["messages"])
        check("physical_outbound_exact_frozen_prompt_state_controls", row["request"] == expected_body)
        rp, rc = row["reserved_prompt_tokens"], row["reserved_completion_tokens"]
        check("reservation_before_attempt", len(physical) < 120 and prompt_total + rp <= 200000 and completion_total + rc <= 64000
              and Decimal(prompt_total + rp) * Decimal(".00000030") + Decimal(completion_total + rc) * Decimal(".00000120") <= Decimal(".50"))
        check("conservative_reservation", rp >= len(canonical(expected_body).encode()) + 1024 and rc == 512)
        cp, cc = row["charged_prompt_tokens"], row["charged_completion_tokens"]
        check("charged_token_types", type(cp) is int and cp >= 0 and type(cc) is int and cc >= 0)
        if row.get("usage_verified"):
            usage = row["usage"]
            check("actual_usage_recount", usage["prompt_tokens"] == cp and usage["completion_tokens"] == cc
                  and usage["total_tokens"] == cp + cc and cp <= rp and cc <= rc)
            if "prompt_cache_hit_tokens" in usage and "prompt_cache_miss_tokens" in usage:
                check("cache_token_recount", usage["prompt_cache_hit_tokens"] + usage["prompt_cache_miss_tokens"] == cp)
                cache_hit += usage["prompt_cache_hit_tokens"]; cache_miss += usage["prompt_cache_miss_tokens"]
        else:
            check("unknown_usage_keeps_reservation", cp == rp and cc == rc)
        charge = cp * .30 / 1e6 + cc * 1.20 / 1e6
        check("per_attempt_cost_recount", math.isclose(row["charged_usd_upper_bound"], charge, abs_tol=1e-12))
        prompt_total += cp; completion_total += cc
        duration = (iso(row["completed_utc"]) - iso(row["started_utc"])).total_seconds()
        check("nonnegative_attempt_latency", duration >= 0)
        latency.append(duration)
        physical.append(row); grouped[row["request_id"]].append(row)
        fatal_seen |= row.get("http_status") in (401, 403) or row.get("reservation_exceeded") is True
        if row.get("status") == "completed":
            check("actual_alias_and_response_id", row["model"] == "deepseek-flash" and type(row["response_id"]) is str and bool(row["response_id"]))
    for rows in grouped.values():
        check("at_most_one_eligible_retry", len(rows) <= 2 and [row["attempt"] for row in rows] == list(range(1, len(rows) + 1)))
        if len(rows) == 2:
            check("retry_only_429_or_5xx", rows[0].get("status") == "failed" and (rows[0].get("http_status") == 429 or 500 <= (rows[0].get("http_status") or 0) <= 599))
    check("cumulative_hard_caps", len(physical) <= 120 and prompt_total <= 200000 and completion_total <= 64000
          and prompt_total * .30 / 1e6 + completion_total * 1.20 / 1e6 <= .5)
    if recovery_details:
        old_attempts = [row for row in original_prefix_rows if row.get("attempted") is True]
        old_usage = original_complete.get("usage", {})
        check("historical_failed_bridge_usage_recount", old_usage.get("physical_attempts_recorded") == len(old_attempts)
              and old_usage.get("charged_prompt_tokens") == sum(row["charged_prompt_tokens"] for row in old_attempts)
              and old_usage.get("charged_completion_tokens") == sum(row["charged_completion_tokens"] for row in old_attempts))
        for index, (directory, stage_amendment, stage_events, prefix_rows) in enumerate(recovery_details):
            prefix_attempts = [row for row in prefix_rows if row.get("attempted") is True]
            check("recovery_amendment_before_additional_calls", all(iso(row["started_utc"]) >= iso(stage_amendment["registered_utc"])
                  for row in physical[len(prefix_attempts):]))
            if index:
                previous_usage = read(resumes[index-1] / "bridge_complete.json").get("usage", {})
                check("prior_failed_recovery_usage_recount", previous_usage.get("physical_attempts_combined") == len(prefix_attempts)
                      and previous_usage.get("charged_prompt_tokens_combined") == sum(row["charged_prompt_tokens"] for row in prefix_attempts)
                      and previous_usage.get("charged_completion_tokens_combined") == sum(row["charged_completion_tokens"] for row in prefix_attempts))
    categories, waits, on_counts = collections.Counter(), [], {}
    format_counts, format_failures, suggestions, rejected_suggestions = (collections.Counter() for _ in range(4))
    reason_lengths = []
    per_seed_applications = collections.defaultdict(collections.Counter)
    past_skipped_slots = set()
    new_attempts_after_past_observations = 0
    combined_events = original_events + recovery_events
    check("no_phase_reported_unrecorded_attempts", not any(row.get("physical_attempt_upper_bound_unconfirmed", 0) > 0 for row in combined_events))
    published = {row["request_id"] for row in combined_events if row.get("status") == "response_published"}
    if recovery_details:
        by_envelope_id = {envelope["request_id"]: slot for slot, envelope in envelopes.items()}
        for directory, stage_amendment, stage_events, prefix_rows in recovery_details:
            for observation in (row for row in stage_events if row.get("status") == "past_slot_skipped"):
                request_id = observation["request_id"]
                slot = by_envelope_id[request_id]
                past_skipped_slots.add(slot)
                logical_id = next(key for key, value in requests.items() if value[0] == slot)
                later_attempts = [row for row in grouped.get(logical_id, []) if iso(row["started_utc"]) >= iso(observation["utc"])]
                new_attempts_after_past_observations += len(later_attempts)
                check("no_new_api_after_past_slot_skip_observation", not later_attempts)
    for (seed, epoch), envelope in sorted(envelopes.items()):
        request_id, state = envelope["request_id"], envelope["state"]
        response_path = api / "responses" / (request_id + ".json")
        applied_path = training / ("smooth_agent_seed%d" % seed) / ("applied_controller_epoch_%02d.json" % epoch)
        logical_id = next(key for key, value in requests.items() if value[0] == (seed, epoch))
        successful = [row for row in grouped.get(logical_id, []) if row.get("status") == "completed" and row.get("finish_reason") == "stop" and row.get("model") == "deepseek-flash"]
        raw = None
        if successful:
            try:
                raw = successful[-1]["response"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                pass  # Match the frozen bridge: missing content uses fallback.
        description = response_diagnostics(raw)
        valid, suggested = description["valid"], description["suggested_align"]
        if successful:
            format_counts["eligible_completed_stop_outputs"] += 1
            for key in ("json_valid", "exact_schema", "action_reason_types_valid"):
                format_counts[key] += description[key]
            format_counts["frozen_contract_valid" if valid else "frozen_contract_invalid"] += 1
            format_failures.update(description["invalid_categories"])
            if description["reason_characters_stripped"] is not None:
                reason_lengths.append(description["reason_characters_stripped"])
            if suggested is not None:
                suggestions["true" if suggested else "false"] += 1
                if not valid:
                    format_counts["valid_json_boolean_action_rejected"] += 1
                    rejected_suggestions["true" if suggested else "false"] += 1
        if applied_path not in content:
            categories["missing_applied_record"] += 1
            check("applied_record_present", args.allow_incomplete)
            continue
        applied = read(applied_path)
        if successful and not valid and suggested is not None:
            format_counts["rejected_boolean_suggestion_with_recorded_applied_action"] += 1
            format_counts["rejected_boolean_suggestion_differs_from_applied_flag"] += suggested != applied["align"]
            format_counts["rejected_boolean_suggestion_applied_fallback"] += applied["fallback"] is True
        expected_state_sha = sha(canonical(state).encode())
        check("applied_identity", applied["request_id"] == request_id and applied["protocol_sha256"] == protocol_sha
              and applied["state_sha256"] == expected_state_sha and applied["epoch"] == epoch and type(applied["align"]) is bool)
        waits.append(applied["wait_seconds"])
        fallback = state["outside_fraction"] > float(Decimal("1") - Decimal(str(state["reference_quantile"]))) and state["gradient_probe_groups"] > state["gradient_missing_count"]
        response = read(response_path) if response_path in content else None
        if applied["fallback"]:
            categories["applied_fallback"] += 1
            per_seed_applications[str(seed)]["fallback"] += 1
            check("applied_registered_protected_fallback", applied["align"] == fallback and applied["decision_source"] == "registered_fallback")
            if applied.get("timeout_or_invalid_response"):
                categories["server_timeout_or_invalid_response"] += 1
                if response is not None and not response["fallback"]:
                    categories["valid_bridge_response_not_applied_after_server_timeout_or_invalid"] += 1
        else:
            categories["applied_external_agent"] += 1
            per_seed_applications[str(seed)]["external_agent"] += 1
            check("applied_matches_published_response", response is not None and request_id in published
                  and all(applied[key] == response[key] for key in ("align", "fallback", "response_sha256", "state_sha256", "reason")))
        if response is not None:
            check("response_identity", response["request_id"] == request_id and response["protocol_sha256"] == protocol_sha and response["state_sha256"] == expected_state_sha)
            expected_hash = sha(raw.encode("utf-8", errors="surrogatepass") if type(raw) is str else b"")
            check("response_raw_hash_and_contract", response["response_sha256"] == expected_hash and response["fallback"] == (not valid)
                  and response["align"] == (suggested if valid else fallback))
    completed_jobs = 0
    for arm in ARMS:
        for seed in (3, 4, 5):
            directory = training / ("%s_seed%d" % (arm, seed))
            if (directory / "result.json") not in content:
                check("matrix_job_result_present", args.allow_incomplete); continue
            result = read(directory / "result.json")
            completed_jobs += result.get("status") == "completed"
            check("matrix_job_completed", result.get("status") == "completed" or args.allow_incomplete)
    for seed in (3, 4, 5):
        agent_path = training / ("smooth_agent_seed%d" % seed) / "outer_actions.json"
        random_path = training / ("smooth_random_seed%d" % seed) / "outer_actions.json"
        schedule_path = training / ("matched_schedule_seed%d.json" % seed)
        if any(path not in content for path in (agent_path, random_path, schedule_path)):
            check("schedule_records_present", args.allow_incomplete); continue
        agent, shuffled, schedule = read(agent_path), read(random_path), read(schedule_path)
        actions = agent["actions"]
        tail = list(actions[1:]); random.Random(20261103 + seed).shuffle(tail)
        check("matched_actual_schedule", agent["protocol_sha256"] == shuffled["protocol_sha256"] == schedule["protocol_sha256"] == protocol_sha
              and len(actions) == 8 and actions[0] is False and schedule["actions"] == shuffled["actions"] == [False] + tail
              and sum(actions) == sum(shuffled["actions"]) == schedule["source_enabled_epochs"])
        for epoch in range(2, 9):
            applied = read(training / ("smooth_agent_seed%d" % seed) / ("applied_controller_epoch_%02d.json" % epoch))
            check("actual_agent_schedule_from_applied_records", actions[epoch - 1] == applied["align"])
        on_counts[str(seed)] = dict(agent_on=sum(actions), matched_random_on=sum(shuffled["actions"]))
    complete_path = api / "bridge_complete.json"
    final_complete_path = resume / "bridge_complete.json" if resume is not None else complete_path
    if final_complete_path in content:
        complete = read(final_complete_path)
        check("bridge_terminal_status", complete["status"] == "queue_complete" or args.allow_incomplete)
        check("no_unrecorded_attempts", complete.get("usage", {}).get("unrecorded_attempt_upper_bound", 0) == 0)
        usage = complete.get("usage", {})
        names = ("physical_attempts_combined", "charged_prompt_tokens_combined", "charged_completion_tokens_combined") if resume is not None else (
            "physical_attempts_recorded", "charged_prompt_tokens", "charged_completion_tokens")
        check("bridge_summary_recount", usage.get(names[0]) == len(physical)
              and usage.get(names[1]) == prompt_total and usage.get(names[2]) == completion_total)
        check("bridge_summary_cost_recount", math.isclose(usage.get("charged_usd_upper_bound_combined" if resume is not None else "charged_usd_upper_bound", -1),
              prompt_total * .30 / 1e6 + completion_total * 1.20 / 1e6, abs_tol=1e-12))
    else:
        check("bridge_completed_record_present", args.allow_incomplete)
    check("input_snapshot_unchanged_during_audit", all(path.exists() and sha(path.read_bytes()) == manifest[str(path)] for path in paths))
    summary = dict(status="passed" if not failures and not args.allow_incomplete else "partial" if not failures else "failed",
        audit_kind="Agent saved-record recount by bridge implementation author; not independent implementation review or human verification",
        source_commit=commit, frozen_source_hashes=frozen_sources, protocol_sha256=protocol_sha,
        prompt_sha256=sha(prompt.encode()), ledger_sha256=manifest.get(str(api / "api_attempts.jsonl")),
        input_files=len(manifest), input_manifest_sha256=sha(canonical(manifest).encode()),
        audit_logic_amendment_sha256=manifest.get(str(audit_amendment)) if audit_amendment is not None else None,
        checks=len(checks), failed_checks=sorted(set(failures)), completed_jobs=completed_jobs,
        logical_epoch_slots=len(envelopes), physical_attempts=len(physical), unique_logical_api_requests_attempted=len(grouped),
        unique_provider_response_ids=len({row["response_id"] for row in physical if row.get("response_id")}),
        retry_attempts=sum(max(0, len(rows)-1) for rows in grouped.values()),
        logical_slots_without_recorded_physical_attempt=len(requests)-len(grouped),
        api_status_counts=dict(collections.Counter(row.get("status") for row in physical)),
        http_status_counts=dict(collections.Counter(str(row.get("http_status")) for row in physical)),
        finish_reason_counts=dict(collections.Counter(str(row.get("finish_reason")) for row in physical)),
        model_version_declaration=protocol["api"]["model_version"],
        actual_response_model_aliases=sorted({row["model"] for row in physical if "model" in row}),
        usage_verified_attempts=sum(row.get("usage_verified") is True for row in physical),
        charged_prompt_tokens=prompt_total, charged_completion_tokens=completion_total,
        prompt_cache_hit_tokens=cache_hit, prompt_cache_miss_tokens=cache_miss,
        charged_usd_upper_bound=prompt_total*.30/1e6+completion_total*1.20/1e6,
        cost_interpretation="Frozen cache-miss price upper bound, not provider invoice; unknown usage retains reservation",
        physical_attempt_latency_seconds=distribution(latency), applied_wait_seconds=distribution(waits),
        application_categories=dict(categories), per_seed_application_counts={seed:dict(value) for seed,value in per_seed_applications.items()},
        response_format=dict(counts=dict(format_counts), invalid_reason_categories=dict(format_failures),
            reason_characters_stripped=distribution(reason_lengths), parsed_boolean_suggestions=dict(suggestions),
            rejected_boolean_suggestions_counterfactual_only=dict(rejected_suggestions),
            interpretation="Parsing a Boolean suggestion does not accept it; the frozen240-character and other checks determine actual fallback. Format rejection and late/nonapplication can overlap."),
        per_seed_applied_on_counts=on_counts,
        infrastructure=dict(recovered=resume is not None,
            original_bridge_status=original_complete.get("status") if original_complete else None,
            all_bridge_event_status_counts=dict(collections.Counter(row.get("status") for row in combined_events)),
            original_bridge_event_status_counts=dict(collections.Counter(row.get("status") for row in original_events)),
            historical_failed_bridge_receipt_sha256=manifest.get(str(complete_path)),
            recovery_chain=recovery_chain,
            recovery_metadata_retry_events=dict(collections.Counter(row.get("status") for row in recovery_events if row.get("status") in ("metadata_ssh_retry", "metadata_scp_retry", "metadata_ssh_invalid_json"))),
            recovery_metadata_retries_scheduled=sum(row.get("status") in ("metadata_ssh_retry", "metadata_scp_retry", "metadata_ssh_invalid_json") and row.get("attempt", 3) < 3 for row in recovery_events),
            recovery_metadata_deferred_cycles=sum(row.get("status")=="metadata_deferred_pending_preserved" for row in recovery_events),
            recovery_past_slots_skipped_without_new_api=len({row["request_id"] for row in recovery_events if row.get("status") == "past_slot_skipped"}),
            recovery_past_slots_without_any_recorded_physical_api=sum(not grouped.get(key) for key,value in requests.items() if value[0] in past_skipped_slots),
            recovery_past_slots_with_preexisting_physical_api=sum(bool(grouped.get(key)) for key,value in requests.items() if value[0] in past_skipped_slots),
            new_physical_attempts_after_past_slot_observations=new_attempts_after_past_observations,
            logical_publications_unique=len(published),
            idempotent_startup_reconciliations=sum(row.get("idempotent_startup_reconciliation") is True for row in recovery_events),
            historical_original_physical_attempts=len([row for row in original_prefix_rows if row.get("attempted") is True]) if resume is not None else None),
        privacy="No raw prompts, replies, request IDs, node/group IDs, article text or labels in this summary",
        limitations=["Saved TRAIN states and source provenance checked; TRAIN measurements were not independently regenerated.",
          "Response model aliases do not independently certify backend weights/version.",
          "Server-applied flags are authoritative; a valid recorded reply not applied after timeout is not proof of exact late arrival without synchronized timestamp evidence.",
          "Matched outer epoch ON counts do not match node-level alignment dose.",
          "External-agent arm includes any observed fallback actions; fallback epochs are not LLM decisions and recovery does not retroactively convert them.",
          "An overlong explanation can reject a JSON Boolean action; this is a contract-design limitation, not evidence that the suggested action itself was poor. Rejected suggestions are counterfactual descriptions only.",
          "API physical-attempt latency sum is not end-to-end training wall time or GPU cost.",
          "Agent rationale prose is not semantically truth-validated and controller performance is outside this usage audit."],
        elapsed_seconds=time.perf_counter()-started, new_api_calls=0, new_model_calls=0, new_training_steps=0, ssh_calls=0, gpu_hours=0)
    write(output / "checks_private.json", checks)
    write(output / "api_records_audit_public_safe.json", summary)
    print(canonical(dict(status=summary["status"], checks=len(checks), failed_checks=sorted(set(failures)), logical_slots=len(envelopes), physical_attempts=len(physical))))


if __name__ == "__main__":
    main()
