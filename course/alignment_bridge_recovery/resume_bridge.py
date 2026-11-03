"""Private infrastructure recovery; frozen scientific sources stay unchanged.

Reuse the ORIGINAL ledger/cache. Never repeat a recorded logical API request,
never spend a new API request on an already-applied epoch, and never rewrite
training decisions. Run only after root confirms the original bridge exited.
"""
import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import shlex
import subprocess
import sys
import time
import uuid


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "protocol", "api-dir", "key-file", "output"):
        parser.add_argument("--" + name, required=True, type=pathlib.Path)
    parser.add_argument("--confirmed-original-exited", action="store_true", required=True)
    args = parser.parse_args()
    repo, protocol_path, api, key_file, output = (x.resolve() for x in
        (args.repo, args.protocol, args.api_dir, args.key_file, args.output))
    sys.path.insert(0, str(repo))
    from course.alignment_agent import api_bridge as bridge
    for path in (protocol_path, api, key_file, output):
        bridge.private_path(path)
    raw_protocol = protocol_path.read_bytes()
    protocol = bridge.strict_json(raw_protocol.decode("utf-8-sig"), 1048576)
    bridge.validate_protocol(protocol)
    protocol_sha = digest(raw_protocol)
    stage = protocol["stage_root"]
    if not re.fullmatch(r"/home/tjy/Project/MonTi-course/runs/[A-Za-z0-9_/-]+", stage) or ".." in stage:
        raise RuntimeError("Invalid frozen metadata stage")
    old_complete_path = api / "bridge_complete.json"
    old_complete = bridge.strict_json(old_complete_path.read_text(encoding="utf-8"))
    old_registered = bridge.strict_json((api / "registered.json").read_text(encoding="utf-8"))
    if old_complete["status"] != "bridge_failed" or old_complete["protocol_sha256"] != protocol_sha or old_registered["protocol_sha256"] != protocol_sha:
        raise RuntimeError("Recovery requires the matching preserved failed bridge receipt")
    if not key_file.is_file():
        raise RuntimeError("Encrypted local credential file unavailable")
    output.mkdir(parents=True, exist_ok=False)
    nonce = uuid.uuid4().hex
    amendment = dict(schema="alignment_bridge_infrastructure_recovery_v1", protocol_sha256=protocol_sha,
        scientific_source_commit=protocol["source_commit"], recovery_source_sha256=digest(pathlib.Path(__file__).read_bytes()),
        old_completion_sha256=digest(old_complete_path.read_bytes()), old_ledger_sha256=digest((api / "api_attempts.jsonl").read_bytes()),
        original_process_exit_confirmed_by_root=True, recovery_nonce=nonce,
        registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        change="Retry metadata transfer/publication only after the confirmed original bridge exit; same frozen prompt, model, protocol, state contract, shared ledger and API caps.",
        invariants=["No repeat of any logical API ID already recorded in the original ledger.",
          "Already-applied server slots receive no new API call; preserved cached answers may be published for record only.",
          "No server applied decision or training record is changed retroactively.",
          "Original receipts/logs remain untouched; recovery events and completion are separate.",
          "Metadata retries only: at most three attempts; original API retry policy unchanged.",
          "Original three-hour daemon deadline and global21logical/120physical/.50USD/token caps retained."])
    bridge.write_json(output / "infrastructure_amendment.json", amendment)
    amendment_sha = digest((output / "infrastructure_amendment.json").read_bytes())
    bridge.write_json(output / "registered.json", dict(protocol_sha256=protocol_sha, amendment_sha256=amendment_sha,
        recovery_nonce=nonce, original_ledger="same original api_attempt1/api_attempts.jsonl",
        caps=bridge.CAPS, prompt_sha256=digest(bridge.PROMPT.encode()), started_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    events = (output / "bridge_events.jsonl").open("x", encoding="utf-8")
    def event(**value):
        value["utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        events.write(bridge.canonical(value) + "\n"); events.flush()
    def ssh(script, *arguments):
        command = "python3 -c " + shlex.quote(script) + " " + " ".join(shlex.quote(str(x)) for x in arguments)
        for attempt in range(1, 4):
            try:
                result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "QY", command],
                    capture_output=True, text=True, timeout=8)
                if result.returncode == 0:
                    return result.stdout
            except (OSError, subprocess.SubprocessError):
                pass
            event(status="metadata_ssh_retry", attempt=attempt)
            if attempt < 3: time.sleep(1)
        raise RuntimeError("Bounded metadata SSH retries exhausted")
    def scp(source, destination):
        for attempt in range(1, 4):
            try:
                result = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", str(source), str(destination)], capture_output=True, timeout=10)
                if result.returncode == 0: return
            except (OSError, subprocess.SubprocessError):
                pass
            event(status="metadata_scp_retry", attempt=attempt)
            if attempt < 3: time.sleep(1)
        raise RuntimeError("Bounded metadata SCP retries exhausted")
    def records(path):
        return [bridge.strict_json(line, 1048576) for line in path.read_text(encoding="utf-8-sig").splitlines()] if path.exists() else []
    ledger = api / "api_attempts.jsonl"
    slots, rejected, processed = {}, set(), set()
    for row in records(api / "bridge_events.jsonl"):
        if row.get("status") == "rejected_before_api": rejected.add(row["request_file"])
    for path in (api / "requests").glob("*.json"):
        if path.name.startswith("api_") or path.name in rejected: continue
        envelope = bridge.validate_request(bridge.strict_json(path.read_text(encoding="utf-8")), protocol_sha)
        prepared = bridge.prepare_request(envelope, protocol, protocol_sha)
        slot = (envelope["seed"], envelope["epoch"])
        if slot in slots and slots[slot] != prepared["cache_key"]:
            raise RuntimeError("An old epoch has conflicting states")
        slots[slot] = prepared["cache_key"]
    started_utc = dt.datetime.fromisoformat(old_registered["started_utc"].replace("Z", "+00:00"))
    remaining = max(0, 3*3600 - (dt.datetime.now(dt.timezone.utc) - started_utc).total_seconds())
    deadline = time.monotonic() + remaining
    disabled, terminal = None, "original_deadline"
    def applied(envelope):
        relative = "matrix_attempt1/smooth_agent_seed%d/applied_controller_epoch_%02d.json" % (envelope["seed"], envelope["epoch"])
        raw = ssh("import pathlib,sys; p=pathlib.Path(sys.argv[1])/sys.argv[2]; print(p.read_text() if p.exists() else 'null')", stage, relative)
        value = bridge.strict_json(raw)
        if value is not None and (value.get("request_id") != envelope["request_id"] or value.get("state_sha256") != bridge.state_sha256(envelope["state"]) or value.get("protocol_sha256") != protocol_sha):
            raise RuntimeError("Applied server slot identity mismatch")
        return value
    def publish(envelope, decision):
        path = api / "responses" / (envelope["request_id"] + ".json")
        if path.exists():
            if bridge.strict_json(path.read_text(encoding="utf-8")) != decision:
                raise RuntimeError("Preserved local response differs; do not overwrite")
        else: bridge.write_json(path, decision)
        temporary = stage + "/responses/." + path.name + ".resume_" + nonce + ".tmp"
        expected = digest(path.read_bytes())
        # SCP retries write the same temporary metadata bytes. Publication is
        # idempotent if a prior SSH success was lost before local acknowledgement.
        scp(path, "QY:" + temporary)
        result = ssh("import hashlib,os,pathlib,sys; p=pathlib.Path(sys.argv[1]); q=pathlib.Path(sys.argv[2]); wanted=sys.argv[3]; exists=q.exists(); good=exists and hashlib.sha256(q.read_bytes()).hexdigest()==wanted;\nif exists and not good: raise RuntimeError('ExistingResponseHashMismatch')\nif not exists:\n if hashlib.sha256(p.read_bytes()).hexdigest()!=wanted: raise RuntimeError('TemporaryHashMismatch')\n try: os.link(str(p),str(q))\n except FileExistsError:\n  if hashlib.sha256(q.read_bytes()).hexdigest()!=wanted: raise RuntimeError('ConcurrentResponseMismatch')\nif p.exists(): p.unlink()\nprint('published_or_identical')", temporary, stage + "/responses/" + path.name, expected)
        if result.strip() != "published_or_identical": raise RuntimeError("Unexpected metadata publication receipt")
    try:
        # This resume claim must match the preserved original protocol/nonce;
        # a second recovery daemon may not acquire a different nonce.
        claim_raw = ssh("import json,pathlib,re,sys; p=pathlib.Path(sys.argv[1]); old=json.loads((p/'bridge_claim.json').read_text());\nif old.get('protocol_sha256')!=sys.argv[2] or not re.fullmatch('[0-9a-f]{32}',str(old.get('bridge_nonce',''))): raise RuntimeError('OriginalClaimMismatch')\nr=p/'bridge_resume_claim.json'; payload={'protocol_sha256':sys.argv[2],'original_bridge_nonce':old['bridge_nonce'],'recovery_nonce':sys.argv[3],'amendment_sha256':sys.argv[4]};\nif r.exists():\n if json.loads(r.read_text())!=payload: raise RuntimeError('AnotherRecoveryClaimExists')\nelse:\n with r.open('x') as f: json.dump(payload,f)\nprint(json.dumps(payload))", stage, protocol_sha, nonce, amendment_sha)
        bridge.write_json(output / "remote_claim_receipt.json", bridge.strict_json(claim_raw))
        # Reconcile existing local responses too: the failed SSH may have linked
        # the final file before losing its acknowledgement. This emits a fresh
        # hash-checked metadata receipt without issuing an API call.
        for response_path in sorted((api / "responses").glob("*.json")):
            if response_path.name in rejected: continue
            envelope_path = api / "requests" / response_path.name
            envelope = bridge.validate_request(bridge.strict_json(envelope_path.read_text(encoding="utf-8")), protocol_sha)
            prepared = bridge.prepare_request(envelope, protocol, protocol_sha)
            logical_id = prepared["request"]["request_id"]
            old_rows = [row for row in records(ledger) if row.get("request_id") == logical_id]
            cached_path = api / "cache" / (prepared["cache_key"] + ".json")
            cached = bridge.strict_json(cached_path.read_text(encoding="utf-8"), 1048576) if cached_path.exists() else None
            if cached is not None and cached.get("request_id") != logical_id: raise RuntimeError("Preserved cache identity mismatch")
            raw = cached.get("raw") if cached is not None else None
            successful = [row for row in old_rows if row.get("status") == "completed" and row.get("finish_reason") == "stop" and row.get("model") == "deepseek-flash"]
            if successful: raw = successful[-1]["response"]["choices"][0]["message"]["content"]
            decision = bridge.resolve_decision(envelope, raw, protocol_sha)
            if bridge.strict_json(response_path.read_text(encoding="utf-8")) != decision:
                raise RuntimeError("Preserved response cannot be reproduced from original ledger/cache")
            prior = applied(envelope)
            if prior is not None:
                event(status="past_slot_skipped", request_id=envelope["request_id"], seed=envelope["seed"], next_epoch=envelope["epoch"], state_sha256=prepared["state_sha256"], recorded_fallback=prior["fallback"])
            publish(envelope, decision)
            event(status="response_published", request_id=envelope["request_id"], seed=envelope["seed"], next_epoch=envelope["epoch"],
                fallback=decision["fallback"], align=decision["align"], used_cache=True, record_only_past_slot=prior is not None,
                idempotent_startup_reconciliation=True)
            processed.add(response_path.name)
        while time.monotonic() < deadline:
            listing = bridge.strict_json(ssh("import pathlib,json,sys; p=pathlib.Path(sys.argv[1]); stop='failed' if (p/'queue_failed.json').exists() else 'complete' if (p/'queue_complete.json').exists() else None; print(json.dumps({'stop':stop,'requests':sorted(x.name for x in (p/'requests').glob('*.json') if not (p/'responses'/x.name).exists())}))", stage), 65536)
            if listing["stop"]:
                terminal = "queue_" + listing["stop"]; break
            for filename in listing["requests"]:
                if time.monotonic() >= deadline: break
                if filename in processed or filename in rejected: continue
                if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,99}\.json", filename): continue
                path = api / "requests" / filename
                if not path.exists():
                    part = output / (filename + ".part")
                    scp("QY:" + stage + "/requests/" + filename, part)
                    part.rename(path)
                try:
                    envelope = bridge.validate_request(bridge.strict_json(path.read_text(encoding="utf-8")), protocol_sha)
                    if filename != envelope["request_id"] + ".json": raise ValueError("Request filename mismatch")
                    prepared = bridge.prepare_request(envelope, protocol, protocol_sha)
                    slot = (envelope["seed"], envelope["epoch"])
                    if slot in slots and slots[slot] != prepared["cache_key"]: raise ValueError("Changed epoch state")
                    slots[slot] = prepared["cache_key"]
                    if len(slots) > 21: raise ValueError("Global logical limit exceeded")
                except (ValueError, bridge.ContractError) as error:
                    rejected.add(filename); event(status="rejected_before_api", request_file=filename, error_type=type(error).__name__); continue
                rows = records(ledger)
                if any(row.get("protocol_sha256") != protocol_sha for row in rows): raise RuntimeError("Shared ledger protocol mismatch")
                logical_id = prepared["request"]["request_id"]
                old_rows = [row for row in rows if row.get("request_id") == logical_id]
                cached_path = api / "cache" / (prepared["cache_key"] + ".json")
                cached = bridge.strict_json(cached_path.read_text(encoding="utf-8"), 1048576) if cached_path.exists() else None
                if cached is not None and cached.get("request_id") != logical_id: raise RuntimeError("Cache identity mismatch")
                raw = cached.get("raw") if cached is not None else None
                successful = [row for row in old_rows if row.get("status") == "completed" and row.get("finish_reason") == "stop" and row.get("model") == "deepseek-flash"]
                if successful:
                    raw = successful[-1]["response"]["choices"][0]["message"]["content"]
                already = applied(envelope)
                if already is not None:
                    event(status="past_slot_skipped", request_id=envelope["request_id"], seed=slot[0], next_epoch=slot[1], state_sha256=prepared["state_sha256"], recorded_fallback=already["fallback"])
                    if old_rows or cached is not None:
                        decision = bridge.resolve_decision(envelope, raw, protocol_sha)
                        publish(envelope, decision)
                        event(status="response_published", request_id=envelope["request_id"], seed=slot[0], next_epoch=slot[1], fallback=decision["fallback"], align=decision["align"], used_cache=True, record_only_past_slot=True)
                    processed.add(filename); continue
                if any(row.get("http_status") in (401, 403) or row.get("reservation_exceeded") for row in rows): disabled = "prior_fatal"
                if not old_rows and cached is None and disabled is None and deadline - time.monotonic() > 180:
                    api_request_path = api / "requests" / ("api_" + filename)
                    if api_request_path.exists():
                        if bridge.strict_json(api_request_path.read_text(encoding="utf-8")) != prepared["request"]: raise RuntimeError("Prepared request mismatch")
                    else: bridge.write_json(api_request_path, prepared["request"])
                    # Recheck immediately before spending: an already-applied
                    # fallback never becomes a fresh API training example.
                    if applied(envelope) is not None:
                        event(status="past_slot_skipped", request_id=envelope["request_id"], seed=slot[0], next_epoch=slot[1], state_sha256=prepared["state_sha256"])
                        processed.add(filename); continue
                    command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(bridge.TRANSPORT), "-RequestFile", str(api_request_path), "-OutputFile", str(ledger), "-KeyFile", str(key_file), "-ProtocolSha256", protocol_sha, "-MaxRequests", "120", "-MaxPromptTokens", "200000", "-MaxCompletionTokens", "64000", "-MaxUsd", "0.50"]
                    try:
                        process = subprocess.run(command, capture_output=True, timeout=145, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        if process.returncode: disabled = "transport_initialization_failed"
                    except subprocess.TimeoutExpired:
                        disabled = "transport_timeout_unrecorded_attempt_upper_bound_two"
                        event(status=disabled, physical_attempt_upper_bound_unconfirmed=2)
                    current = [row for row in records(ledger) if row.get("request_id") == logical_id]
                    successful = [row for row in current if row.get("status") == "completed" and row.get("finish_reason") == "stop" and row.get("model") == "deepseek-flash"]
                    raw = successful[-1]["response"]["choices"][0]["message"]["content"] if successful else None
                    if not cached_path.exists(): bridge.write_json(cached_path, dict(raw=raw, request_id=logical_id))
                decision = bridge.resolve_decision(envelope, raw, protocol_sha)
                publish(envelope, decision)
                event(status="response_published", request_id=envelope["request_id"], seed=slot[0], next_epoch=slot[1], fallback=decision["fallback"], align=decision["align"], used_cache=bool(old_rows or cached is not None), server_slot_already_applied_at_publication=applied(envelope) is not None, api_disabled_reason=disabled)
                processed.add(filename)
            time.sleep(2)
    except Exception as error:
        terminal = "resume_failed"
        event(status=terminal, error_type=type(error).__name__)
        raise
    finally:
        events.close()
        try:
            all_rows = records(ledger)
            attempted = [row for row in all_rows if row.get("attempted") is True]
            usage = dict(physical_attempts_combined=len(attempted), charged_prompt_tokens_combined=sum(row.get("charged_prompt_tokens", 0) for row in attempted),
                charged_completion_tokens_combined=sum(row.get("charged_completion_tokens", 0) for row in attempted),
                charged_usd_upper_bound_combined=sum(row.get("charged_usd_upper_bound", 0) for row in attempted))
        except (ValueError, OSError):
            attempted = []
            usage = dict(status="ledger_unreadable_preserved_accounting_uncertain")
        usage["unrecorded_attempt_upper_bound"] = 2 if disabled == "transport_timeout_unrecorded_attempt_upper_bound_two" else 0
        bridge.write_json(output / "bridge_complete.json", dict(status=terminal, protocol_sha256=protocol_sha,
            amendment_sha256=amendment_sha, processed_requests=len(processed), unique_epoch_slots=len(slots),
            api_disabled_reason=disabled, completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            usage=usage,
            new_model_calls=0, new_training_steps=0, gpu_hours=0))
    print(bridge.canonical(dict(status=terminal, physical_attempts_combined=usage.get("physical_attempts_combined"))))


if __name__ == "__main__":
    main()
