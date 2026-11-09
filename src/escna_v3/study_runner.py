"""Run independent mechanism studies with immutable request IDs and sealed votes."""
from dataclasses import asdict
import json
from pathlib import Path
from .mechanisms import evaluate_sealed_ballots, normalized_entropy
from .providers import request_payload, schema_for, utc_now
from .study_controller import decide_control
from .study_tasks import strict_winner
from .study_prompt_fix import explicit_evidence_view

PROTOCOL = "v3-mechanism-m2"
METHODS = {"A": ("cluster_shapley", "greedy", "structured_exchange"),
           "B": ("entropy", "value_priority", "fixed_round_robin"),
           "C": ("constraint", "neutral")}
MAX_CALLS = {"A": 20, "B": 6, "C": 9}
MAX_INPUT_BYTES = 24000


def participant_view(task, role, transcript):
    s = task.scenario
    if role not in s.participants:
        raise ValueError("unknown participant")
    return {"task": task.public_description, "your_role": role,
            "options": s.options, "base_utilities": s.base_utilities, "BATNA": s.batna,
            "utility_rule": "Add each distinct record's adjustment once only. A custodian need not be the beneficiary. Missing records are unknown, not necessarily zero.",
            "available_records": [{"record_id": f.fact_id, "owner": f.owner, "text": f.text}
                for f in s.facts if f.is_public or f.owner == role],
            "public_dialogue": transcript}


def host_view(safe_action):
    # Deliberate whitelist: neither task snapshot nor controller audit is sent.
    if set(safe_action) != {"action", "target_owner", "public_advice"}:
        raise ValueError("unsafe controller payload")
    if safe_action["action"] == "invite_owner":
        return {"action": "invite_owner", "target_owner": safe_action["target_owner"],
                "instruction": "Invite this owner to publish any or all of their records and briefly explain their implications. Do not assert, guess or name any record or decision option."}
    return {"action": safe_action["action"], "public_advice": safe_action["public_advice"]}


def _normal(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def run_study(task, spec, client, result_path):
    if spec["protocol"] not in {"v3-mechanism-m1", PROTOCOL} or task.task_id != spec["task_id"] or spec["method"] not in METHODS[task.module]:
        raise ValueError("study identity mismatch")
    if task.metadata["phase"] != spec["phase"]:
        raise ValueError("pilot/formal isolation failure")
    if spec["provider"] != client.profile["provider"]:
        raise ValueError("declared provider differs from actual client")
    if "model" in spec and spec["model"] != client.profile["model"]:
        raise ValueError("declared model differs from actual client")
    if spec["phase"] == "formal" and "model" not in spec:
        raise ValueError("formal run requires a frozen model ID")
    identity = _normal({"spec": spec, "task_snapshot": asdict(task), "profile": client.profile})
    path = Path(result_path)
    if path.exists():
        old = json.loads(path.read_text())
        if any(old.get(k) != v for k, v in identity.items()):
            raise ValueError("saved run identity changed")
        return old
    s = task.scenario
    transcript = list(task.prefix_events)
    disclosed = s.initial_public_fact_ids()
    counts = {p: sum(e["speaker"] == p for e in transcript) for p in s.participants}
    used, audit, requests, checkpoints, failures = [], [], [], [], []
    private_explanations = []
    published_at = {fid: 0 for fid in disclosed}
    output = {**identity, "status": "completed", "started_at": utc_now()}
    revised = spec["protocol"] == PROTOCOL
    record_channel_only = revised and spec["phase"] == "formal" and task.module in {"A", "B"}
    if revised and (client.profile.get("completion_budget_basis") != "visible_output" or
                    not client.profile.get("strict_disclosure")):
        raise ValueError("m2 requires explicitly frozen visible-budget and disclosure policies")
    def actor_view(role):
        return explicit_evidence_view(task, role, transcript) if revised else participant_view(task, role, transcript)

    def request(step, kind, system, view, role=None):
        if len(requests) >= MAX_CALLS[task.module]:
            raise ValueError("logical request budget exceeded")
        owned = tuple(f.fact_id for f in s.facts if f.owner == role and not f.is_public
                      and (not revised or f.fact_id not in disclosed))
        user = json.dumps(view, ensure_ascii=False)
        payload = request_payload(client.profile, system, user, schema_for(kind, s.options, owned,
                                  strict_disclosure=client.profile.get("strict_disclosure", False)))
        if len(json.dumps(payload, ensure_ascii=False).encode()) > MAX_INPUT_BYTES:
            failures.append({"step": step, "kind": "input_budget", "request_sent": False})
            return None
        row = client.generate(f"{spec['run_id']}_{step}", system, user, kind=kind,
            options=s.options, fact_ids=owned,
            metadata={"protocol": spec["protocol"], "phase": spec["phase"], "run_id": spec["run_id"],
                      "task_id": task.task_id, "module": task.module, "method": spec["method"],
                      "role": role, "measurement_only": kind == "ballot"})
        requests.append(row["request_id"])
        if row["status"] != "completed":
            failures.append({"step": step, "kind": row.get("failure_kind"), "request_id": row["request_id"]})
            return None
        return row["parsed"]

    goal = ("You represent your own individual preferences. Seek a unanimously acceptable agreement, "
            "negotiate for your utility, and consider compromises: no agreement pays your BATNA. "
            "Do not automatically maximise total utility on behalf of everyone. " if task.module == "C" else
            "Seek an option with high total committee utility that you judge individually acceptable. ")
    system = ("This is a synthetic decision simulation, not a human study. " + goal +
              "Use only available records, public statements and the explicit arithmetic. Do not invent evidence. "
              "Keep any message within 60 words. Return only the required JSON.")

    def invite(round_number):
        decision = decide_control(s, sorted(disclosed), counts, spec["method"], used,
                                  task.metadata.get("complete_brief_authorized", False), spec["seed"] + round_number)
        audit.append({"round": round_number, **decision})
        generated = request(f"host_{round_number}", "facilitator",
            "You are a language-only facilitator. Render the supplied action in at most 45 words. "
            "Do not add factual claims, infer private evidence, change targets or override participant choices.",
            host_view(decision["safe_action"]))
        if generated is None:
            return None
        transcript.append({"turn": len(transcript)+1, "speaker": "facilitator", "message": generated["message"],
                           "recommendation": "ABSTAIN", "published_records": []})
        return decision["safe_action"]

    def discuss(role, step):
        instruction = (" Publish any or all of your as-yet unpublished owned records using disclose_fact_ids; "
                       "their exact text will be attached. You may publish all records in one message. "
                       "Discuss the available evidence and recommend an option or ABSTAIN.")
        if record_channel_only:
            instruction += (" This is a controlled record-exchange experiment: your message explanation is stored "
                            "privately for audit, not broadcast. Only the exact records selected in disclose_fact_ids "
                            "and your recommendation are broadcast. Select every owned record you wish others to see.")
        parsed = request(step, "participant", system + instruction,
                         actor_view(role), role)
        if parsed is None:
            return False
        new = set(parsed["disclose_fact_ids"]) - disclosed
        counts[role] += 1
        for fid in new:
            published_at[fid] = sum(counts.values())
        disclosed.update(new)
        message = parsed["message"]
        if record_channel_only:
            private_explanations.append({"step": step, "speaker": role, "message": message})
            message = "This participant submitted the attached registered records and recommendation."
        transcript.append({"turn": len(transcript)+1, "speaker": role, "message": message,
            "recommendation": parsed["recommendation"],
            "published_records": [{"record_id": f.fact_id, "owner": f.owner, "text": f.text}
                                  for f in s.facts if f.fact_id in new]})
        return True

    def checkpoint(budget):
        ballots = {}
        for role in s.participants:
            instruction = (" Submit a sealed ballot unseen by other participants. List every option you judge "
                           "acceptable at or above your BATNA and recommend one. If none is acceptable, ABSTAIN "
                           "with an empty list. Your vote does not alter any subsequent discussion.")
            row = request(f"b{budget}_vote_{role}", "ballot", system + instruction,
                          actor_view(role), role)
            if row is not None:
                ballots[role] = row
        evidence_sets = [set(k) for k in task.sufficient_evidence_sets]
        private_ids = {f.fact_id for f in s.facts if not f.is_public}
        metrics = {"structured_fact_publication_rate": len(disclosed & private_ids)/len(private_ids) if private_ids else None,
                   "sufficient_set_coverage": max((len(disclosed & k)/len(k) for k in evidence_sets), default=None),
                   "contains_minimal_sufficient_set": any(k <= disclosed for k in evidence_sets) if evidence_sets else None,
                   "current_public_evidence_correct": strict_winner(s, sorted(disclosed)) == s.optimal_decision,
                   "published_record_bytes": sum(len(f.text.encode()) for f in s.facts if f.fact_id in disclosed & private_ids),
                   "normalized_entropy": normalized_entropy(list(counts.values()))}
        complete = len(ballots) == len(s.participants)
        evaluation = asdict(evaluate_sealed_ballots(s, ballots)) if complete else None
        metrics.update(accuracy_itt=int(bool(evaluation and evaluation["full_truth_accuracy"])),
                       feasible_agreement_itt=int(bool(evaluation and evaluation["unanimity_accept"] and evaluation["batna_satisfied"] and evaluation["nondominated"])))
        checkpoints.append({"owner_budget": budget, "status": "completed" if complete else "failed",
                            "ballots": ballots, "evaluation": evaluation, "metrics": metrics,
                            "public_fact_ids": sorted(disclosed), "request_count_so_far": len(requests)})

    if task.module in {"A", "B"}:
        rounds = 4 if task.module == "A" else 1
        for index in range(1, rounds+1):
            action = invite(index)
            if action is None:
                break
            role = action["target_owner"]
            used.append(role)
            if not discuss(role, f"owner_{index}"):
                break
            if task.module == "B" or index >= 2:
                checkpoint(index)
                # Measurement failures do not generate replacement votes or
                # contaminate the continuing operational dialogue.
    else:
        if invite(1) is not None:
            for role in s.participants:
                if not discuss(role, f"discussion_{role}"):
                    break
            if len([e for e in transcript if e["speaker"] in s.participants]) == 4:
                checkpoint(4)
    expected = (2, 3, 4) if task.module == "A" else (1,) if task.module == "B" else (4,)
    for budget in expected:
        if not any(c["owner_budget"] == budget for c in checkpoints):
            checkpoints.append({"owner_budget": budget, "status": "not_reached", "ballots": {},
                                "evaluation": None, "metrics": {"accuracy_itt": 0, "feasible_agreement_itt": 0}})
    if failures:
        output["status"] = "failed"
    output.update(finished_at=utc_now(), transcript=transcript, control_audit=audit,
                  request_ids=requests, failures=failures, checkpoints=checkpoints,
                  publication_turns=published_at, used_owners=used, speaking_counts=counts,
                  private_explanations=private_explanations, record_channel_only=record_channel_only)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2); handle.write("\n")
    return output
