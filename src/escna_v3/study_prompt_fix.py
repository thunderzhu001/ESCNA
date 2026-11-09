"""Disclosure-contract fix for m2; NOT active in recorded m1 runs.

Explicitly distinguish already-public evidence from records an owner may newly
publish. These helpers must be versioned into a new protocol before paid use.
They do not relax privacy or change any recorded result.
"""
from .providers import schema_for


def explicit_evidence_view(task, role, transcript=()):
    s = task.scenario
    if role not in s.participants:
        raise ValueError("unknown participant")
    published = s.initial_public_fact_ids() | {
        record["record_id"] for event in transcript for record in event.get("published_records", [])}
    private = [f for f in s.facts if f.owner == role and f.fact_id not in published]
    def record(f):
        return {"record_id": f.fact_id, "custodian": f.owner, "text": f.text}
    return {"task": task.public_description, "your_role": role,
            "participants": list(s.participants), "options": list(s.options),
            "public_base_utilities": s.base_utilities, "public_BATNA": s.batna,
            "utility_rule": "Start with ALL FOUR participants' base utilities. Each distinct record adds its stated beneficiary/option adjustment ONCE. Custodians and beneficiaries are different concepts.",
            "initially_public_records": [record(f) for f in s.facts if f.is_public],
            "your_unpublished_private_records": [record(f) for f in private],
            "allowed_new_disclosure_ids": [f.fact_id for f in private],
            "disclosure_rule": "disclose_fact_ids may contain ONLY allowed_new_disclosure_ids. Already-public records may be discussed in message, but must not be listed as a new disclosure. If allowed_new_disclosure_ids is empty, return [].",
            "public_dialogue": list(transcript)}


def strict_participant_contract(options, allowed_new_ids):
    return schema_for("participant", options, allowed_new_ids, strict_disclosure=True)
