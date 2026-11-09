"""Finite synthetic mechanism tasks; text and payoffs share one source.

These are purpose-built computational probes, not independent human scenarios
or HiddenBench adaptations. Numeric/domain variants share dependency templates.
"""
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from escna.models import PrivateFact, Scenario

DOMAINS = {
    "university_budget": ("laboratory retrofit", "access grants", "community tutoring"),
    "supplier_selection": ("Northstar contract", "Delta contract", "Harbor contract"),
    "emergency_response": ("mobile clinics", "water tankers", "bridge sensors"),
}
ROLES = ("P1", "P2", "P3", "P4")
# Rows are record custodians, columns are affected option positions. Custody is
# deliberately not the identity of the beneficiary of a record.
PATTERNS = (
    ((2, 10, 4), (-2, 10, -2), (1, 10, 3), (-1, -3, -1)),
    ((1, 14, 4), (-1, 14, -2), (2, 14, 3), (-2, -12, -1)),
    ((-12, 2, 4), (-12, 1, -4), (4, 3, 6), (2, -1, -2)),
)


@dataclass(frozen=True)
class StudyTask:
    task_id: str
    module: str
    family_id: str
    scenario: Scenario
    public_description: str
    metadata: dict
    sufficient_evidence_sets: tuple = ()
    prefix_events: tuple = ()

    @classmethod
    def from_dict(cls, data):
        return cls(**{**data, "scenario": Scenario.from_dict(data["scenario"]),
                      "sufficient_evidence_sets": tuple(tuple(x) for x in data.get("sufficient_evidence_sets", [])),
                      "prefix_events": tuple(data.get("prefix_events", []))})


def fact_text(owner, beneficiary, option, delta, topic):
    return (f"{owner} holds the {topic} record: add {delta:+g} utility points "
            f"to {beneficiary} for {option}. All other utilities are unchanged.")


def strict_winner(s, ids):
    values = {o: s.welfare(o, ids) for o in s.options}
    winners = [o for o, v in values.items() if v == max(values.values())]
    return winners[0] if len(winners) == 1 else None


def evidence_certificate(s):
    """Enumerate all 4096 subsets once, independent of controller outcomes."""
    ids = tuple(f.fact_id for f in s.facts)
    target = strict_winner(s, ids)
    minimal = []
    winning_count = 0
    for size in range(len(ids) + 1):
        for subset in combinations(ids, size):
            selected = frozenset(subset)
            if strict_winner(s, subset) == target:
                winning_count += 1
                if not any(k <= selected for k in minimal):
                    minimal.append(selected)
    # With negative evidence, completing a minimal subset need not preserve its
    # ranking after more disclosure. Score current winner separately.
    return tuple(tuple(sorted(k)) for k in minimal), {
        "enumerated_subsets": 2 ** len(ids), "winning_subsets": winning_count,
        "full_information_winner": target,
        "local_winners": {p: strict_winner(s, [f.fact_id for f in s.facts if f.owner == p]) for p in ROLES},
        "nonmonotone_evidence_possible": True,
    }


def acquisition_task(domain="university_budget", pattern=0, variant=0, phase="pilot"):
    options = DOMAINS[domain]
    # Rotate labels and custodians together with the *generated* factual text.
    mapped = options[variant % 3:] + options[:variant % 3]
    decoy, target, third = mapped
    offset = variant % 4
    target_base = ({"university_budget": 25., "supplier_selection": 24.5,
                    "emergency_response": 25.5}[domain] if phase == "formal" else 25.)
    base = {p: {decoy: 30., target: target_base, third: 24.} for p in ROLES}
    facts = []
    for i, row in enumerate(PATTERNS[pattern]):
        owner = ROLES[(i + offset) % 4]
        beneficiary = ROLES[(i + 1 + offset) % 4]
        for j, delta in enumerate(row):
            option = mapped[j]
            topic = ("capacity adjustment", "service benefit", "transition cost")[j]
            facts.append(PrivateFact(f"R{i*3+j+1:02d}", owner,
                fact_text(owner, beneficiary, option, delta, topic), topic,
                {beneficiary: {option: delta}}, False))
    task_id = f"{phase}_A_{domain}_p{pattern}_v{variant}"
    s = Scenario(task_id, domain, variant + 1, "hidden", ROLES, options,
                 base, {p: 8. for p in ROLES}, tuple(facts), target)
    evidence, certificate = evidence_certificate(s)
    if certificate["full_information_winner"] != target:
        raise ValueError("task has no unique intended full-information winner")
    if any(o == target for o in certificate["local_winners"].values()):
        raise ValueError("one owner's private information already solves task")
    return StudyTask(task_id, "A", f"{domain}_pattern{pattern}", s,
        f"Synthetic {domain.replace('_', ' ')} committee. Four custodians hold twelve additive records. "
        "Acquire evidence and select an acceptable option with high total committee utility.",
        {"phase": phase, "dependency_template": pattern, "public_decoy": decoy,
         "complete_brief_authorized": True, "owner_record_count": 3,
         "checkpoints": [2, 3, 4], "certificate": certificate,
         "public_welfare_gap": 4*(30-target_base),
         "semantic_template_count_not_sample_size": True}, evidence)


def entropy_task(domain="university_budget", variant=0, control="key", phase="pilot"):
    # Same utilities and public prefix in each key/irrelevant pair. Swap custody
    # of the two unavailable records, never their numeric beneficiaries.
    source = acquisition_task(domain, 2, variant, phase)
    s = source.scenario
    key_id, irrelevant_id = "R04", "R12"
    facts = []
    low, other = "P4", "P1"
    for f in s.facts:
        owner = f.owner
        public = f.fact_id not in {key_id, irrelevant_id}
        if not public:
            owner = low if (f.fact_id == key_id) == (control == "key") else other
        beneficiary, effects = next(iter(f.utility_effects.items()))
        option, delta = next(iter(effects.items()))
        facts.append(replace(f, owner=owner, is_public=public,
            text=fact_text(owner, beneficiary, option, delta, f.topic)))
    task_id = f"{phase}_B_{domain}_v{variant}_{control}"
    s = replace(s, scenario_id=task_id, facts=tuple(facts))
    public = [f.fact_id for f in facts if f.is_public]
    if strict_winner(s, public) == s.optimal_decision:
        raise ValueError("B missing record is not decision pivotal")
    if strict_winner(s, public + [key_id]) != s.optimal_decision:
        raise ValueError("B key record does not uniquely recover the target")
    if strict_winner(s, public + [irrelevant_id]) == s.optimal_decision:
        raise ValueError("B irrelevant control unexpectedly solves the task")
    if s.fact_map()[key_id].owner == s.fact_map()[irrelevant_id].owner:
        raise ValueError("B controls must have different custodians")
    schedule = ("P1", "P1", "P2", "P1", "P3", "P1", "P2", "P1", "P1", "P1")
    prefix = tuple({"turn": i+1, "speaker": p,
        "message": "The committee is reviewing the public dossier; the unfiled records remain unverified.",
        "recommendation": "ABSTAIN", "published_records": [], "source": "synthetic_fixed_prefix"}
        for i, p in enumerate(schedule))
    return StudyTask(task_id, "B", f"{domain}_entropy_v{variant}", s,
        "Synthetic continuation experiment. The attached ten-message prefix is a fixed fixture, "
        "not generated behavior. Two records remain private. One owner will be invited before sealed voting.",
        {"phase": phase, "control": control, "low_exposure_owner": low,
         "key_record_id": key_id, "irrelevant_record_id": irrelevant_id,
         "complete_brief_authorized": True, "exposure_counts": {p: schedule.count(p) for p in ROLES},
         "prefix_fixture_not_human_behavior": True}, ((key_id,),), prefix)


def constraint_task(domain="university_budget", variant=0, control="binding", phase="pilot"):
    options = DOMAINS[domain] + ("balanced portfolio",)
    # Two feasible compromises with unequal individual gains, plus unsafe
    # attractive proposals. Participants must actually negotiate and vote.
    values = ((52, 12, 37, 28), (14, 51, 25, 34), (17, 18, 33, 28), (13, 52, 25, 34))
    if phase == "formal":
        compromise_columns = {
            "university_budget": ((37,25,33,25), (28,34,28,34)),
            "supplier_selection": ((29,35,26,38), (35,25,34,28)),
            "emergency_response": ((42,24,34,26), (26,36,25,39)),
        }[domain]
        values = tuple((row[0], row[1], compromise_columns[0][i], compromise_columns[1][i])
                       for i, row in enumerate(values))
    shift = variant % 4
    base = {p: {o: float(values[(i+shift) % 4][j]) for j, o in enumerate(options)} for i, p in enumerate(ROLES)}
    regime = "nonbinding" if control != "binding" else "tight" if phase == "formal" and variant == 2 else "nominal"
    batna = (8. if regime == "nonbinding" else
             {"university_budget": 26., "supplier_selection": 26., "emergency_response": 25.}[domain]
             if regime == "tight" else 24.)
    task_id = f"{phase}_C_{domain}_v{variant}_{control}"
    winner = max(options, key=lambda o: sum(base[p][o] for p in ROLES))
    s = Scenario(task_id, domain, variant+1, "symmetric", ROLES, options,
                 base, {p: batna for p in ROLES}, (), winner)
    return StudyTask(task_id, "C", f"{domain}_constraint" if phase == "formal" else f"{domain}_constraint_v{variant}", s,
        "Synthetic full-information negotiation. Utilities represent individual preferences. "
        "Any agreement requires all four participants to accept the selected option; otherwise each receives their BATNA. "
        "No side payments or new options are permitted.",
        {"phase": phase, "control": control, "constraint_regime": regime, "complete_brief_authorized": False,
         "actual_sealed_votes_not_projected": True})


def pilot_tasks():
    return [acquisition_task(), entropy_task(control="key"),
            entropy_task(control="irrelevant"), constraint_task()]
