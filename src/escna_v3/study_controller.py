"""Owner-action Shapley, entropy invitation, and public constraint advice.

Owner groups are an explicit action-space adaptation: no embeddings, semantic
clusters, LLM confidence values, or unproven complexity reduction.
"""
from itertools import permutations, combinations
from math import factorial
import random
from time import perf_counter
from .mechanisms import normalized_entropy


def decide_control(scenario, disclosed_fact_ids=(), counts=None, method="structured_exchange",
                   used_owners=(), complete_brief_authorized=False, seed=0):
    s = scenario
    started = perf_counter()
    known = frozenset(disclosed_fact_ids) | s.initial_public_fact_ids()
    if not known <= set(s.fact_map()):
        raise ValueError("unknown public record")
    owners = tuple(p for p in s.participants if p not in used_owners)
    counts = counts or {p: 0 for p in s.participants}
    audit = {"method": method, "utility_evaluations": 0, "permutations": 0,
             "embedding_calls": 0, "embedding_tokens": 0, "grouping": "record_custodian",
             "normalized_entropy": normalized_entropy([counts.get(p, 0) for p in s.participants])}
    private_methods = {"cluster_shapley", "greedy", "value_priority"}
    if method in private_methods and not complete_brief_authorized:
        raise PermissionError("private utility-based ranking requires authorized complete briefing")
    grouping_started = perf_counter()
    groups = {p: {f.fact_id for f in s.facts if f.owner == p} - known for p in owners}
    grouping_seconds = perf_counter() - grouping_started
    cache = {}
    def value(coalition):
        key = frozenset(coalition)
        if key not in cache:
            ids = known | set().union(*(groups[p] for p in key)) if key else known
            chosen = max(s.options, key=lambda o: (s.welfare(o, ids), -s.options.index(o)))
            cache[key] = s.welfare(chosen, tuple(s.fact_map()))
            audit["utility_evaluations"] += len(s.options) + 1
        return cache[key]
    target, advice = None, None
    if method in {"constraint", "neutral"}:
        if s.information_regime != "symmetric":
            raise PermissionError("constraint advice accepts public full-information tasks only")
        if method == "constraint":
            utilities = {o: s.utilities(o, tuple(s.fact_map())) for o in s.options}
            feasible = [o for o in s.options if all(utilities[o][p] >= s.batna[p] for p in s.participants)]
            pareto = [o for o in feasible if not any(
                all(utilities[q][p] >= utilities[o][p] for p in s.participants) and
                any(utilities[q][p] > utilities[o][p] for p in s.participants) for q in s.options)]
            ranges = {p: max(1., max(utilities[o][p] for o in s.options) - s.batna[p]) for p in s.participants}
            pick = max(pareto, key=lambda o: (min((utilities[o][p]-s.batna[p])/ranges[p] for p in s.participants), o)) if pareto else None
            advice = {"suggested_option": pick, "individually_rational_options": feasible,
                      "nondominated_feasible_options": pareto, "public_utilities": utilities,
                      "rule": "maximise the minimum normalised gain above BATNA among nondominated individually rational options; participants retain final choice"}
            audit["utility_evaluations"] = len(s.options)
        else:
            advice = {"rule": "invite participants to review the public utilities and negotiate an acceptable agreement; do not prescribe an option"}
        action = "public_negotiation_advice"
    else:
        if not owners:
            raise ValueError("no unused owner available")
        if method == "cluster_shapley":
            sampling_started = perf_counter()
            rng, scores = random.Random(seed), {p: 0. for p in owners}
            for _ in range(128):
                order = list(owners); rng.shuffle(order)
                seen = set()
                for p in order:
                    scores[p] += (value(seen | {p}) - value(seen)) / 128
                    seen.add(p)
            audit["sampling_seconds"] = perf_counter() - sampling_started
            audit["sampling_utility_evaluations"] = audit["utility_evaluations"]
            # Cold cache for exact validation: a fair, separately reported
            # diagnostic cost, never a free side effect of sampled evaluations.
            cache.clear()
            exact_started = perf_counter()
            exact = {p: 0. for p in owners}
            for order in permutations(owners):
                seen = set()
                for p in order:
                    exact[p] += (value(seen | {p}) - value(seen)) / factorial(len(owners))
                    seen.add(p)
            audit.update(permutations=128, scores=scores, exact_scores=exact,
                         exact_seconds=perf_counter()-exact_started,
                         exact_utility_evaluations=audit["utility_evaluations"]-audit["sampling_utility_evaluations"],
                         approximation_max_abs_error=max(abs(scores[p]-exact[p]) for p in owners),
                         approximation_top_matches_exact=max(owners, key=lambda p: (scores[p], -s.participants.index(p))) == max(owners, key=lambda p: (exact[p], -s.participants.index(p))),
                         exact_validation_permutations=factorial(len(owners)))
            target = max(owners, key=lambda p: (scores[p], -s.participants.index(p)))
        elif method == "greedy":
            scores = {p: value({p}) - value(set()) for p in owners}
            audit["scores"] = scores
            target = max(owners, key=lambda p: (scores[p], -s.participants.index(p)))
        elif method == "value_priority":
            # One-step full-truth decision quality improvement, same value as A.
            scores = {p: value({p}) - value(set()) for p in owners}
            audit["scores"] = scores
            target = max(owners, key=lambda p: (scores[p], -s.participants.index(p)))
        elif method == "entropy":
            target = min(owners, key=lambda p: (counts.get(p, 0), s.participants.index(p))) if audit["normalized_entropy"] < .9 else owners[0]
        elif method in {"structured_exchange", "fixed_round_robin"}:
            target = owners[0]
        else:
            raise ValueError("unknown controller")
        action = "invite_owner"
    audit.update(wall_seconds=perf_counter()-started, grouping_seconds=grouping_seconds, embedding_seconds=0.,
                 coalition_cache_entries=len(cache), complete_brief_authorized=complete_brief_authorized)
    return {"safe_action": {"action": action, "target_owner": target, "public_advice": advice},
            "private_audit": audit}
