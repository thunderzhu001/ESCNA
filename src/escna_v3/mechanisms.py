"""Independent, stateless v3 mechanisms and outcome evaluation.

This module has no provider, network, hash, logging, or frozen-v2 dependency.
The selector's full-truth brief is an offline authorised input.  Its language
payload contains an owner invitation only; fact text and private fact IDs stay
in selection metadata.

There are two intentionally different decision rules.  ``surrogate_decision``
uses the first declared option to resolve a tie, solely so the revised value
function is deterministic.  ``sealed_ballot`` uses unique plurality and
returns no decision on a tie or on all abstention; it never consults the
scenario's answer or welfare as a tie-break.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from escna.models import PrivateFact, Scenario
from escna.shapley import (
    cluster_embeddings,
    cluster_facts,
    cluster_shapley,
    exact_shapley,
    permutation_shapley,
    rank_correlation,
    tfidf_embeddings,
)


class AuthorizationError(ValueError):
    """A private fact was selected without explicit consent."""


def _ids(value: Iterable[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


@dataclass(frozen=True)
class _Math:
    fact_ids: tuple[str, ...]
    facts: Mapping[str, PrivateFact]
    participants: tuple[str, ...]
    options: tuple[str, ...]
    base: Mapping[str, Mapping[str, float]]
    effects: Mapping[str, Mapping[str, Mapping[str, float]]]
    base_welfare: Mapping[str, float]
    full: Mapping[str, Mapping[str, float]]
    full_welfare: Mapping[str, float]


def _prepare(scenario: Scenario) -> _Math:
    """Validate once and cache only local math for one public call."""

    participants = tuple(scenario.participants)
    options = tuple(scenario.options)
    facts = {fact.fact_id: fact for fact in scenario.facts}
    if len(facts) != len(scenario.facts):
        raise ValueError("fact identifiers must be unique")
    base: dict[str, dict[str, float]] = {}
    full: dict[str, dict[str, float]] = {}
    for option in options:
        base[option] = {}
        for person in participants:
            value = scenario.base_utilities[person][option]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("utility must be a finite number")
            base[option][person] = float(value)
        full[option] = dict(base[option])
    effects: dict[str, dict[str, dict[str, float]]] = {}
    for fact in scenario.facts:
        per_option: dict[str, dict[str, float]] = {option: {} for option in options}
        for person, option_effects in fact.utility_effects.items():
            if person not in participants:
                raise ValueError(f"fact {fact.fact_id} affects an unknown participant")
            for option, value in option_effects.items():
                if option not in options:
                    raise ValueError(f"fact {fact.fact_id} affects an unknown option")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"fact {fact.fact_id} has a non-finite utility effect")
                per_option[option][person] = float(value)
                full[option][person] += float(value)
        effects[fact.fact_id] = per_option
    return _Math(
        tuple(fact.fact_id for fact in scenario.facts),
        facts,
        participants,
        options,
        base,
        effects,
        {option: float(sum(values.values())) for option, values in base.items()},
        full,
        {option: float(sum(values.values())) for option, values in full.items()},
    )


def _checked_ids(ctx: _Math, value: Iterable[str] | str | None) -> tuple[str, ...]:
    ids = _ids(value)
    if len(ids) != len(set(ids)):
        raise ValueError("fact identifiers must be unique")
    unknown = [fact_id for fact_id in ids if fact_id not in ctx.facts]
    if unknown:
        raise ValueError(f"unknown fact identifier(s): {', '.join(unknown)}")
    return ids


def _utility(ctx: _Math, option: str, ids: Iterable[str]) -> dict[str, float]:
    if option not in ctx.options:
        raise ValueError(f"unknown option: {option}")
    result = dict(ctx.base[option])
    for fact_id in ids:
        for person, value in ctx.effects[fact_id][option].items():
            result[person] += value
    return result


def robust_utility(
    scenario: Scenario,
    option: str,
    fact_ids: Iterable[str] | str | None = None,
    *,
    participant: str | None = None,
) -> dict[str, float] | float:
    """One utility routine for partial or full truth.

    Omitting ``fact_ids`` means all true facts, even for a hidden-information
    scenario.  A participant may be supplied when one scalar is needed.
    """

    ctx = _prepare(scenario)
    if participant is not None and participant not in ctx.participants:
        raise ValueError(f"unknown participant: {participant}")
    ids = ctx.fact_ids if fact_ids is None else _checked_ids(ctx, fact_ids)
    result = _utility(ctx, option, ids)
    return result[participant] if participant is not None else result


def _surrogate(ctx: _Math, ids: Iterable[str]) -> str:
    selected = set(ids)
    welfare = dict(ctx.base_welfare)
    for fact_id in selected:
        for option in ctx.options:
            welfare[option] += sum(ctx.effects[fact_id][option].values())
    # First option wins an exact tie.  This rule belongs only to d(D).
    return max(ctx.options, key=lambda option: (welfare[option], -ctx.options.index(option)))


def _truth_decision(ctx: _Math) -> str | None:
    best = max(ctx.full_welfare.values())
    leaders = [option for option in ctx.options if math.isclose(ctx.full_welfare[option], best, rel_tol=1e-12, abs_tol=1e-12)]
    return leaders[0] if len(leaders) == 1 else None


def _nondominated(ctx: _Math, option: str) -> bool:
    return not any(
        other != option
        and all(ctx.full[other][person] >= ctx.full[option][person] for person in ctx.participants)
        and any(ctx.full[other][person] > ctx.full[option][person] for person in ctx.participants)
        for other in ctx.options
    )


def surrogate_decision(scenario: Scenario, fact_ids: Iterable[str] | str | None = None) -> str:
    ctx = _prepare(scenario)
    ids = ctx.fact_ids if fact_ids is None else _checked_ids(ctx, fact_ids)
    return _surrogate(ctx, ids)


def revised_information_value(
    scenario: Scenario,
    disclosed_ids: Iterable[str] | str | None,
    candidate_ids: Iterable[str] | str | None = None,
) -> float:
    """Return ``U_full(d(D union S)) - U_full(d(D))``.

    ``U_full`` is full-truth welfare of the selected surrogate option.  This
    is an authorised offline decision-support value, not an LLM effect.
    """

    ctx = _prepare(scenario)
    disclosed = _checked_ids(ctx, disclosed_ids)
    candidate = _checked_ids(ctx, candidate_ids)
    before = _surrogate(ctx, disclosed)
    union = set(disclosed).union(candidate)
    after = _surrogate(ctx, (fact_id for fact_id in ctx.fact_ids if fact_id in union))
    return float(ctx.full_welfare[after] - ctx.full_welfare[before])


def normalized_entropy(counts: Mapping[str, int | float] | Sequence[int | float] | None) -> float:
    """Shannon entropy divided by ``log(number of speakers)`` and bounded."""

    values = list(counts.values()) if isinstance(counts, Mapping) else list(counts or ())
    if not values:
        return 0.0
    clean: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("exposure counts must be finite numbers")
        if float(value) < 0:
            raise ValueError("exposure counts must be non-negative")
        clean.append(float(value))
    total = sum(clean)
    if total <= 0.0 or len(clean) <= 1:
        return 0.0
    entropy = -sum((value / total) * math.log(value / total) for value in clean if value)
    return min(1.0, max(0.0, float(entropy / math.log(len(clean)))))


@dataclass(frozen=True)
class InformationSelection:
    """Selection metadata and a safe owner-only action."""

    action: str
    target_owner: str | None
    timing: Any = None
    selected_fact_ids: tuple[str, ...] = ()
    score: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def language_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": self.action}
        if self.target_owner is not None:
            payload["target_owner"] = self.target_owner
        if self.timing is not None:
            payload["timing"] = self.timing
        return payload

    @property
    def target_participant(self) -> str | None:
        return self.target_owner


def _empty_selection(method: str, disclosed: tuple[str, ...], candidates: tuple[str, ...], owner: str | None, timing: Any, authorized: bool, reason: str) -> InformationSelection:
    return InformationSelection(
        "no_action",
        None,
        timing,
        metadata={
            "method": method,
            "disclosed_fact_ids": disclosed,
            "candidate_fact_ids": candidates,
            "scheduled_owner": owner,
            "timing": timing,
            "complete_brief_authorized": authorized,
            "reason": reason,
            "value_function": "U_full(d(D union S)) - U_full(d(D))",
            "surrogate_tie_break": "first scenario option; surrogate only",
        },
    )


def choose_information_action(
    scenario: Scenario,
    method: str = "greedy",
    disclosed_ids: Iterable[str] | str | None = (),
    scheduled_owner: str | None = None,
    timing: Any = None,
    *,
    complete_brief_authorized: bool = False,
    candidate_fact_ids: Iterable[str] | str | None = None,
    samples: int = 128,
    seed: int = 0,
    positive_only: bool = False,
) -> InformationSelection:
    """Select one fact and return only ``invite_owner`` metadata.

    Methods are ``greedy``, ``shapley``, ``cluster_shapley``,
    ``exact_shapley`` (at most eight facts), and ``round_robin``.  A supplied
    scheduled owner is a hard constraint.  The revised ``V`` reads the full
    truth brief, so it requires the explicit ``complete_brief_authorized=True``
    bool even when an action candidate list is restricted.  The candidate list
    is an action control, not a claim that the value function saw only that
    subset.  Equal scores use ascending fact ID.  Non-positive scores are
    retained unless ``positive_only`` is requested, so bad-news and zero-value
    controls stay observable.
    """

    method = str(method).lower().strip()
    if method not in {"greedy", "shapley", "cluster_shapley", "exact_shapley", "round_robin"}:
        raise ValueError(f"unknown information-selection method: {method}")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    ctx = _prepare(scenario)
    disclosed = _checked_ids(ctx, disclosed_ids)
    if not isinstance(complete_brief_authorized, bool) or not complete_brief_authorized:
        raise AuthorizationError("complete_brief_authorized=True is required for V(S)")
    requested = ctx.fact_ids if candidate_fact_ids is None else _checked_ids(ctx, candidate_fact_ids)
    disclosed_set = set(disclosed)
    requested_set = set(requested)
    candidates = tuple(fact_id for fact_id in ctx.fact_ids if fact_id in requested_set and fact_id not in disclosed_set)
    if scheduled_owner is not None:
        if scheduled_owner not in ctx.participants and scheduled_owner != "PUBLIC":
            raise ValueError(f"unknown scheduled owner: {scheduled_owner}")
        candidates = tuple(fact_id for fact_id in candidates if ctx.facts[fact_id].owner == scheduled_owner)
    elif method == "round_robin":
        # Stateless calls cannot remember the previous turn.  Use participant
        # declaration order as the frozen starting point, then fact ID within
        # that owner's turn; callers with a real schedule pass scheduled_owner.
        for owner in (*ctx.participants, "PUBLIC"):
            owner_candidates = tuple(fact_id for fact_id in candidates if ctx.facts[fact_id].owner == owner)
            if owner_candidates:
                candidates = owner_candidates
                break
    if not candidates:
        return _empty_selection(method, disclosed, candidates, scheduled_owner, timing, complete_brief_authorized, "no_undisclosed_candidate_fact")

    before = _surrogate(ctx, disclosed)

    def value(coalition: frozenset[str]) -> float:
        union = disclosed_set.union(coalition)
        after = _surrogate(ctx, (fact_id for fact_id in ctx.fact_ids if fact_id in union))
        return float(ctx.full_welfare[after] - ctx.full_welfare[before])

    if method == "greedy":
        scores = {fact_id: value(frozenset((fact_id,))) for fact_id in candidates}
    elif method == "shapley":
        scores = permutation_shapley(candidates, value, samples=samples, seed=seed)
    elif method == "exact_shapley":
        if len(candidates) > 8:
            raise ValueError("exact_shapley is restricted to at most 8 facts")
        scores = exact_shapley(candidates, value)
    elif method == "cluster_shapley":
        clusters = cluster_facts([ctx.facts[fact_id] for fact_id in candidates])
        scores = cluster_shapley(clusters, value, samples=samples, seed=seed)
    else:
        # Round-robin is the cheap exposure control; it intentionally does not
        # use value to decide whose turn it is.
        scores = {fact_id: 0.0 for fact_id in candidates}
    selected = min(candidates, key=lambda fact_id: (-float(scores[fact_id]), fact_id))
    score = float(scores[selected])
    if positive_only and score <= 0.0:
        return _empty_selection(method, disclosed, candidates, scheduled_owner, timing, complete_brief_authorized, "no_positive_information_value")
    owner = ctx.facts[selected].owner
    metadata: dict[str, Any] = {
        "method": method,
        "disclosed_fact_ids": disclosed,
        "candidate_fact_ids": candidates,
        "selected_fact_ids": (selected,),
        "selected_owner": owner,
        "scheduled_owner": scheduled_owner,
        "timing": timing,
        "scores": dict(scores),
        "score": score,
        "complete_brief_authorized": complete_brief_authorized,
        "value_function": "U_full(d(D union S)) - U_full(d(D))",
        "surrogate_tie_break": "first scenario option; surrogate only",
        "selection_tie_break": "highest score, then fact_id ascending",
    }
    if method == "cluster_shapley":
        metadata["clusters"] = tuple(tuple(cluster) for cluster in clusters)
    return InformationSelection("invite_owner", owner, timing, (selected,), score, metadata)


def _resolve(ballots: Mapping[str, Mapping[str, Any]], options: Sequence[str]) -> tuple[str | None, dict[str, int], dict[str, Mapping[str, Any]], str]:
    if not isinstance(ballots, Mapping):
        raise ValueError("ballots must be a participant-to-dict mapping")
    allowed = set(options)
    tally: Counter[str] = Counter()
    normalized: dict[str, Mapping[str, Any]] = {}
    abstains = 0
    for person, ballot in ballots.items():
        if not isinstance(ballot, Mapping):
            raise ValueError(f"ballot for {person} must be a mapping")
        recommendation = ballot.get("recommendation", "ABSTAIN")
        if not isinstance(recommendation, str):
            raise ValueError("recommendation must be a string")
        acceptable = ballot.get("acceptable_options", ())
        if isinstance(acceptable, str) or not isinstance(acceptable, (list, tuple)):
            raise ValueError("acceptable_options must be a list")
        if len(set(acceptable)) != len(acceptable) or any(item not in allowed for item in acceptable):
            raise ValueError("acceptable_options contains an invalid or duplicate option")
        if recommendation == "ABSTAIN" and acceptable:
            raise ValueError("ABSTAIN ballots must have an empty acceptable_options")
        if recommendation != "ABSTAIN" and recommendation not in acceptable:
            raise ValueError("recommendation must be one of acceptable_options")
        normalized[str(person)] = ballot
        if recommendation == "ABSTAIN":
            abstains += 1
        elif recommendation in allowed:
            tally[recommendation] += 1
    if not tally:
        return None, dict(tally), normalized, "abstain" if abstains else "no_valid_vote"
    top = max(tally.values())
    leaders = [option for option in options if tally.get(option, 0) == top]
    if len(leaders) != 1:
        return None, dict(tally), normalized, "tie"
    return leaders[0], dict(tally), normalized, "unique_plurality"


def sealed_ballot(ballots: Mapping[str, Mapping[str, Any]], options: Sequence[str]) -> str | None:
    """Resolve v3 dict ballots by unique plurality; tie/abstain => ``None``."""

    return _resolve(ballots, options)[0]


def full_truth_decision(scenario: Scenario) -> str | None:
    ctx = _prepare(scenario)
    return _truth_decision(ctx)


def nondominated_options(scenario: Scenario) -> tuple[str, ...]:
    ctx = _prepare(scenario)
    return tuple(option for option in ctx.options if _nondominated(ctx, option))


@dataclass(frozen=True)
class BallotEvaluation:
    decision: str | None
    full_truth_decision: str | None
    full_truth_accuracy: bool | None
    unanimity_accept: bool
    batna_satisfied: bool
    nondominated: bool
    decision_utilities: Mapping[str, float]
    payoffs: Mapping[str, float]
    batna_by_participant: Mapping[str, bool]
    no_decision: bool
    tally: Mapping[str, int]
    reason: str
    batna_evaluated: bool

    @property
    def protocol_payoff(self) -> float:
        return float(sum(self.payoffs.values()))

    def metrics(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "full_truth_accuracy": self.full_truth_accuracy,
            "unanimity_accept": self.unanimity_accept,
            "batna_satisfied": self.batna_satisfied,
            "nondominated": self.nondominated,
            "no_decision": self.no_decision,
            "batna_evaluated": self.batna_evaluated,
            "protocol_payoff": self.protocol_payoff,
            "payoffs": dict(self.payoffs),
            "tally": dict(self.tally),
        }


def evaluate_sealed_ballots(scenario: Scenario, ballots: Mapping[str, Mapping[str, Any]]) -> BallotEvaluation:
    """Evaluate full-truth accuracy, acceptance, BATNA, and non-dominance separately."""

    ctx = _prepare(scenario)
    if set(ballots) != set(ctx.participants):
        raise ValueError("sealed ballot mapping must contain every participant exactly once")
    decision, tally, normalized, reason = _resolve(ballots, ctx.options)
    truth_label = _truth_decision(ctx)
    if decision is None:
        payoffs = {person: float(scenario.batna[person]) for person in ctx.participants}
        return BallotEvaluation(
            None,
            truth_label,
            False,
            False,
            False,
            False,
            {},
            payoffs,
            {person: False for person in ctx.participants},
            True,
            tally,
            reason,
            False,
        )
    utilities = dict(ctx.full[decision])
    batna_by = {person: utilities[person] >= float(scenario.batna[person]) for person in ctx.participants}
    accepted = all(
        person in normalized
        and normalized[person].get("recommendation") != "ABSTAIN"
        and decision in normalized[person].get("acceptable_options", ())
        for person in ctx.participants
    )
    # A selected but not unanimously accepted option is not an achieved
    # agreement: its realized protocol payoff is BATNA, while candidate
    # utilities remain separately available for BATNA/Pareto metrics.
    payoffs = dict(utilities) if accepted else {person: float(scenario.batna[person]) for person in ctx.participants}
    return BallotEvaluation(
        decision,
        truth_label,
        None if truth_label is None else bool(decision == truth_label),
        bool(accepted),
        all(batna_by.values()),
        _nondominated(ctx, decision),
        utilities,
        payoffs,
        batna_by,
        False,
        tally,
        reason,
        True,
    )


def _rank_or_na(left: Mapping[str, float] | None, right: Mapping[str, float] | None) -> float | None:
    if not left or not right:
        return None
    keys = sorted(set(left).intersection(right))
    if len(keys) < 2:
        return None
    a = [float(left[key]) for key in keys]
    b = [float(right[key]) for key in keys]
    if len(set(a)) <= 1 or len(set(b)) <= 1:
        return None
    # The reused scalar arithmetic can drift by a few ulps outside [-1, 1].
    return min(1.0, max(-1.0, float(rank_correlation(dict(left), dict(right)))))


def validate_shapley_cost(
    scenario: Scenario,
    disclosed_ids: Iterable[str] | str | None = (),
    candidate_fact_ids: Iterable[str] | str | None = None,
    *,
    samples: int = 128,
    seed: int = 0,
    max_exact_items: int = 8,
) -> dict[str, Any]:
    """Offline exact/permutation/cluster comparison with explicit call costs."""

    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if not isinstance(max_exact_items, int) or isinstance(max_exact_items, bool) or max_exact_items < 0:
        raise ValueError("max_exact_items must be a non-negative integer")
    preparation_started = time.perf_counter()
    ctx = _prepare(scenario)
    preparation_seconds = time.perf_counter() - preparation_started
    disclosed = _checked_ids(ctx, disclosed_ids)
    disclosed_set = set(disclosed)
    raw_candidates = ctx.fact_ids if candidate_fact_ids is None else _checked_ids(ctx, candidate_fact_ids)
    candidates = tuple(fact_id for fact_id in raw_candidates if fact_id not in disclosed_set)

    before = _surrogate(ctx, disclosed)

    def value(coalition: frozenset[str]) -> float:
        union = disclosed_set.union(coalition)
        after = _surrogate(ctx, (fact_id for fact_id in ctx.fact_ids if fact_id in union))
        return float(ctx.full_welfare[after] - ctx.full_welfare[before])

    result: dict[str, Any] = {
        "items": candidates,
        "disclosed_fact_ids": disclosed,
        "samples": samples,
        "seed": seed,
        "value_function": "U_full(d(D union S)) - U_full(d(D))",
        "prepare_runtime_seconds": preparation_seconds,
        "exact_scores": None,
        "permutation_scores": {},
        "cluster_scores": {},
        "clusters": (),
        "exact_value_calls": 0,
        "permutation_value_calls": 0,
        "cluster_value_calls": 0,
        "exact_runtime_seconds": None,
        "permutation_runtime_seconds": 0.0,
        "cluster_runtime_seconds": 0.0,
        "tfidf_runtime_seconds": 0.0,
        "rank_exact_vs_permutation": None,
        "rank_exact_vs_cluster": None,
        "rank_permutation_vs_cluster": None,
    }
    if len(candidates) <= max_exact_items:
        calls = 0

        def exact_value(coalition: frozenset[str]) -> float:
            nonlocal calls
            calls += 1
            return value(coalition)

        started = time.perf_counter()
        result["exact_scores"] = exact_shapley(candidates, exact_value)
        result["exact_runtime_seconds"] = time.perf_counter() - started
        result["exact_value_calls"] = calls
    calls = 0

    def permutation_value(coalition: frozenset[str]) -> float:
        nonlocal calls
        calls += 1
        return value(coalition)

    started = time.perf_counter()
    result["permutation_scores"] = permutation_shapley(candidates, permutation_value, samples=samples, seed=seed)
    result["permutation_runtime_seconds"] = time.perf_counter() - started
    result["permutation_value_calls"] = calls
    facts = [ctx.facts[fact_id] for fact_id in candidates]
    started = time.perf_counter()
    vectors = tfidf_embeddings(facts)
    result["tfidf_runtime_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    clusters = cluster_embeddings(facts, vectors)
    result["cluster_runtime_seconds"] = time.perf_counter() - started
    calls = 0

    def cluster_value(coalition: frozenset[str]) -> float:
        nonlocal calls
        calls += 1
        return value(coalition)

    started = time.perf_counter()
    result["cluster_scores"] = cluster_shapley(clusters, cluster_value, samples=samples, seed=seed)
    result["cluster_runtime_seconds"] = time.perf_counter() - started
    result["cluster_value_calls"] = calls
    result["clusters"] = tuple(tuple(cluster) for cluster in clusters)
    result["rank_exact_vs_permutation"] = _rank_or_na(result["exact_scores"], result["permutation_scores"])
    result["rank_exact_vs_cluster"] = _rank_or_na(result["exact_scores"], result["cluster_scores"])
    result["rank_permutation_vs_cluster"] = _rank_or_na(result["permutation_scores"], result["cluster_scores"])
    return result


__all__ = [
    "AuthorizationError",
    "BallotEvaluation",
    "InformationSelection",
    "choose_information_action",
    "evaluate_sealed_ballots",
    "full_truth_decision",
    "nondominated_options",
    "normalized_entropy",
    "revised_information_value",
    "robust_utility",
    "sealed_ballot",
    "surrogate_decision",
    "validate_shapley_cost",
]
