from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PrivateFact:
    fact_id: str
    owner: str
    text: str
    topic: str
    utility_effects: Mapping[str, Mapping[str, float]]
    is_public: bool = False

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PrivateFact":
        return cls(
            fact_id=str(value["fact_id"]),
            owner=str(value["owner"]),
            text=str(value["text"]),
            topic=str(value["topic"]),
            utility_effects=value["utility_effects"],
            is_public=bool(value.get("is_public", False)),
        )


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    domain: str
    variant: int
    information_regime: str
    participants: tuple[str, ...]
    options: tuple[str, ...]
    base_utilities: Mapping[str, Mapping[str, float]]
    batna: Mapping[str, float]
    facts: tuple[PrivateFact, ...]
    optimal_decision: str

    def __post_init__(self) -> None:
        if self.information_regime not in {"symmetric", "hidden"}:
            raise ValueError("information_regime must be symmetric or hidden")
        if self.optimal_decision not in self.options:
            raise ValueError("optimal_decision must be one of options")
        if len(set(self.participants)) != len(self.participants):
            raise ValueError("participant identifiers must be unique")
        if not self.participants or not self.options or len(set(self.options)) != len(self.options):
            raise ValueError("participants and unique options are required")
        if set(self.base_utilities) != set(self.participants) or set(self.batna) != set(self.participants):
            raise ValueError("base utilities and BATNA must cover every participant exactly")
        if any(set(self.base_utilities[participant]) != set(self.options) for participant in self.participants):
            raise ValueError("base utilities must cover every option")
        if len({fact.fact_id for fact in self.facts}) != len(self.facts):
            raise ValueError("fact identifiers must be unique")
        for fact in self.facts:
            if fact.owner != "PUBLIC" and fact.owner not in self.participants:
                raise ValueError(f"unknown fact owner: {fact.owner}")
            if any(participant not in self.participants for participant in fact.utility_effects):
                raise ValueError(f"fact {fact.fact_id} affects an unknown participant")
            if any(option not in self.options for effects in fact.utility_effects.values() for option in effects):
                raise ValueError(f"fact {fact.fact_id} affects an unknown option")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Scenario":
        return cls(
            scenario_id=str(value["scenario_id"]),
            domain=str(value["domain"]),
            variant=int(value["variant"]),
            information_regime=str(value["information_regime"]),
            participants=tuple(value["participants"]),
            options=tuple(value["options"]),
            base_utilities=value["base_utilities"],
            batna=value["batna"],
            facts=tuple(PrivateFact.from_dict(item) for item in value["facts"]),
            optimal_decision=str(value["optimal_decision"]),
        )

    def fact_map(self) -> dict[str, PrivateFact]:
        return {fact.fact_id: fact for fact in self.facts}

    def initial_public_fact_ids(self) -> set[str]:
        if self.information_regime == "symmetric":
            return {fact.fact_id for fact in self.facts}
        return {fact.fact_id for fact in self.facts if fact.is_public}

    def utilities(self, option: str, fact_ids: Sequence[str]) -> dict[str, float]:
        if option not in self.options:
            raise ValueError(f"unknown option: {option}")
        result = {participant: float(self.base_utilities[participant][option]) for participant in self.participants}
        lookup = self.fact_map()
        for fact_id in fact_ids:
            fact = lookup[fact_id]
            for participant, effects in fact.utility_effects.items():
                result[participant] += float(effects.get(option, 0.0))
        return result

    def welfare(self, option: str, fact_ids: Sequence[str]) -> float:
        return sum(self.utilities(option, fact_ids).values())

    def best_option(self, fact_ids: Sequence[str]) -> str:
        return max(self.options, key=lambda option: (self.welfare(option, fact_ids), option))


@dataclass(frozen=True)
class DialogueEvent:
    run_id: str
    turn: int
    speaker: str
    content: str
    model: str
    disclosed_fact_ids: tuple[str, ...] = ()
    proposal: str | None = None
    event_type: str = "participant"
    action: str | None = None
    reason: str | None = None
    target_participant: str | None = None
    target_fact_id: str | None = None
    state_snapshot: Mapping[str, Any] | None = None
    attempt_count: int = 1
    observed_model: str | None = None
    response_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float | None = None
    attempt_failure_kinds: tuple[str, ...] = ()
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if self.turn < 1 or self.event_type not in {"participant", "facilitator"}:
            raise ValueError("invalid dialogue event turn or type")
        if self.attempt_count not in {1, 2}:
            raise ValueError("v2 dialogue requests allow at most one retry")
        if len(self.attempt_failure_kinds) != self.attempt_count - 1:
            raise ValueError("successful request audit does not match attempt count")
        if any(
            kind not in {"network_or_timeout", "server", "rate_limit"}
            for kind in self.attempt_failure_kinds
        ):
            raise ValueError("successful request contains an ineligible retry reason")
        if any(not isinstance(fact_id, str) for fact_id in self.disclosed_fact_ids):
            raise ValueError("disclosed fact identifiers must be strings")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DialogueEvent":
        payload = dict(value)
        payload["disclosed_fact_ids"] = tuple(payload.get("disclosed_fact_ids", ()))
        payload["attempt_failure_kinds"] = tuple(payload.get("attempt_failure_kinds", ()))
        return cls(**payload)


@dataclass(frozen=True)
class FacilitationState:
    turn: int
    normalized_entropy: float
    information_pooling_rate: float
    disclosed_fact_ids: tuple[str, ...]
    unresolved_fact_ids: tuple[str, ...]
    speaker_counts: Mapping[str, int]
    current_proposals: tuple[str, ...]


@dataclass(frozen=True)
class Intervention:
    action: str
    reason: str
    target_participant: str | None
    message: str
    target_fact_id: str | None = None
    generation_prompt: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    phase: str
    scenario_id: str
    provider: str
    model: str
    condition: str
    seed: int
    max_participant_messages: int = 40
    temperature: float = 0.0
    max_output_tokens: int = 512
    entropy_threshold: float = 0.55
    shapley_samples: int = 128
    protocol_version: str = "pre-pilot-2026-08-26-v2.1"
    ablation: str | None = None


@dataclass
class RunRecord:
    config: RunConfig
    status: str
    events: list[DialogueEvent]
    final_decision: str | None
    metrics: dict[str, float | int | bool | None]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float | None = None
    observed_model_versions: tuple[str, ...] = ()
    started_at: str | None = None
    completed_at: str | None = None
    retry_count: int = 0
    max_attempt_count: int = 1
    failure_kind: str | None = None
    failure_response_excerpt: str | None = None
    failed_request: Mapping[str, Any] | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"completed", "failed"}:
            raise ValueError("run status must be completed or failed")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
