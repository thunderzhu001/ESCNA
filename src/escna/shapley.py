from __future__ import annotations

import itertools
import math
import random
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence

from .models import PrivateFact

ValueFunction = Callable[[frozenset[str]], float]


def exact_shapley(items: Sequence[str], value: ValueFunction) -> dict[str, float]:
    """Compute exact Shapley values; intended only for small validation cases."""
    n = len(items)
    if n == 0:
        return {}
    factorial = __import__("math").factorial
    denominator = factorial(n)
    result = {item: 0.0 for item in items}
    for item in items:
        others = [candidate for candidate in items if candidate != item]
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(n - size - 1) / denominator
            for subset in itertools.combinations(others, size):
                coalition = frozenset(subset)
                result[item] += weight * (value(coalition | {item}) - value(coalition))
    return result


def permutation_shapley(
    items: Sequence[str], value: ValueFunction, samples: int = 128, seed: int = 0
) -> dict[str, float]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = random.Random(seed)
    result = {item: 0.0 for item in items}
    for _ in range(samples):
        order = list(items)
        rng.shuffle(order)
        coalition: frozenset[str] = frozenset()
        previous = value(coalition)
        for item in order:
            coalition = coalition | {item}
            current = value(coalition)
            result[item] += current - previous
            previous = current
    return {item: contribution / samples for item, contribution in result.items()}


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(sorted({token.lower() for token in re.findall(r"[A-Za-z0-9]+", text) if len(token) > 2}))


def tfidf_embeddings(facts: Sequence[PrivateFact]) -> dict[str, dict[str, float]]:
    """Create deterministic sparse TF--IDF vectors, including a topic feature."""
    documents = {fact.fact_id: list(_tokens(fact.text)) + [f"topic:{fact.topic.lower()}"] for fact in facts}
    document_frequency: dict[str, int] = defaultdict(int)
    for tokens in documents.values():
        for token in sorted(set(tokens)):
            document_frequency[token] += 1
    count = len(documents)
    vectors: dict[str, dict[str, float]] = {}
    for fact_id, tokens in documents.items():
        frequencies: dict[str, int] = defaultdict(int)
        for token in tokens:
            frequencies[token] += 1
        vectors[fact_id] = {
            token: (1.0 + math.log(frequency)) * (math.log((1.0 + count) / (1.0 + document_frequency[token])) + 1.0)
            for token, frequency in frequencies.items()
        }
    return vectors


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    numerator = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = sum(value * value for value in left.values()) ** 0.5
    right_norm = sum(value * value for value in right.values()) ** 0.5
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def cluster_embeddings(
    facts: Sequence[PrivateFact],
    vectors: dict[str, dict[str, float]],
    max_clusters: int = 12,
    similarity: float = 0.35,
) -> list[list[str]]:
    clusters: list[list[PrivateFact]] = []
    for fact in facts:
        best_index = -1
        best_score = -1.0
        for index, cluster in enumerate(clusters):
            centroid: dict[str, float] = defaultdict(float)
            for candidate in cluster:
                for token, value in vectors[candidate.fact_id].items():
                    centroid[token] += value / len(cluster)
            score = _cosine(vectors[fact.fact_id], dict(centroid))
            if score > best_score:
                best_index, best_score = index, score
        if best_index >= 0 and (best_score >= similarity or len(clusters) >= max_clusters):
            clusters[best_index].append(fact)
        else:
            clusters.append([fact])
    return [[fact.fact_id for fact in cluster] for cluster in clusters]


def cluster_facts(facts: Sequence[PrivateFact], max_clusters: int = 12, similarity: float = 0.35) -> list[list[str]]:
    """Dependency-free deterministic clustering over sparse TF--IDF embeddings."""
    return cluster_embeddings(facts, tfidf_embeddings(facts), max_clusters=max_clusters, similarity=similarity)


def cluster_shapley(
    clusters: Sequence[Sequence[str]],
    fact_value: ValueFunction,
    samples: int = 128,
    seed: int = 0,
) -> dict[str, float]:
    cluster_ids = [f"cluster_{index}" for index in range(len(clusters))]
    members = {cluster_id: tuple(clusters[index]) for index, cluster_id in enumerate(cluster_ids)}

    def cluster_value(coalition: frozenset[str]) -> float:
        fact_ids = frozenset(itertools.chain.from_iterable(members[item] for item in coalition))
        return fact_value(fact_ids)

    cluster_scores = permutation_shapley(cluster_ids, cluster_value, samples=samples, seed=seed)
    fact_scores: dict[str, float] = {}
    for cluster_id, fact_ids in members.items():
        singleton_weights = {
            fact_id: abs(fact_value(frozenset({fact_id})) - fact_value(frozenset())) for fact_id in fact_ids
        }
        total = sum(singleton_weights.values())
        for fact_id in fact_ids:
            share = singleton_weights[fact_id] / total if total else 1.0 / len(fact_ids)
            fact_scores[fact_id] = cluster_scores[cluster_id] * share
    return fact_scores


def rank_correlation(left: dict[str, float], right: dict[str, float]) -> float:
    """Spearman correlation without third-party dependencies; ties use average ranks."""
    keys = sorted(set(left) & set(right))
    if len(keys) < 2:
        return 1.0

    def ranks(values: dict[str, float]) -> dict[str, float]:
        ordered = sorted(keys, key=lambda key: values[key])
        grouped: dict[float, list[int]] = defaultdict(list)
        for index, key in enumerate(ordered, start=1):
            grouped[values[key]].append(index)
        return {key: sum(grouped[values[key]]) / len(grouped[values[key]]) for key in keys}

    a, b = ranks(left), ranks(right)
    mean_a = sum(a.values()) / len(keys)
    mean_b = sum(b.values()) / len(keys)
    numerator = sum((a[key] - mean_a) * (b[key] - mean_b) for key in keys)
    denom_a = sum((a[key] - mean_a) ** 2 for key in keys) ** 0.5
    denom_b = sum((b[key] - mean_b) ** 2 for key in keys) ** 0.5
    return numerator / (denom_a * denom_b) if denom_a and denom_b else 1.0
