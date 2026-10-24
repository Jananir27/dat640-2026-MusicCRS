"""Part 4 R2: model-independent Rocchio feedback and candidate contract.

All vectors in a request must belong to the same embedding space. The module
does not load models, call an LLM, mutate the catalog, or own conversation state.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import re
from typing import Mapping, Sequence

import numpy as np


def name_key(value: str) -> str:
    return " ".join(re.findall(r"\w+", str(value).casefold()))


def values(value) -> list[str]:
    if value is None:
        return []
    return [str(v).strip() for v in (value if isinstance(value, (list, tuple)) else [value]) if str(v).strip()]


@dataclass
class FeedbackState:
    """Latest explicit vote wins; hard exclusions remain independent of votes."""
    votes: dict[str, int] = field(default_factory=dict)
    liked_artists: set[str] = field(default_factory=set)
    disliked_artists: set[str] = field(default_factory=set)
    excluded_track_ids: set[str] = field(default_factory=set)
    excluded_artists: set[str] = field(default_factory=set)

    def rate(self, track_id: str, positive: bool) -> None:
        self.votes[track_id] = 1 if positive else -1

    def rate_artist(self, artist: str, positive: bool) -> None:
        key = name_key(artist)
        self.liked_artists = {name_key(a) for a in self.liked_artists}
        self.disliked_artists = {name_key(a) for a in self.disliked_artists}
        (self.disliked_artists if positive else self.liked_artists).discard(key)
        (self.liked_artists if positive else self.disliked_artists).add(key)

    def excludes(self, metadata: dict) -> bool:
        artists = {name_key(a) for a in values(metadata.get("artist_name"))}
        track_id = metadata["track_id"]
        return (track_id in self.excluded_track_ids or self.votes.get(track_id) == -1
                or bool(artists & {name_key(a) for a in self.excluded_artists | self.disliked_artists}))

    @property
    def has_preferences(self) -> bool:
        return bool(self.votes or self.liked_artists or self.disliked_artists)


@dataclass
class RankedTrack:
    """Shared Janani/Tobi interface; metadata remains the catalog authority.

    score is a comparable, finite relevance prior (larger is better). scores
    may preserve lexical/semantic/fusion scores; do not compare their raw
    scales. Embeddings are optional and never serialized to the browser.
    """
    track_id: str
    metadata: dict
    score: float
    embedding: Sequence[float] | None = None
    scores: dict[str, float] = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)


def unit_vector(vector, dimension: int | None = None) -> np.ndarray | None:
    if vector is None:
        return None
    try:
        result = np.asarray(vector, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        return None
    if dimension is not None and result.size != dimension:
        return None
    norm = np.linalg.norm(result)
    return result / norm if np.isfinite(norm) and norm > 0 else None


class RocchioFeedback:
    """Normalize(alpha*q + beta*mean(likes) - gamma*mean(dislikes)).

    Dense embeddings have meaningful negative coordinates, so the sparse
    term-vector nonnegative clipping rule is deliberately not applied here.
    Ranking adds the change in query cosine to the base relevance prior.
    """
    def __init__(self, alpha=1.0, beta=0.75, gamma=0.25, weight=0.75):
        for value in (alpha, beta, gamma, weight):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Feedback weights must be finite and nonnegative")
        self.alpha, self.beta, self.gamma, self.weight = alpha, beta, gamma, weight

    def updated_query(self, query, feedback: FeedbackState, embeddings: Mapping[str, Sequence[float]]):
        original = unit_vector(query)
        dimension = len(original) if original is not None else None
        vectors = {}
        for track_id, vote in feedback.votes.items():
            vector = unit_vector(embeddings.get(track_id), dimension)
            if vector is not None:
                if dimension is None:
                    dimension = len(vector)
                vectors[track_id] = vector
        if dimension is None:
            return original
        result = self.alpha * original if original is not None else np.zeros(dimension)
        for vote, weight in ((1, self.beta), (-1, -self.gamma)):
            group = [vector for track_id, vector in vectors.items() if feedback.votes[track_id] == vote]
            if group:
                result = result + weight * np.mean(group, axis=0)
        updated = unit_vector(result, dimension)
        return updated if updated is not None else original

    def rerank(self, candidates: Sequence[RankedTrack], feedback: FeedbackState,
               embeddings: Mapping[str, Sequence[float]], query=None,
               excluded_ids: set[str] | None = None) -> list[RankedTrack]:
        original = unit_vector(query)
        updated = self.updated_query(query, feedback, embeddings)
        dimension = len(updated) if updated is not None else None
        liked = [(track_id, unit_vector(vector, dimension)) for track_id, vector in embeddings.items()
                 if feedback.votes.get(track_id) == 1]
        usable_feedback = any(unit_vector(embeddings.get(track_id), dimension) is not None for track_id in feedback.votes)
        result, seen = [], set()
        for candidate in candidates:
            if candidate.track_id in seen or candidate.track_id in (excluded_ids or set()):
                continue
            if candidate.track_id != candidate.metadata.get("track_id"):
                raise ValueError("Candidate identity does not match metadata")
            if not math.isfinite(candidate.score):
                raise ValueError("Candidate relevance must be finite")
            if feedback.excludes(candidate.metadata):
                continue
            seen.add(candidate.track_id)
            vector = unit_vector(candidate.embedding if candidate.embedding is not None
                                 else embeddings.get(candidate.track_id), dimension)
            delta = 0.0
            evidence = dict(candidate.evidence)
            if updated is not None and vector is not None and usable_feedback:
                before = float(vector @ original) if original is not None else 0.0
                after = float(vector @ updated)
                delta = after - before
                evidence.update(feedback_applied=True, original_cosine=before, updated_cosine=after)
                similarities = [(float(vector @ positive), track_id) for track_id, positive in liked
                                if positive is not None]
                if similarities:
                    similarity, track_id = max(similarities)
                    if similarity >= 0.5:
                        evidence.update(liked_track_id=track_id, liked_similarity=similarity)
            artist_match = bool({name_key(a) for a in values(candidate.metadata.get("artist_name"))}
                                & {name_key(a) for a in feedback.liked_artists})
            score = candidate.score + self.weight * delta + 0.1 * artist_match
            evidence["liked_artist_match"] = artist_match
            result.append(replace(candidate, score=score, evidence=evidence,
                                  scores={**candidate.scores, "base": candidate.score,
                                          "feedback_delta": delta, "final": score}))
        # Stable sort preserves the retriever's order for ties and absent feedback.
        return sorted(result, key=lambda candidate: candidate.score, reverse=True)
