"""Isolated experimental ranking that lets feedback compete with lexical rank.

No evaluation labels, query IDs, artist-specific rules or model tuning are read
here. Query constraints, seed styles and artist relationships use catalog facts.
The previous application and feedback_retrieval implementation are unchanged.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

import numpy as np

from .feedback_retrieval import FeedbackAwareRetriever, FeedbackRetrievalResult, MODES
from .feedback_language import split_exclusion_request
from .music_features import classify_tags, extract_preferences, track_style
from .relevance_feedback import FeedbackState, RankedTrack, name_key, unit_vector, values


@dataclass(frozen=True)
class AdaptiveConfig:
    influence: float = 0.8
    negative_weight: float = 0.25
    mood_weight: float = 0.15
    diversity_penalty: float = 0.04

    def __post_init__(self):
        for value in (self.influence, self.negative_weight, self.mood_weight, self.diversity_penalty):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Adaptive weights must be finite numbers between zero and one")


_PARENTS = {
    "rock": "alternative rock|alt rock|hard rock|classic rock|progressive rock|indie rock|grunge|post grunge|punk rock|pop rock|psychedelic rock|southern rock|garage rock|folk rock|blues rock|nu metal|alternative metal|rap rock|rap metal",
    "metal": "nu metal|alternative metal|rap metal|heavy metal|progressive metal|groove metal|metalcore|thrash metal|death metal|black metal|power metal|melodic death metal",
    "hip hop": "hip hop rap|hiphop|rap|trap|trap rap|gangsta rap|conscious hip hop|west coast hip hop|east coast hip hop|alternative hip hop|jazz rap|jazz hop|hip hop jazz",
    "pop": "indie pop|latin pop|chamber pop|baroque pop|orchestral pop|art pop|dream pop|electropop|electro pop|synth pop|dance pop|bubblegum pop|teen pop|pop rock|folk pop|k pop|j pop",
    "electronic": "electronica|edm|house|deep house|progressive house|acid house|techno|synth pop|electropop|electro pop|electro|trip hop|downtempo|big beat|chillwave|trance",
    "dance": "house|deep house|progressive house|acid house|dance pop|electro house|disco house|french house|eurodance",
    "r&b": "r b soul|r b|rnb|rhythm and blues|neo soul|quiet storm|contemporary rnb|alternative rnb",
    "soul": "r b soul|neo soul|blue eyed soul|quiet storm|pop soul",
    "jazz": "modal jazz|cool jazz|hard bop|bebop|jazz fusion|fusion jazz|acid jazz|vocal jazz|jazz vocal|smooth jazz|contemporary jazz|instrumental jazz|free jazz",
}
_PARENTS = {parent: set(aliases.split("|")) for parent, aliases in _PARENTS.items()}
_SPECIFIC = set().union(*_PARENTS.values()) | {"post hardcore", "shoegaze", "emo", "industrial rock", "industrial metal"}


def clean_request(query):
    """Remove initial recommendation filler; preserve musical names and context."""
    text = re.sub(r"^(?:please\s+)?(?:recommend|suggest)(?:\s+me)?(?:\s+(?:some|a few|more))?\s+", "", query, flags=re.I)
    return re.sub(r"\s+(?:songs?|tracks?|music)\s*[.!?]*$", "", text, flags=re.I).strip()


def request_preferences(request):
    """Small bridge for explicit request terms, not artist-name genre inference.

    Callers with structured R1 constraints should pass genres/moods directly.
    'Similar to Daft Punk' must not turn the artist's name into a punk constraint.
    """
    text, _ = split_exclusion_request(request.split(";")[0])
    text = re.split(r"\b(?:similar to|more like|like)\b", text, maxsplit=1, flags=re.I)[0]
    genres, moods = extract_preferences(text)
    return set(genres), set(moods), "alternative rock" in name_key(text)


def metadata_features(row):
    tags = values(row.get("tag_list"))
    if isinstance(row.get("tag_list"), str):
        tags = row["tag_list"].split(",")
    keys = {name_key(tag) for tag in tags}
    genres, moods = classify_tags(tags)
    for parent, aliases in _PARENTS.items():
        if keys & aliases:
            genres.add(parent)
    style = dict(track_style({"tag_list": tags}))
    style.update({genre: 1.0 for genre in genres})
    style.update({key: 2.0 for key in keys & _SPECIFIC})
    return genres, moods, style, keys


def style_cosine(left, right):
    norm = math.sqrt(sum(v * v for v in left.values()) * sum(v * v for v in right.values()))
    return sum(v * right.get(k, 0) for k, v in left.items()) / norm if norm else 0.0


class AdaptiveMusicRetriever(FeedbackAwareRetriever):
    """Seed candidate expansion and explicit request/style scoring.

    The parent provides the validated index and BM25/full-catalog search helpers.
    All three modes share query cleanup, genre eligibility and diversity policy.
    Only feedback modes use seed preferences in their scoring. Candidate budgets
    and extra seed-centroid retrieval are reported in per-track evidence.
    """

    def __init__(self, catalog, lexical, dense, *, config=None, **kwargs):
        super().__init__(catalog, lexical, dense, **kwargs)
        self.config = config if config is not None else AdaptiveConfig()
        self.features = {t: metadata_features(row) for t, row in self.metadata.items()}

    def retrieve(self, query, feedback: FeedbackState, *, topk=5, mode="feedback_retrieval",
                 excluded_ids=(), allowed_ids=None, hide_liked=True, request=None,
                 genres=None, moods=None):
        if mode not in MODES or type(topk) is not int or topk < 0:
            raise ValueError("Invalid retrieval mode or topk")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("A nonempty original query is required")
        if topk == 0:
            return FeedbackRetrievalResult(mode, [], (), (), (), False, (), ())
        request = query if request is None else request
        inferred_genres, inferred_moods, alternative_rock = request_preferences(request)
        required = inferred_genres if genres is None else set(genres)
        wanted_moods = inferred_moods if moods is None else set(moods)
        alternative_rock = alternative_rock and {"alternative", "rock"}.issubset(required)
        query = clean_request(query)
        if not query:
            raise ValueError("The request must contain musical content")
        depth = max(self.candidate_k, topk)
        excluded = set(excluded_ids) | set(feedback.excluded_track_ids)
        excluded.update(t for t, vote in feedback.votes.items() if vote == -1 or (hide_liked and vote == 1))
        # Also block another catalog ID for the same title/artist identity.
        song_keys = {(name_key(title), artist) for t in excluded if t in self.metadata
                     for title in values(self.metadata[t].get("track_name")) for artist in self.artists[t]}
        blocked = {name_key(a) for a in feedback.excluded_artists | feedback.disliked_artists}
        requested = set(self.metadata) if allowed_ids is None else set(allowed_ids) & self.metadata.keys()
        allowed = set()
        for t in requested:
            row_genres, _, _, keys = self.features[t]
            if t in excluded or self.artists[t] & blocked or (required and not row_genres & required):
                continue
            if alternative_rock and not (keys & {"alternative rock", "alt rock"}
                                         or {"alternative", "rock"}.issubset(row_genres)):
                continue
            if any((name_key(title), artist) in song_keys for title in values(self.metadata[t].get("track_name"))
                   for artist in self.artists[t]):
                continue
            allowed.add(t)
        eligible = np.array([t in allowed for t in self.track_ids], dtype=bool)
        original = unit_vector(self.dense.encode_query(query), self.dimension)
        if original is None:
            raise ValueError("Invalid query embedding or embedding space mismatch")
        seeds = {t: self.matrix[self.positions[t]] for t in feedback.votes if t in self.positions}
        positives = [t for t in seeds if feedback.votes[t] == 1]
        negatives = [t for t in seeds if feedback.votes[t] == -1]
        positive_rows = [t for t, vote in feedback.votes.items() if vote == 1 and t in self.metadata]
        negative_rows = [t for t, vote in feedback.votes.items() if vote == -1 and t in self.metadata]
        updated = self.rocchio.updated_query(original, feedback, seeds)
        query_changed = bool(seeds) and not np.allclose(original, updated, atol=1e-7, rtol=1e-7)
        lexical = self._lexical(query, allowed, depth) if self.lexical_weight else []
        original_semantic, original_scores = self._semantic(original, eligible, depth)
        semantic, semantic_scores = original_semantic, original_scores
        seed_candidates = []
        query_used = mode == "feedback_retrieval" and query_changed and self.dense_weight > 0
        if query_used:
            semantic, semantic_scores = self._semantic(updated, eligible, depth)
        if mode == "feedback_retrieval" and positives and self.dense_weight:
            center = unit_vector(np.mean([seeds[t] for t in positives], axis=0))
            if center is not None:
                seed_candidates, _ = self._semantic(center, eligible, depth)
        if not self.dense_weight:
            semantic, original_semantic, seed_candidates = [], [], []
        original_pool = tuple(dict.fromkeys(lexical + original_semantic))
        original_set = set(original_pool)
        # Retain original candidates so expansion cannot remove baseline choices.
        pool = tuple(dict.fromkeys(list(original_pool) + semantic + seed_candidates))
        ranks = [{t: rank for rank, t in enumerate(branch, 1)} for branch in (lexical, semantic)]
        normalizer = (self.lexical_weight + self.dense_weight) / (self.rrf_k + 1)
        ranked = []
        liked_artists = set().union(*(self.artists[t] for t in positive_rows)) if positive_rows else set()
        liked_artists.update(name_key(a) for a in feedback.liked_artists)
        has_feedback = bool(positive_rows or negative_rows or feedback.liked_artists)
        strength = self.config.influence if mode != "baseline" and has_feedback else 0.0
        for t in pool:
            rrf = sum(weight / (self.rrf_k + branch[t]) for branch, weight in
                      zip(ranks, (self.lexical_weight, self.dense_weight)) if t in branch)
            prior = rrf / normalizer
            _, track_moods, styles, _ = self.features[t]
            mood_fit = len(wanted_moods & track_moods) / len(wanted_moods) if wanted_moods else 0.0
            pos_style = max((style_cosine(styles, self.features[p][2]) for p in positive_rows), default=0.0)
            neg_style = max((style_cosine(styles, self.features[n][2]) for n in negative_rows), default=0.0)
            artist_match = bool(self.artists[t] & liked_artists)
            position = self.positions.get(t)
            vector = self.matrix[position] if position is not None else None
            pos_sim = float(np.mean([vector @ seeds[p] for p in positives])) if vector is not None and positives else 0.0
            neg_sim = float(np.mean([vector @ seeds[n] for n in negatives])) if vector is not None and negatives else 0.0
            if positive_rows:
                context = 0.5 * pos_sim + 0.3 * pos_style + 0.2 * artist_match
            else:
                context = float(vector @ updated) if vector is not None else prior
                if feedback.liked_artists:
                    context = 0.8 * context + 0.2 * artist_match
            context -= self.config.negative_weight * (0.75 * neg_sim + 0.25 * neg_style)
            score = (1 - strength) * prior + strength * context + self.config.mood_weight * mood_fit
            evidence = {"feedback_applied": bool(strength), "feedback_query_used_for_retrieval": query_used,
                        "new_to_original_pool": t not in original_set, "seed_candidate": t in seed_candidates,
                        "liked_artist_match": artist_match and bool(strength),
                        "positive_style_similarity": pos_style, "negative_style_similarity": neg_style,
                        "requested_genres": sorted(required), "requested_moods": sorted(wanted_moods),
                        "mood_match": sorted(wanted_moods & track_moods)}
            if vector is not None:
                evidence.update(original_cosine=float(original_scores[position]), updated_cosine=float(vector @ updated))
                if positives:
                    similarity, positive_id = max((float(vector @ seeds[p]), p) for p in positives)
                    if similarity >= 0.5 and strength:
                        evidence.update(liked_track_id=positive_id, liked_similarity=similarity)
            scores = {"rrf": rrf, "fusion_prior": prior, "context": context, "positive_cosine": pos_sim,
                      "negative_cosine": neg_sim, "mood_fit": mood_fit, "feedback_influence": strength,
                      "before_diversity": score, "final": score}
            ranked.append(RankedTrack(t, self.metadata[t], score, vector, scores, evidence))
        result, artist_counts = [], Counter()
        while ranked and len(result) < topk:
            def adjusted(track):
                count = max((artist_counts[a] for a in self.artists[track.track_id]), default=0)
                return track.score - self.config.diversity_penalty * count
            best = max(range(len(ranked)), key=lambda i: adjusted(ranked[i]))
            track = ranked.pop(best)
            track.score = adjusted(track)
            track.scores["final"] = track.score
            if track.track_id in allowed and not feedback.excludes(track.metadata):
                result.append(track)
                artist_counts.update(self.artists[track.track_id])
        return FeedbackRetrievalResult(mode, result, original_pool, pool,
                                       tuple(t for t in pool if t not in original_set), query_used,
                                       tuple(seeds), tuple(t for t in feedback.votes if t not in seeds))
