"""R1 preference extraction/retrieval followed by the shared R2 feedback stage."""
from __future__ import annotations

from time import perf_counter
import re
import unicodedata
from typing import Any

from .feedback_workflow import remember_r1_exclusions
from .music_features import classify_tags


def _norm(value: Any) -> str:
    plain = unicodedata.normalize("NFKD", str(value)).casefold()
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w]+", " ", plain).split())


def _values(value: Any) -> list[str]:
    return [str(x) for x in value] if isinstance(value, (list, tuple, set)) else ([] if value is None else [str(value)])


def _artist_matches(artist_field: Any, forbidden: set[str]) -> bool:
    # Also handle compound credits such as 'Linkin Park feat. Jay-Z'.
    for artist in _values(artist_field):
        name = _norm(artist)
        for banned in forbidden:
            if banned and (name == banned or f" {banned} " in f" {name} "):
                return True
    return False


def _track_matches(title_field: Any, forbidden: set[str]) -> bool:
    # Exact metadata title comparisons, ignoring punctuation and case.
    return any(_norm(title) in forbidden for title in _values(title_field))


def recommend_with_llm(request: str, parser: Any, service: Any, state: Any,
                       *, limit: int = 5, candidate_limit: int = 20) -> dict:
    """Return track metadata after one LLM call and deterministic exclusions.

    The source retriever is MusicService.recommend_tracks(), which is the
    project's existing final-retrieval pipeline (not a new retrieval model).
    """
    if limit <= 0 or candidate_limit < limit:
        raise ValueError("candidate_limit must be >= limit > 0")

    t0 = perf_counter()
    prefs = parser.parse(request)
    t1 = perf_counter()

    query_parts = list(prefs["genre"]) + list(prefs["mood"])
    query_parts += [f"music similar to {artist}" for artist in prefs["preferred_artists"]]
    if not query_parts:
        query_parts = [request]
    # Do not inject excluded names into the retrieval query.
    retrieval_query = " ".join(query_parts)

    playlist_ids = set(getattr(state, "playlist_ids", []))
    shared_feedback = hasattr(service, "rerank_feedback") and hasattr(service, "catalog")
    if shared_feedback:
        remember_r1_exclusions(service, state, prefs, request)
    candidates = service.recommend_tracks(
        retrieval_query,
        exclude_ids=playlist_ids,
        limit=candidate_limit,
        state=state,
        request=retrieval_query,
        use_playlist_context=bool(prefs["use_playlist_context"] and playlist_ids),
    )
    t2 = perf_counter()

    excluded_artists = {_norm(x) for x in prefs["excluded_artists"]}
    excluded_tracks = {_norm(x) for x in prefs["excluded_tracks"]}
    playlist_artists: set[str] = set()
    if prefs["avoid_playlist_artists"]:
        for track_id in playlist_ids:
            seed = service.get_track(track_id)
            if seed:
                playlist_artists.update(_norm(x) for x in _values(seed.get("artist_name")))

    recommendations = []
    for track in candidates:
        track_id = track.get("track_id")
        if track_id in playlist_ids:
            continue
        if not shared_feedback and _artist_matches(track.get("artist_name"), excluded_artists | playlist_artists):
            continue
        if not shared_feedback and _track_matches(track.get("track_name"), excluded_tracks):
            continue
        recommendations.append(track)
    if shared_feedback:
        genres, _ = classify_tags(prefs["genre"])
        _, moods = classify_tags(prefs["mood"])
        recommendations = service.rerank_feedback(
            recommendations, state, retrieval_query, limit, genres=genres, moods=moods)
    else:
        recommendations = recommendations[:limit]
    t3 = perf_counter()
    return {
        "preferences": prefs,
        "recommendations": recommendations,
        "candidate_count": len(candidates),
        "timing_seconds": {
            "llm": round(t1 - t0, 3),
            "retrieval": round(t2 - t1, 3),
            "filtering": round(t3 - t2, 3),
            "total": round(t3 - t0, 3),
        },
    }

