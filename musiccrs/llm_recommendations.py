"""R1 preference extraction/retrieval followed by the shared R2 feedback stage."""
from __future__ import annotations

from time import perf_counter
from typing import Any

from .feedback_workflow import remember_r1_exclusions
from .music_features import classify_tags
from .relevance_feedback import artist_keys
from .request_constraints import resolve_preference_exclusions, split_request_exclusions


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
        positive_request, _ = split_request_exclusions(request)
        query_parts = [positive_request]
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

    if not shared_feedback:
        catalog = getattr(getattr(service, "catalog", None), "metadata_dict", None)
        partial_catalog = catalog is None
        if partial_catalog:
            catalog = {track["track_id"]: track for track in candidates}
            for tid in playlist_ids:
                seed = service.get_track(tid)
                if seed:
                    catalog[tid] = seed
        exclusions = resolve_preference_exclusions(prefs, catalog, request, playlist_ids,
                                                    allow_missing=partial_catalog)

    recommendations = []
    for track in candidates:
        track_id = track.get("track_id")
        if track_id in playlist_ids:
            continue
        if not shared_feedback and (track_id in exclusions.track_ids
                                    or artist_keys(track) & exclusions.artists):
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

