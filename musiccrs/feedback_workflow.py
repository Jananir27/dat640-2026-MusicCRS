"""R2 integration over R1 indexes; user state stays in ConversationState.

The no-feedback path preserves R1's candidate order. With preferences and full
indexes, use the existing adaptive scorer to retrieve beyond that pool. Small
injected providers can still use the model-independent Rocchio reranker.
"""
from __future__ import annotations

from dataclasses import replace
from threading import Lock
from types import SimpleNamespace

from .adaptive_retrieval import AdaptiveMusicRetriever, metadata_features
from .relevance_feedback import RankedTrack, RocchioFeedback, artist_keys, name_key
from .request_constraints import ConstraintError as FeedbackReferenceError, resolve_preference_exclusions


class FeedbackRankingError(RuntimeError):
    """The feedback provider failed; do not substitute unfiltered results."""


def remember_r1_exclusions(service, state, preferences, request=""):
    """Resolve LLM constraints against catalog identities before storing them.

    Validate all track references first. Titles shared by different artists need
    an artist; a qualified reference excludes all exact title/artist duplicates.
    This also prevents full-catalog feedback expansion undoing R1 filtering.
    """
    resolved = resolve_preference_exclusions(preferences, service.catalog.metadata_dict, request, state.playlist_ids)
    state.feedback.excluded_artists.update(resolved.artists)
    state.feedback.excluded_track_ids.update(resolved.track_ids)
    return resolved


class FeedbackRecommendationAdapter:
    """One final feedback stage shared by standard and LLM recommendations."""

    def __init__(self, service):
        self.service = service
        self._index = None
        self._artist_keys = None
        self._index_lock = Lock()

    def _adaptive_index(self, retriever, provider):
        hybrid = getattr(retriever, "hybrid", retriever)
        lexical = getattr(hybrid, "bm25", None)
        dense = provider if self.service._embedding_provider is not None else getattr(hybrid, "dense", provider)
        if lexical is None or not all(hasattr(dense, key) for key in
                                      ("track_ids", "track_embeddings", "encode_query")):
            return None
        with self._index_lock:
            if self._index is None:
                # FinalRetriever protects its query encoder with a lock. Keep
                # that protection while sharing its aligned immutable matrix.
                encode = (retriever.encode_query if self.service._embedding_provider is None
                          and hasattr(retriever, "encode_query") else dense.encode_query)
                view = SimpleNamespace(track_ids=dense.track_ids, track_embeddings=dense.track_embeddings,
                                       encode_query=encode)
                self._index = AdaptiveMusicRetriever(
                    self.service.catalog, lexical, view,
                    candidate_k=getattr(hybrid, "candidate_k", 50),
                    rrf_k=getattr(hybrid, "rrf_k", 60),
                    dense_weight=getattr(hybrid, "dense_weight", 0.25),
                )
                self._artist_keys = {t: artist_keys(row) for t, row in self.service.catalog.metadata_dict.items()}
        return self._index

    def rank(self, tracks, state, query, limit=5, *, allowed_ids=None, genres=None, moods=None,
             exclude_playlist=True):
        if type(limit) is not int or limit < 0:
            raise ValueError("Recommendation limit must be a nonnegative integer")
        if limit == 0:
            return []
        catalog = self.service.catalog.metadata_dict
        excluded = (set(state.playlist_ids) if exclude_playlist else set()) | state.skip_recommendation_ids
        excluded.update(t for t, vote in state.feedback.votes.items() if vote == 1)
        try:
            preferences = bool(any(t in catalog for t in state.feedback.votes) or state.feedback.liked_artists)
            retriever = self.service._get_retriever() if preferences else None
            provider = self.service._embedding_provider or retriever
            index = self._adaptive_index(retriever, provider) if preferences else None
            # Pool reranking only needs the supplied rows. Full-catalog features
            # and artist identities are computed once when creating the index.
            requested = (set(catalog) if index is not None else
                         {t.track_id if isinstance(t, RankedTrack) else t["track_id"] for t in tracks} & catalog.keys())
            allowed = requested if allowed_ids is None else requested & set(allowed_ids)
            if genres:
                allowed = {t for t in allowed if (index.features[t][0] if index is not None
                                                  else metadata_features(catalog[t])[0]) & set(genres)}
            if index is not None:
                blocked = {name_key(a) for a in state.feedback.excluded_artists | state.feedback.disliked_artists}
                hard = set(state.feedback.excluded_track_ids) | excluded
                hard.update(t for t, vote in state.feedback.votes.items() if vote == -1)
                allowed = {t for t in allowed if t not in hard and not self._artist_keys[t] & blocked}
            else:
                allowed = {t for t in allowed if not state.feedback.excludes(catalog[t])}
            if index is not None:
                result = index.retrieve(query, state.feedback, topk=limit, excluded_ids=excluded,
                                        allowed_ids=allowed, genres=genres, moods=moods)
                original_ids = {t.track_id if isinstance(t, RankedTrack) else t["track_id"] for t in tracks}
                return [replace(t, scores={**t.scores, "feedback_delta": t.evidence.get("updated_cosine", 0)
                                           - t.evidence.get("original_cosine", 0)},
                                evidence={**t.evidence, "feedback_path": "adaptive_catalog",
                                             "outside_r1_pool": t.track_id not in original_ids,
                                             "usable_feedback_ids": list(result.usable_feedback_ids),
                                             "unavailable_feedback_ids": list(result.unavailable_feedback_ids)})
                        for t in result.tracks if t.track_id in allowed and t.track_id not in excluded
                        and not state.feedback.excludes(t.metadata)]

            candidates = []
            for rank, track in enumerate(tracks):
                t = track.track_id if isinstance(track, RankedTrack) else track["track_id"]
                if t not in allowed:
                    continue
                if isinstance(track, RankedTrack):
                    if track.metadata.get("track_id") != t:
                        raise ValueError("Candidate identity does not match metadata")
                    candidate = replace(track, metadata=catalog[t])
                else:
                    prior = 1 - rank / max(1, len(tracks))
                    candidate = RankedTrack(t, catalog[t], prior, scores={"rank_prior": prior})
                candidates.append(candidate)
            embeddings, vector = {}, None
            if state.feedback.votes and hasattr(provider, "get_track_embeddings"):
                ids = list(dict.fromkeys([t.track_id for t in candidates] + list(state.feedback.votes)))
                embeddings = provider.get_track_embeddings(ids)
                if hasattr(provider, "encode_query"):
                    vector = provider.encode_query(query)
            ranked = RocchioFeedback().rerank(candidates, state.feedback, embeddings, vector,
                                              excluded_ids=excluded)
            return [replace(t, evidence={**t.evidence, "feedback_path": "r1_pool_rocchio"})
                    for t in ranked[:limit]]
        except Exception as exc:
            raise FeedbackRankingError("Feedback ranking is unavailable") from exc

    def rerank(self, tracks, state, query, limit=5, **constraints):
        state.ranking_evidence = {}
        ranked = self.rank(tracks, state, query, limit, **constraints)
        state.ranking_evidence = {t.track_id: {**t.evidence, "scores": t.scores} for t in ranked}
        return [t.metadata for t in ranked]
