"""Opt-in feedback-aware retrieval; no changes to the existing chat pipeline.

Search the full available catalog embedding matrix with the Rocchio-adjusted
query, fuse those candidates with BM25 for the original text, then rerank.
Reuse the existing FeedbackState, RankedTrack and RocchioFeedback contracts.

Run a separate three-mode comparison with cached catalog/models:
    python -m musiccrs.feedback_retrieval --query "rock music" --like TRACK_ID

All modes use the same filters and fusion weights. They are a controlled new
experiment, not a replay of the older benchmark's rank-prior scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Collection

import numpy as np

from .relevance_feedback import FeedbackState, RankedTrack, RocchioFeedback, name_key, unit_vector, values


MODES = ("baseline", "rerank_only", "feedback_retrieval")


@dataclass
class FeedbackRetrievalResult:
    """Request-local results and evidence of candidate expansion; no shared votes."""

    mode: str
    tracks: list[RankedTrack]
    original_candidate_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    new_candidate_ids: tuple[str, ...]
    query_changed: bool
    usable_feedback_ids: tuple[str, ...]
    unavailable_feedback_ids: tuple[str, ...]


class FeedbackAwareRetriever:
    """Independent adapter over an existing catalog, BM25 and dense retriever.

    ``dense`` must expose aligned ``track_ids``, ``track_embeddings`` and
    ``encode_query(text)`` in the same embedding space. DenseRetriever already
    provides these. No models are loaded here and no source arrays are mutated.

    ``allowed_ids`` carries structured genre/playlist constraints from a caller;
    ``excluded_ids`` can contain playlist seeds and previously displayed songs.
    Persistent track/artist exclusions and dislikes come from FeedbackState.
    Liked tracks remain embedding seeds but are withheld by default.
    """

    def __init__(self, catalog, lexical, dense, *, candidate_k=50, rrf_k=60,
                 lexical_weight=1.0, dense_weight=0.25, rocchio=None):
        if type(candidate_k) is not int or candidate_k <= 0:
            raise ValueError("candidate_k must be a positive integer")
        if type(rrf_k) is not int or rrf_k <= 0:
            raise ValueError("rrf_k must be a positive integer")
        if any(not math.isfinite(w) or w < 0 for w in (lexical_weight, dense_weight)):
            raise ValueError("Fusion weights must be finite and nonnegative")
        if lexical_weight + dense_weight == 0:
            raise ValueError("At least one fusion weight must be positive")
        self.metadata = catalog.metadata_dict
        self.lexical, self.dense = lexical, dense
        self.candidate_k, self.rrf_k = candidate_k, rrf_k
        self.lexical_weight, self.dense_weight = lexical_weight, dense_weight
        self.rocchio = rocchio if rocchio is not None else RocchioFeedback()

        ids = list(dense.track_ids)
        matrix = np.asarray(dense.track_embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(ids) or matrix.shape[1] == 0:
            raise ValueError("Dense track IDs and embedding matrix must be aligned")
        if len(ids) != len(set(ids)):
            raise ValueError("Dense track IDs must be unique")
        if any(track_id != row.get("track_id") for track_id, row in self.metadata.items()):
            raise ValueError("Catalog keys must match metadata track identities")
        self.dimension = matrix.shape[1]
        with np.errstate(over="ignore", invalid="ignore"):
            norms = np.linalg.norm(matrix, axis=1)
        valid = (np.isfinite(matrix).all(axis=1) & np.isfinite(norms) & (norms > 0)
                 & np.array([t in self.metadata for t in ids], dtype=bool))
        self.track_ids = tuple(t for t, keep in zip(ids, valid) if keep)
        # The real DenseRetriever matrix is already normalized. Share it when
        # possible; injected/raw matrices are normalized into a separate array.
        if valid.all() and np.allclose(norms, 1, atol=1e-6, rtol=1e-6):
            self.matrix = matrix
        else:
            self.matrix = matrix[valid] / norms[valid, None]
        self.positions = {t: i for i, t in enumerate(self.track_ids)}
        self.artists = {t: {name_key(a) for a in values(row.get("artist_name"))}
                        for t, row in self.metadata.items()}

    def _semantic(self, vector, eligible, depth):
        """Score every usable catalog row before choosing allowed top candidates."""
        scores = self.matrix @ np.asarray(vector, dtype=np.float32)
        positions = np.flatnonzero(eligible)
        # Stable full sorting is inexpensive at this catalog size and resolves
        # boundary ties deterministically in catalog embedding order.
        selected = positions[np.argsort(-scores[positions], kind="stable")[:depth]]
        return [self.track_ids[i] for i in selected], scores

    def _lexical(self, query, allowed, depth):
        """Overfetch if exclusions consume the leading BM25 candidates."""
        index_ids = getattr(self.lexical, "track_ids", None)
        size = len(index_ids) if index_ids is not None else len(self.metadata)
        fetch = min(depth, size)
        while fetch:
            raw = self.lexical.text_to_item_retrieval(query, fetch)
            result = list(dict.fromkeys(t for t in raw if t in allowed))
            if len(result) >= depth or fetch == size:
                return result[:depth]
            fetch = min(size, fetch * 2)
        return []

    def retrieve(self, query: str, feedback: FeedbackState, *, topk=5,
                 mode="feedback_retrieval", excluded_ids: Collection[str] = (),
                 allowed_ids: Collection[str] | None = None, hide_liked=True) -> FeedbackRetrievalResult:
        """Retrieve from the catalog, not just the previous recommendation pool.

        baseline: original-query BM25 + original-query semantic fusion.
        rerank_only: that same candidate pool, followed by existing Rocchio reranking.
        feedback_retrieval: original-query BM25 + adjusted-query semantic fusion,
            followed by that same reranker, anchored to the ORIGINAL query.

        The caller retains FeedbackState across turns. This object never owns or
        mutates user feedback. Unsupported/missing seed vectors are reported.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if type(topk) is not int or topk < 0:
            raise ValueError("topk must be a nonnegative integer")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("A nonempty original query is required")
        if topk == 0:
            return FeedbackRetrievalResult(mode, [], (), (), (), False, (), ())
        depth = max(self.candidate_k, topk)
        excluded = set(excluded_ids) | set(feedback.excluded_track_ids)
        excluded.update(t for t, vote in feedback.votes.items() if vote == -1 or (hide_liked and vote == 1))
        blocked_artists = {name_key(a) for a in feedback.excluded_artists | feedback.disliked_artists}
        requested = set(self.metadata) if allowed_ids is None else set(allowed_ids)
        allowed = {t for t in requested if t in self.metadata and t not in excluded
                   and not (self.artists[t] & blocked_artists)}
        eligible = np.array([t in allowed for t in self.track_ids], dtype=bool)
        original = unit_vector(self.dense.encode_query(query), self.dimension)
        if original is None:
            raise ValueError("Encoded query must be finite, nonzero and in the catalog embedding space")
        seeds = {t: self.matrix[self.positions[t]] for t in feedback.votes if t in self.positions}
        updated = self.rocchio.updated_query(original, feedback, seeds)
        changed = bool(seeds) and not np.allclose(original, updated, atol=1e-7, rtol=1e-7)
        lexical = self._lexical(query, allowed, depth) if self.lexical_weight else []
        original_semantic, original_scores = self._semantic(original, eligible, depth)
        query_used = mode == "feedback_retrieval" and changed and self.dense_weight > 0
        if query_used:
            semantic, semantic_scores = self._semantic(updated, eligible, depth)
        else:
            semantic, semantic_scores = original_semantic, original_scores
        if self.dense_weight == 0:
            semantic, original_semantic = [], []
        original_pool = tuple(dict.fromkeys(lexical + original_semantic))
        original_set = set(original_pool)
        pool = tuple(dict.fromkeys(lexical + semantic))
        novel = tuple(t for t in pool if t not in original_set)
        lexical_ranks = {t: rank for rank, t in enumerate(lexical, 1)}
        semantic_ranks = {t: rank for rank, t in enumerate(semantic, 1)}
        original_ranks = {t: rank for rank, t in enumerate(original_semantic, 1)}
        normalizer = (self.lexical_weight + self.dense_weight) / (self.rrf_k + 1)
        candidates, vectors = [], dict(seeds)
        for track_id in pool:
            lexical_rank, semantic_rank = lexical_ranks.get(track_id), semantic_ranks.get(track_id)
            rrf = ((self.lexical_weight / (self.rrf_k + lexical_rank) if lexical_rank else 0)
                   + (self.dense_weight / (self.rrf_k + semantic_rank) if semantic_rank else 0))
            position = self.positions.get(track_id)
            if position is not None:
                vectors[track_id] = self.matrix[position]
            evidence = {"retrieval_sources": [name for name, rank in
                        (("bm25", lexical_rank), ("semantic", semantic_rank)) if rank],
                        "lexical_rank": lexical_rank, "semantic_rank": semantic_rank,
                        "original_semantic_rank": original_ranks.get(track_id),
                        "new_to_original_pool": track_id not in original_set,
                        "feedback_query_used_for_retrieval": query_used,
                        "feedback_applied": False}
            scores = {"rrf": rrf, "fusion_prior": rrf / normalizer}
            if position is not None:
                scores.update(original_semantic_cosine=float(original_scores[position]),
                              retrieval_semantic_cosine=float(semantic_scores[position]))
            candidates.append(RankedTrack(track_id, self.metadata[track_id], rrf / normalizer,
                                          vectors.get(track_id), scores, evidence))
        if mode == "baseline":
            ranked = sorted(candidates, key=lambda t: t.score, reverse=True)
        else:
            # Pass the original anchor: feeding updated into this reranker would
            # apply the Rocchio query update twice.
            ranked = self.rocchio.rerank(candidates, feedback, vectors, original, excluded_ids=excluded)
        final = [t for t in ranked if t.track_id in allowed and not feedback.excludes(t.metadata)][:topk]
        return FeedbackRetrievalResult(mode, final, original_pool, pool, novel,
                                       query_used,
                                       tuple(seeds), tuple(t for t in feedback.votes if t not in seeds))


def main():
    """Separate executable demo; creates no changes to the running application."""
    import argparse
    import json
    from pathlib import Path
    from time import perf_counter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--like", action="append", default=[], help="Catalog track ID; repeatable")
    parser.add_argument("--dislike", action="append", default=[], help="Catalog track ID; repeatable")
    parser.add_argument("--exclude-track", action="append", default=[])
    parser.add_argument("--exclude-artist", action="append", default=[])
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, help="Write to a NEW JSON file; existing files are refused")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.candidate_k <= 0 or args.topk <= 0:
        parser.error("--candidate-k and --topk must be positive")
    if args.output and args.output.exists():
        parser.error("--output already exists; choose a new file to preserve previous results")
    from retrieval.data_loader import MusicCatalogLoader
    from retrieval.bm25 import BM25Retriever
    from retrieval.dense import DenseRetriever
    from .recommendation_explanations import explain_recommendation
    import torch

    torch.set_num_threads(args.threads)
    catalog = MusicCatalogLoader()
    unknown = (set(args.like) | set(args.dislike) | set(args.exclude_track)) - catalog.metadata_dict.keys()
    if unknown:
        parser.error("Unknown catalog IDs: " + ", ".join(sorted(unknown)))
    if set(args.like) & set(args.dislike):
        parser.error("A track cannot be both --like and --dislike in one invocation")
    feedback = FeedbackState(excluded_track_ids=set(args.exclude_track),
                             excluded_artists={name_key(a) for a in args.exclude_artist})
    for track_id in args.like:
        feedback.rate(track_id, True)
    for track_id in args.dislike:
        feedback.rate(track_id, False)
    lexical = BM25Retriever(catalog=catalog, corpus_types=["track_name", "artist_name", "album_name", "tag_list"])
    dense = DenseRetriever()
    retriever = FeedbackAwareRetriever(catalog, lexical, dense, candidate_k=args.candidate_k)
    dense.encode_query("music")  # model warmup is outside the observed request time
    comparisons = {}
    for mode in MODES:
        started = perf_counter()
        result = retriever.retrieve(args.query, feedback, mode=mode, topk=args.topk)
        records = []
        for track in result.tracks:
            reason, facts = explain_recommendation(track.metadata, args.query, feedback=feedback,
                                                  evidence=track.evidence, catalog=catalog.metadata_dict)
            records.append({"track_id": track.track_id, "metadata": track.metadata,
                            "score": track.score, "scores": track.scores, "evidence": track.evidence,
                            "explanation": reason, "explanation_facts": facts})
        comparisons[mode] = {"request_ms": (perf_counter() - started) * 1000,
                             "query_changed": result.query_changed,
                             "original_candidate_ids": result.original_candidate_ids,
                             "candidate_ids": result.candidate_ids, "new_candidate_ids": result.new_candidate_ids,
                             "usable_feedback_ids": result.usable_feedback_ids,
                             "unavailable_feedback_ids": result.unavailable_feedback_ids, "tracks": records}
    document = {"query": args.query, "likes": args.like, "dislikes": args.dislike,
                "excluded_tracks": args.exclude_track, "excluded_artists": args.exclude_artist,
                "candidate_k_per_branch": args.candidate_k, "catalog_size": len(catalog.metadata_dict),
                "searchable_embedding_count": len(retriever.track_ids),
                "weights": {"lexical": 1.0, "semantic": 0.25, "rrf_k": 60,
                            "alpha": 1.0, "beta": 0.75, "gamma": 0.25, "feedback": 0.75},
                "note": "Single warm requests; candidate-expansion demonstration, not a judged quality evaluation.",
                "comparisons": comparisons}
    if args.output:
        with args.output.open("x") as stream:
            stream.write(json.dumps(document, indent=2) + "\n")
    print(json.dumps({mode: {"new_candidates": len(r["new_candidate_ids"]),
                            "returned": [{"title": t["metadata"]["track_name"],
                                          "artist": t["metadata"]["artist_name"],
                                          "new_to_original_pool": t["evidence"]["new_to_original_pool"]}
                                         for t in r["tracks"]]}
                      for mode, r in comparisons.items()}, indent=2))


if __name__ == "__main__":
    main()
