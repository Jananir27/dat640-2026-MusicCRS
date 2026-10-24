"""Online adapter for the final G1 ranking pipeline.

The exported artifact freezes the tuned weights and development recommendation
frequencies. No tuning, ground-truth access, or BM25-only fallback occurs here.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from threading import Lock

from .base import RetrievalModule
from .exact_match import ExactMetadataMatcher
from .tune_diversity_reranking import rerank_prediction
from .tune_exact_by_specificity import rerank_tracks
from .tune_prediction_fusion import weighted_rrf
from .tune_selective_expansion import selective_rerank


DEFAULT_ARTIFACT = Path(__file__).resolve().parents[1] / "cache/final_retrieval.json"
SPECIFICITIES = ("HH", "HL", "LH", "LL")
SELECTIVE_KEYS = ("protect_top", "novelty_weight", "original_bonus", "reservoir_start", "reservoir_end")


def validate_artifact(data: dict) -> dict:
    """Reject missing/invalid configuration rather than substitute defaults."""
    def integer(value, minimum, maximum=None):
        return (type(value) is int and value >= minimum
                and (maximum is None or value <= maximum))

    def weight(value):
        return (type(value) in (int, float) and math.isfinite(value) and value >= 0)

    def require(condition):
        if not condition:
            raise ValueError("Invalid artifact value")

    try:
        require(data["schema_version"] == 1)
        for section in ("fusion_weights", "exact_weights"):
            require(all(weight(data[section][spec]) for spec in SPECIFICITIES))
        hybrid = data["hybrid"]
        require(integer(hybrid["candidate_k"], 20))
        require(hybrid["reservoir_k"] == 100)
        require(integer(hybrid["rrf_k"], 1))
        require(integer(hybrid["fusion_rrf_k"], 1))
        require(weight(hybrid["dense_weight"]) and hybrid["dense_weight"] > 0)
        diversity = data["diversity"]
        require(integer(diversity["protect_top"], 0, 20))
        require(weight(diversity["novelty_weight"]))
        selective = data["selective"]
        require(integer(selective["protect_top"], 0, 20))
        require(integer(selective["reservoir_start"], 1, 100))
        require(integer(selective["reservoir_end"], selective["reservoir_start"], 100))
        require(weight(selective["novelty_weight"]))
        require(weight(selective["original_bonus"]))
        for section in ("diversity_statistics", "selective_statistics"):
            stats = data[section]
            require(integer(stats["total_turns"], 1))
            require(isinstance(stats["frequency"], dict) and stats["frequency"])
            require(all(isinstance(key, str) and integer(count, 1, stats["total_turns"])
                        for key, count in stats["frequency"].items()))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid final retrieval artifact; rebuild it with retrieval.export_final.") from exc
    return data


def load_artifact(path: str | Path | None = None) -> dict:
    artifact_path = Path(path or os.environ.get("MUSICCRS_RETRIEVAL_ARTIFACT", DEFAULT_ARTIFACT))
    try:
        with artifact_path.open(encoding="utf-8") as handle:
            return validate_artifact(json.load(handle))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Final G1 retrieval artifact is missing: {artifact_path}. "
            "Run python -m retrieval.export_final --artifacts-dir /path/to/G1/results "
            "and set MUSICCRS_RETRIEVAL_ARTIFACT to the exported file."
        ) from exc


class FinalRetriever(RetrievalModule):
    """Adaptive/goal hybrid fusion, exact matching, diversity, selective expansion."""

    def __init__(self, catalog, artifact=None, *, bm25_retriever=None, dense_retriever=None):
        # Validate before any embedding model is downloaded or initialized.
        self.artifact = validate_artifact(artifact) if artifact is not None else load_artifact()
        self.catalog = catalog
        self.matcher = ExactMetadataMatcher(catalog=catalog)
        self._lock = Lock()
        from .hybrid import HybridRetriever
        from .bm25 import BM25Retriever

        lexical = bm25_retriever if bm25_retriever is not None else BM25Retriever(
            catalog=catalog, corpus_types=["track_name", "artist_name", "album_name", "tag_list"],
        )
        config = self.artifact["hybrid"]
        self.hybrid = HybridRetriever(
            bm25_retriever=lexical, dense_retriever=dense_retriever,
            candidate_k=config["candidate_k"], rrf_k=config["rrf_k"],
            dense_weight=config["dense_weight"],
        )

    def _hybrid_ranking(self, query: str, candidate_k: int, topk: int) -> list[str]:
        # Each branch retains its G1 candidate depth (50 vs 100). Taking the
        # first 20 of the top-100 fusion would change the original base ranking.
        lexical_k = min(candidate_k, len(self.catalog.metadata_dict))
        dense_k = min(candidate_k, len(self.hybrid.dense.track_ids))
        lexical = self.hybrid.bm25.text_to_item_retrieval(query, lexical_k) if lexical_k else []
        semantic = self.hybrid.dense.text_to_item_retrieval(query, dense_k) if dense_k else []
        return self.hybrid._fuse(lexical, semantic, topk)

    def rerank(self, adaptive, goal, reservoir, request: str, specificity: str) -> list[str]:
        """Apply the saved post-retrieval stages; useful for artifact replay."""
        config = self.artifact
        tracks = weighted_rrf(
            adaptive, goal, goal_weight=config["fusion_weights"][specificity],
            rrf_k=config["hybrid"]["fusion_rrf_k"], topk=20,
        )
        exact_scores = {track_id: self.matcher.score_track(request, track_id) for track_id in tracks}
        tracks = rerank_tracks(tracks, exact_scores, config["exact_weights"][specificity])
        stats = config["diversity_statistics"]
        tracks = rerank_prediction(
            tracks, stats["frequency"], stats["total_turns"], **config["diversity"],
        )
        stats = config["selective_statistics"]
        return selective_rerank(
            tracks, reservoir, stats["frequency"], stats["total_turns"], **config["selective"],
        )

    def retrieve_context(self, item: dict, turn_number: int, topk: int = 20) -> list[str]:
        """Use G1 conversation dictionaries; missing specificity defaults to LL."""
        if topk <= 0:
            return []
        from .run_adaptive_goal_hybrid import build_adaptive_query, get_current_user_message
        from .run_goal_aware_hybrid import build_goal_aware_query

        request = get_current_user_message(item["conversations"], turn_number)
        if not request.strip():
            return []
        specificity = (item.get("conversation_goal") or {}).get("specificity", "LL")
        if specificity not in SPECIFICITIES:
            specificity = "LL"
        adaptive_query = build_adaptive_query(item, turn_number, self.catalog)
        goal_query = build_goal_aware_query(item, turn_number, self.catalog)
        config = self.artifact["hybrid"]
        with self._lock:
            adaptive = self._hybrid_ranking(adaptive_query, config["candidate_k"], 20)
            goal = self._hybrid_ranking(goal_query, config["candidate_k"], 20)
            reservoir = self._hybrid_ranking(adaptive_query, config["reservoir_k"], 100)
            ranked = self.rerank(adaptive, goal, reservoir, request, specificity)
        # The final G1 result has at most 20 unique, catalog-valid tracks.
        return list(dict.fromkeys(t for t in ranked if t in self.catalog.metadata_dict))[:min(topk, 20)]

    def encode_query(self, query: str):
        """Use the same dense embedding space as candidate tracks (Part 4 R2)."""
        with self._lock:
            encoder = getattr(self.hybrid.dense, "encode_query", None)
            return encoder(query) if encoder else None

    def retrieve_query(self, query: str, topk: int = 20) -> list[str]:
        """Part 4 bridge for a direct discovery query without dialogue labels.

        Retain the saved G1 fusion/reranking stages. Indexing scaffolding such
        as 'CURRENT REQUEST' is useful in long challenge conversations but can
        dominate a short genre query with literal tracks titled 'Request'.
        """
        if not query.strip() or topk <= 0:
            return []
        config = self.artifact["hybrid"]
        with self._lock:
            ranked = self._hybrid_ranking(query, config["candidate_k"], 20)
            reservoir = self._hybrid_ranking(query, config["reservoir_k"], 100)
            result = self.rerank(ranked, ranked, reservoir, query, "LL")
        return list(dict.fromkeys(t for t in result if t in self.catalog.metadata_dict))[:min(topk, 20)]

    def get_track_embeddings(self, track_ids: list[str]):
        lookup = getattr(self.hybrid.dense, "get_track_embeddings", None)
        return lookup(track_ids) if lookup else {}

    def text_to_item_retrieval(self, query: str, topk: int) -> list[str]:
        return self.retrieve_context({
            "conversations": [{"role": "user", "content": query, "turn_number": 1}],
            "conversation_goal": {"specificity": "LL"},
        }, 1, topk)

    def batch_text_to_item_retrieval(self, queries: list[str], topk: int) -> list[list[str]]:
        return [self.text_to_item_retrieval(query, topk) for query in queries]
