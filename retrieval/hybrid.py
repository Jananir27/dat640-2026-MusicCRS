"""Hybrid BM25 + dense retrieval using weighted Reciprocal Rank Fusion."""

from .base import RetrievalModule
from .bm25 import BM25Retriever
from .dense import DenseRetriever


class HybridRetriever(RetrievalModule):
    """Combine BM25 and dense retrieval using weighted RRF."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever | None = None,
        dense_retriever: DenseRetriever | None = None,
        candidate_k: int = 50,
        rrf_k: int = 60,
        dense_weight: float = 0.25,
    ) -> None:

        print("Initializing hybrid retriever...")

        # IMPORTANT:
        # BM25Retriever must use the tag-enriched configuration.
        self.bm25 = (
            bm25_retriever
            if bm25_retriever is not None
            else BM25Retriever(
                corpus_types=[
                    "track_name",
                    "artist_name",
                    "album_name",
                    "tag_list",
                ]
            )
        )

        self.dense = (
            dense_retriever
            if dense_retriever is not None
            else DenseRetriever()
        )

        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight

        print("Hybrid configuration:")
        print(f"  candidate_k : {candidate_k}")
        print(f"  rrf_k       : {rrf_k}")
        print(f"  BM25 weight : 1.0")
        print(f"  Dense weight: {dense_weight}")

    def _fuse(
        self,
        bm25_results: list[str],
        dense_results: list[str],
        topk: int,
    ) -> list[str]:
        """Fuse BM25 and dense rankings with weighted RRF."""

        scores = {}

        # -----------------------------------------
        # BM25 ranking
        # -----------------------------------------

        for rank, track_id in enumerate(
            bm25_results,
            start=1,
        ):
            scores[track_id] = (
                scores.get(track_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

        # -----------------------------------------
        # Dense ranking
        # -----------------------------------------

        for rank, track_id in enumerate(
            dense_results,
            start=1,
        ):
            scores[track_id] = (
                scores.get(track_id, 0.0)
                + self.dense_weight
                / (self.rrf_k + rank)
            )

        # -----------------------------------------
        # Sort combined ranking
        # -----------------------------------------

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            track_id
            for track_id, _ in ranked[:topk]
        ]

    def text_to_item_retrieval(
        self,
        query: str,
        topk: int,
    ) -> list[str]:
        """Retrieve tracks for one query."""

        candidate_k = max(
            self.candidate_k,
            topk,
        )

        bm25_results = (
            self.bm25.text_to_item_retrieval(
                query,
                candidate_k,
            )
        )

        dense_results = (
            self.dense.text_to_item_retrieval(
                query,
                candidate_k,
            )
        )

        return self._fuse(
            bm25_results,
            dense_results,
            topk,
        )

    def batch_text_to_item_retrieval(
        self,
        queries: list[str],
        topk: int,
    ) -> list[list[str]]:
        """Retrieve tracks for multiple queries."""

        candidate_k = max(
            self.candidate_k,
            topk,
        )

        print("Running BM25 retrieval...")

        bm25_results = (
            self.bm25.batch_text_to_item_retrieval(
                queries,
                candidate_k,
            )
        )

        print("Running dense retrieval...")

        dense_results = (
            self.dense.batch_text_to_item_retrieval(
                queries,
                candidate_k,
            )
        )

        print("Fusing BM25 + dense rankings...")

        predictions = []

        for bm25_ranked, dense_ranked in zip(
            bm25_results,
            dense_results,
        ):
            predictions.append(
                self._fuse(
                    bm25_ranked,
                    dense_ranked,
                    topk,
                )
            )

        return predictions
