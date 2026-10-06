"""Dense text-to-track retrieval using Qwen3 embeddings.

Uses precomputed Qwen3-Embedding-0.6B metadata embeddings for tracks
and encodes conversation queries with the same embedding model.

Automatically uses CUDA GPU when available and falls back to CPU.
Tracks are ranked using cosine similarity.
"""

import numpy as np
import torch

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from .base import RetrievalModule


DEFAULT_EMBEDDING_DATASET = (
    "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"
)

DEFAULT_EMBEDDING_FIELD = "metadata-qwen3_embedding_0.6b"

DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

EMBEDDING_DIM = 1024


class DenseRetriever(RetrievalModule):
    """Dense retriever using Qwen3 text embeddings."""

    def __init__(
        self,
        dataset_name: str = DEFAULT_EMBEDDING_DATASET,
        split: str = "all_tracks",
        embedding_field: str = DEFAULT_EMBEDDING_FIELD,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:

        # ---------------------------------------------------------
        # Device selection
        # ---------------------------------------------------------

        if torch.cuda.is_available():
            self.device = "cuda"

            print("CUDA GPU detected")
            print("GPU:", torch.cuda.get_device_name(0))

            gpu_memory = (
                torch.cuda.get_device_properties(0).total_memory
                / 1024**3
            )

            print(f"GPU memory: {gpu_memory:.2f} GB")

        else:
            self.device = "cpu"

            print("CUDA GPU not available")
            print("Using CPU")

        # ---------------------------------------------------------
        # Load precomputed track embeddings
        # ---------------------------------------------------------

        print("Loading track embeddings...")

        dataset = load_dataset(
            dataset_name,
            split=split,
        )

        track_ids = []
        track_embeddings = []

        invalid_count = 0

        for item in dataset:

            embedding = item[embedding_field]

            if (
                embedding is None
                or len(embedding) != EMBEDDING_DIM
            ):
                invalid_count += 1
                continue

            track_ids.append(item["track_id"])
            track_embeddings.append(embedding)

        self.track_ids = track_ids

        self.track_embeddings = np.stack(
            track_embeddings
        ).astype(np.float32)

        print(
            f"Loaded {len(self.track_ids)} valid "
            f"track embeddings"
        )

        if invalid_count:
            print(
                f"Skipped {invalid_count} "
                f"invalid/missing embeddings"
            )

        print(
            "Track embedding matrix:",
            self.track_embeddings.shape,
        )

        # ---------------------------------------------------------
        # Normalize track embeddings
        # ---------------------------------------------------------

        print("Normalizing track embeddings...")

        norms = np.linalg.norm(
            self.track_embeddings,
            axis=1,
            keepdims=True,
        )

        norms[norms == 0] = 1.0

        self.track_embeddings = (
            self.track_embeddings / norms
        )

        # ---------------------------------------------------------
        # Load Qwen model
        # ---------------------------------------------------------

        print("Loading Qwen3 embedding model...")

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
        )

        print(
            f"Qwen model loaded on: {self.device}"
        )

    # -------------------------------------------------------------
    # Query encoding
    # -------------------------------------------------------------
    def _encode_queries(
            self, queries: list[str],
            ) -> np.ndarray:
        """Encode conversation queries using Qwen3."""
        if self.device == "cuda":
            batch_size = 2
        else:
            batch_size = 8

        # Prevent very long conversation histories from exhausting GPU memory.
        self.model.max_seq_length = 2048
        print(
            f"Encoding {len(queries)} queries "
            f"on {self.device} "
            f"(batch_size={batch_size}, "
            f"max_seq_length={self.model.max_seq_length})"
        )
        with torch.inference_mode():
            embeddings = self.model.encode(
                    queries,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=True,
            )
        return embeddings.astype(np.float32) 

	# Single-query retrieval
    def text_to_item_retrieval(
            self,
            query: str,
            topk: int,
            ) -> list[str]:
        """Retrieve top-k tracks for a single query."""

        query_embedding = self._encode_queries([query])[0]

        scores = self.track_embeddings @ query_embedding

        top_indices = np.argpartition(
                scores,
                -topk,
                )[-topk:]
        top_indices = top_indices[
                np.argsort(scores[top_indices])[::-1]
                ]
        return [
                self.track_ids[i]
                for i in top_indices
                ]
    # -------------------------------------------------------------
    # Batch retrieval
    # -------------------------------------------------------------

    def batch_text_to_item_retrieval(
        self,
        queries: list[str],
        topk: int,
    ) -> list[list[str]]:

        query_embeddings = self._encode_queries(
            queries
        )

        predictions = []

        print("Calculating track similarities...")

        for index, query_embedding in enumerate(
            query_embeddings
        ):

            scores = (
                self.track_embeddings
                @ query_embedding
            )

            top_indices = np.argpartition(
                scores,
                -topk,
            )[-topk:]

            top_indices = top_indices[
                np.argsort(
                    scores[top_indices]
                )[::-1]
            ]

            predictions.append(
                [
                    self.track_ids[i]
                    for i in top_indices
                ]
            )

            if (index + 1) % 500 == 0:
                print(
                    f"Retrieved {index + 1}/"
                    f"{len(query_embeddings)} queries"
                )

        return predictions
