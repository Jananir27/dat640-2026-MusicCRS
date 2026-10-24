"""Run Hybrid + BPR personalized retrieval for MusicCRS.

Pipeline
--------
Conversation
    |
    +--> BM25 + Tags
    |
    +--> Qwen Dense
             |
             v
        Weighted RRF
             |
          Top-N candidates
             |
             v
      BPR personalization
       (warm users only)
             |
             v
          Final Top-K

Cold users do not have BPR embeddings, so their hybrid ranking is
kept unchanged.

Recommended initial configuration:
    candidate_k = 50
    topk = 20
    dense_weight = 0.25
    rrf_k = 60
    bpr_weight = 0.01
"""

import argparse
import json
import time

import numpy as np
from datasets import load_dataset

from .data_loader import MusicCatalogLoader
from .hybrid import HybridRetriever
from .run_bm25_baseline import _build_retrieval_input


NUM_TURNS = 8

CORPUS_TYPES = [
    "track_name",
    "artist_name",
    "album_name",
    "tag_list",
]

DEFAULT_DIALOGUE_DATASET = (
    "talkpl-ai/TalkPlayData-Challenge-Dataset"
)

DEFAULT_USER_EMBEDDINGS = (
    "talkpl-ai/TalkPlayData-Challenge-User-Embeddings"
)

DEFAULT_TRACK_EMBEDDINGS = (
    "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"
)


# ============================================================
# BPR data
# ============================================================

def load_user_bpr_embeddings(
    dataset_name: str,
) -> dict[str, np.ndarray]:
    """Load normalized BPR vectors for warm test users."""

    print()
    print("Loading user BPR embeddings...")

    dataset = load_dataset(dataset_name)

    embeddings = {}

    for row in dataset["test_warm"]:

        vector = row["cf-bpr"]

        if vector is None or len(vector) == 0:
            continue

        vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        norm = np.linalg.norm(vector)

        if norm == 0:
            continue

        embeddings[row["user_id"]] = (
            vector / norm
        )

    print(
        f"Loaded {len(embeddings)} "
        "warm-user BPR embeddings."
    )

    return embeddings


def load_track_bpr_embeddings(
    dataset_name: str,
) -> dict[str, np.ndarray]:
    """Load normalized BPR vectors for tracks."""

    print()
    print("Loading track BPR embeddings...")

    dataset = load_dataset(
        dataset_name,
        split="all_tracks",
    )

    embeddings = {}

    skipped = 0

    for row in dataset:

        vector = row["cf-bpr"]

        if vector is None or len(vector) == 0:
            skipped += 1
            continue

        vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        norm = np.linalg.norm(vector)

        if norm == 0:
            skipped += 1
            continue

        embeddings[row["track_id"]] = (
            vector / norm
        )

    print(
        f"Loaded {len(embeddings)} "
        "track BPR embeddings."
    )

    if skipped:
        print(
            f"Skipped {skipped} tracks "
            "without usable BPR embeddings."
        )

    return embeddings


# ============================================================
# BPR reranking
# ============================================================

def rerank_with_bpr(
    track_ids: list[str],
    user_vector: np.ndarray,
    track_embeddings: dict[str, np.ndarray],
    bpr_weight: float = 0.01,
    rrf_k: int = 60,
    topk: int = 20,
) -> list[str]:
    """Rerank hybrid candidates using BPR similarity.

    Hybrid candidate rank remains the primary relevance signal.

    The BPR similarity is normalized within the candidate set.

    final_score =
        hybrid_rank_score
        +
        bpr_weight * normalized_bpr_score
    """

    if not track_ids:
        return []

    similarities = []

    # --------------------------------------------------------
    # User-track collaborative similarity
    # --------------------------------------------------------

    for track_id in track_ids:

        track_vector = track_embeddings.get(
            track_id
        )

        if track_vector is None:
            similarity = 0.0
        else:
            # Both vectors already normalized.
            similarity = float(
                np.dot(
                    user_vector,
                    track_vector,
                )
            )

        similarities.append(
            similarity
        )

    similarities = np.asarray(
        similarities,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Normalize BPR similarity inside candidate set
    # --------------------------------------------------------

    minimum = float(
        similarities.min()
    )

    maximum = float(
        similarities.max()
    )

    if maximum > minimum:

        normalized_bpr = (
            similarities - minimum
        ) / (
            maximum - minimum
        )

    else:

        normalized_bpr = np.zeros_like(
            similarities
        )

    # --------------------------------------------------------
    # Hybrid rank + BPR
    # --------------------------------------------------------

    scored_tracks = []

    for index, track_id in enumerate(
        track_ids
    ):

        hybrid_rank = index + 1

        hybrid_score = (
            1.0
            / (
                rrf_k
                + hybrid_rank
            )
        )

        bpr_score = (
            bpr_weight
            * float(
                normalized_bpr[index]
            )
        )

        final_score = (
            hybrid_score
            + bpr_score
        )

        scored_tracks.append(
            (
                track_id,
                final_score,
                hybrid_rank,
            )
        )

    # Highest score first.
    # hybrid_rank is used as deterministic tie breaker.
    scored_tracks.sort(
        key=lambda item: (
            -item[1],
            item[2],
        )
    )

    return [
        track_id
        for track_id, _, _ in
        scored_tracks[:topk]
    ]


# ============================================================
# Main retrieval
# ============================================================

def run_hybrid_bpr(
    retriever: HybridRetriever,
    catalog: MusicCatalogLoader,
    dataset,
    user_embeddings: dict[str, np.ndarray],
    track_embeddings: dict[str, np.ndarray],
    candidate_k: int,
    topk: int,
    bpr_weight: float,
    rrf_k: int,
) -> list[dict]:
    """Run hybrid retrieval followed by BPR personalization."""

    print()
    print("=" * 70)
    print("BUILDING RETRIEVAL QUERIES")
    print("=" * 70)

    queries = []
    query_metadata = []

    # --------------------------------------------------------
    # Build all queries first
    # --------------------------------------------------------

    for session_index, item in enumerate(
        dataset
    ):

        for turn_number in range(
            1,
            NUM_TURNS + 1,
        ):

            query = _build_retrieval_input(
                conversations=item[
                    "conversations"
                ],
                target_turn_number=(
                    turn_number
                ),
                catalog=catalog,
                corpus_types=CORPUS_TYPES,
            )

            queries.append(query)

            query_metadata.append(
                {
                    "session_id":
                        item["session_id"],

                    "user_id":
                        item["user_id"],

                    "turn_number":
                        turn_number,
                }
            )

        if (
            (session_index + 1) % 100 == 0
            or session_index + 1
            == len(dataset)
        ):

            print(
                f"Built queries for "
                f"{session_index + 1}/"
                f"{len(dataset)} sessions"
            )

    print()
    print(
        f"Total queries: {len(queries)}"
    )

    # --------------------------------------------------------
    # Hybrid candidate generation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING HYBRID RETRIEVAL")
    print("=" * 70)

    print(
        f"Retrieving top {candidate_k} "
        "hybrid candidates per query..."
    )

    retrieval_start = time.time()

    hybrid_candidates = (
        retriever.batch_text_to_item_retrieval(
            queries,
            candidate_k,
        )
    )

    retrieval_elapsed = (
        time.time()
        - retrieval_start
    )

    print()
    print(
        f"Hybrid retrieval completed in "
        f"{retrieval_elapsed / 60:.2f} minutes"
    )

    # --------------------------------------------------------
    # BPR personalization
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("APPLYING BPR PERSONALIZATION")
    print("=" * 70)

    predictions = []

    personalized_turns = 0
    cold_turns = 0

    for info, candidates in zip(
        query_metadata,
        hybrid_candidates,
    ):

        user_id = info["user_id"]

        user_vector = (
            user_embeddings.get(
                user_id
            )
        )

        # ----------------------------------------------------
        # Warm user
        # ----------------------------------------------------

        if user_vector is not None:

            final_tracks = (
                rerank_with_bpr(
                    track_ids=candidates,
                    user_vector=user_vector,
                    track_embeddings=(
                        track_embeddings
                    ),
                    bpr_weight=(
                        bpr_weight
                    ),
                    rrf_k=rrf_k,
                    topk=topk,
                )
            )

            personalized_turns += 1

        # ----------------------------------------------------
        # Cold user
        # ----------------------------------------------------

        else:

            final_tracks = (
                candidates[:topk]
            )

            cold_turns += 1

        predictions.append(
            {
                "session_id":
                    info["session_id"],

                "turn_number":
                    info["turn_number"],

                "predicted_track_ids":
                    final_tracks,

                "predicted_response":
                    "",
            }
        )

    print(
        f"Personalized turns : "
        f"{personalized_turns}"
    )

    print(
        f"Cold/fallback turns: "
        f"{cold_turns}"
    )

    return predictions


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run MusicCRS Hybrid + "
            "BPR personalized retrieval"
        )
    )

    parser.add_argument(
        "--dialogue_dataset",
        default=DEFAULT_DIALOGUE_DATASET,
    )

    parser.add_argument(
        "--user_embeddings",
        default=DEFAULT_USER_EMBEDDINGS,
    )

    parser.add_argument(
        "--track_embeddings",
        default=DEFAULT_TRACK_EMBEDDINGS,
    )

    parser.add_argument(
        "--split",
        default="test",
    )

    parser.add_argument(
        "--max_sessions",
        type=int,
        default=None,
        help=(
            "Optional number of sessions "
            "for quick experiments"
        ),
    )

    parser.add_argument(
        "--candidate_k",
        type=int,
        default=50,
        help=(
            "Number of hybrid candidates "
            "before BPR reranking"
        ),
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help=(
            "Final number of tracks "
            "after BPR reranking"
        ),
    )

    parser.add_argument(
        "--dense_weight",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--rrf_k",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--bpr_weight",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    if args.candidate_k < args.topk:
        raise ValueError(
            "--candidate_k must be >= --topk"
        )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MusicCRS HYBRID + BPR RETRIEVAL")
    print("=" * 70)

    print(
        f"Split          : {args.split}"
    )

    print(
        f"Sessions       : "
        f"{args.max_sessions or 'ALL'}"
    )

    print(
        f"Candidate-k    : "
        f"{args.candidate_k}"
    )

    print(
        f"Final top-k    : "
        f"{args.topk}"
    )

    print(
        "BM25 weight    : 1.0"
    )

    print(
        f"Dense weight   : "
        f"{args.dense_weight}"
    )

    print(
        f"RRF k          : "
        f"{args.rrf_k}"
    )

    print(
        f"BPR weight     : "
        f"{args.bpr_weight}"
    )

    print(
        f"Output         : "
        f"{args.output}"
    )

    # --------------------------------------------------------
    # Dialogue dataset
    # --------------------------------------------------------

    print()
    print("Loading dialogue dataset...")

    dataset = load_dataset(
        args.dialogue_dataset,
        split=args.split,
    )

    if args.max_sessions is not None:

        number_sessions = min(
            args.max_sessions,
            len(dataset),
        )

        dataset = dataset.select(
            range(number_sessions)
        )

    print(
        f"Loaded {len(dataset)} sessions."
    )

    # --------------------------------------------------------
    # Catalog
    # --------------------------------------------------------

    print()
    print("Loading music catalog...")

    catalog = MusicCatalogLoader()

    # --------------------------------------------------------
    # User BPR
    # --------------------------------------------------------

    user_embeddings = (
        load_user_bpr_embeddings(
            args.user_embeddings
        )
    )

    # --------------------------------------------------------
    # Track BPR
    # --------------------------------------------------------

    track_embeddings = (
        load_track_bpr_embeddings(
            args.track_embeddings
        )
    )

    # --------------------------------------------------------
    # Hybrid retriever
    # --------------------------------------------------------

    print()
    print("Initializing HybridRetriever...")

    retriever = HybridRetriever(
        candidate_k=args.candidate_k,
        dense_weight=args.dense_weight,
        rrf_k=args.rrf_k,
    )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    total_start = time.time()

    predictions = run_hybrid_bpr(
        retriever=retriever,
        catalog=catalog,
        dataset=dataset,
        user_embeddings=user_embeddings,
        track_embeddings=track_embeddings,
        candidate_k=args.candidate_k,
        topk=args.topk,
        bpr_weight=args.bpr_weight,
        rrf_k=args.rrf_k,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print()
    print("Saving predictions...")

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            predictions,
            f,
            indent=2,
        )

    elapsed = (
        time.time()
        - total_start
    )

    print()
    print("=" * 70)
    print("HYBRID + BPR COMPLETED")
    print("=" * 70)

    print(
        f"Predictions : "
        f"{len(predictions)}"
    )

    print(
        f"Saved to    : "
        f"{args.output}"
    )

    print(
        f"Runtime     : "
        f"{elapsed / 60:.2f} minutes"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
