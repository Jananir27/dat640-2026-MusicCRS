"""Tune BPR personalization on top of existing hybrid predictions.

This script does NOT run BM25, Qwen, or HybridRetriever again.

It uses:
    - Existing hybrid predictions
    - Dialogue session -> user_id mapping
    - User CF-BPR embeddings
    - Track CF-BPR embeddings

Warm users:
    Hybrid candidates are reranked using collaborative-filtering similarity.

Cold users:
    Original hybrid ranking is retained because their BPR embedding is empty.

The script evaluates several BPR weights and reports the best configuration.
"""

import argparse
import json

import numpy as np
from datasets import load_dataset

from .evaluation.evaluate import evaluate


# ---------------------------------------------------------
# BPR weights to test
# ---------------------------------------------------------

BPR_WEIGHTS = [
    0.00,
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.30,
]


# ---------------------------------------------------------
# Load user BPR embeddings
# ---------------------------------------------------------

def load_user_embeddings(
    dataset_name: str,
) -> dict[str, np.ndarray]:
    """Load valid warm-user BPR embeddings."""

    print()
    print("Loading user BPR embeddings...")

    dataset = load_dataset(dataset_name)

    user_embeddings = {}

    # Warm users have valid CF-BPR vectors.
    for row in dataset["test_warm"]:

        embedding = row["cf-bpr"]

        if embedding is None or len(embedding) == 0:
            continue

        vector = np.asarray(
            embedding,
            dtype=np.float32,
        )

        # Normalize once so cosine similarity becomes dot product.
        norm = np.linalg.norm(vector)

        if norm > 0:
            vector = vector / norm

        user_embeddings[row["user_id"]] = vector

    print(
        f"Loaded {len(user_embeddings)} "
        "warm-user embeddings"
    )

    return user_embeddings


# ---------------------------------------------------------
# Load track BPR embeddings
# ---------------------------------------------------------

def load_track_embeddings(
    dataset_name: str,
) -> dict[str, np.ndarray]:
    """Load and normalize track BPR embeddings."""

    print()
    print("Loading track BPR embeddings...")

    dataset = load_dataset(
        dataset_name,
        split="all_tracks",
    )

    track_embeddings = {}

    skipped = 0

    for row in dataset:

        embedding = row["cf-bpr"]

        if embedding is None or len(embedding) == 0:
            skipped += 1
            continue

        vector = np.asarray(
            embedding,
            dtype=np.float32,
        )

        norm = np.linalg.norm(vector)

        if norm > 0:
            vector = vector / norm

        track_embeddings[row["track_id"]] = vector

    print(
        f"Loaded {len(track_embeddings)} "
        "track embeddings"
    )

    if skipped:
        print(
            f"Skipped {skipped} tracks "
            "without BPR embeddings"
        )

    return track_embeddings


# ---------------------------------------------------------
# Session -> user mapping
# ---------------------------------------------------------

def load_session_users(
    dataset_name: str,
    split: str,
) -> dict[str, str]:
    """Map session IDs to user IDs."""

    print()
    print("Loading session -> user mapping...")

    dataset = load_dataset(
        dataset_name,
        split=split,
    )

    session_users = {
        row["session_id"]: row["user_id"]
        for row in dataset
    }

    print(
        f"Loaded {len(session_users)} "
        "session-user mappings"
    )

    return session_users


# ---------------------------------------------------------
# BPR similarity
# ---------------------------------------------------------

def get_bpr_similarity(
    user_vector: np.ndarray,
    track_id: str,
    track_embeddings: dict[str, np.ndarray],
) -> float:
    """Compute cosine similarity between user and track BPR vectors."""

    track_vector = track_embeddings.get(track_id)

    if track_vector is None:
        return 0.0

    # Both vectors are already normalized.
    return float(
        np.dot(
            user_vector,
            track_vector,
        )
    )


# ---------------------------------------------------------
# Rerank one prediction
# ---------------------------------------------------------

def rerank_tracks(
    track_ids: list[str],
    user_vector: np.ndarray,
    track_embeddings: dict[str, np.ndarray],
    bpr_weight: float,
    rrf_k: int,
) -> list[str]:
    """Rerank existing hybrid candidates using BPR similarity.

    Hybrid ranking remains the primary signal.

    Score:

        hybrid_score =
            1 / (rrf_k + hybrid_rank)

        final_score =
            hybrid_score
            + bpr_weight * scaled_BPR_similarity

    BPR similarities are min-max normalized within the candidate set
    before being added to the hybrid ranking score.
    """

    if not track_ids:
        return track_ids

    # -----------------------------------------------------
    # Calculate BPR similarity for each candidate
    # -----------------------------------------------------

    similarities = []

    for track_id in track_ids:

        similarity = get_bpr_similarity(
            user_vector=user_vector,
            track_id=track_id,
            track_embeddings=track_embeddings,
        )

        similarities.append(similarity)

    similarities = np.asarray(
        similarities,
        dtype=np.float32,
    )

    # -----------------------------------------------------
    # Normalize BPR scores inside this candidate set
    # -----------------------------------------------------

    minimum = float(similarities.min())
    maximum = float(similarities.max())

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

    # -----------------------------------------------------
    # Combine hybrid rank + BPR
    # -----------------------------------------------------

    scored_tracks = []

    for index, track_id in enumerate(track_ids):

        rank = index + 1

        hybrid_score = (
            1.0 / (rrf_k + rank)
        )

        personalization_score = (
            bpr_weight
            * float(normalized_bpr[index])
        )

        final_score = (
            hybrid_score
            + personalization_score
        )

        scored_tracks.append(
            (
                track_id,
                final_score,
            )
        )

    scored_tracks.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        track_id
        for track_id, _ in scored_tracks
    ]


# ---------------------------------------------------------
# Build personalized predictions
# ---------------------------------------------------------

def build_predictions(
    hybrid_predictions: list[dict],
    session_users: dict[str, str],
    user_embeddings: dict[str, np.ndarray],
    track_embeddings: dict[str, np.ndarray],
    bpr_weight: float,
    rrf_k: int,
) -> tuple[list[dict], int, int]:
    """Rerank hybrid predictions for warm users."""

    predictions = []

    personalized_turns = 0
    fallback_turns = 0

    for prediction in hybrid_predictions:

        session_id = prediction["session_id"]

        user_id = session_users.get(
            session_id
        )

        user_vector = (
            user_embeddings.get(user_id)
            if user_id is not None
            else None
        )

        original_tracks = prediction[
            "predicted_track_ids"
        ]

        # -------------------------------------------------
        # Warm user
        # -------------------------------------------------

        if (
            user_vector is not None
            and bpr_weight > 0
        ):

            ranked_tracks = rerank_tracks(
                track_ids=original_tracks,
                user_vector=user_vector,
                track_embeddings=track_embeddings,
                bpr_weight=bpr_weight,
                rrf_k=rrf_k,
            )

            personalized_turns += 1

        # -------------------------------------------------
        # Cold user or weight=0
        # -------------------------------------------------

        else:

            ranked_tracks = list(
                original_tracks
            )

            fallback_turns += 1

        predictions.append(
            {
                "session_id":
                    session_id,

                "turn_number":
                    prediction["turn_number"],

                "predicted_track_ids":
                    ranked_tracks,

                "predicted_response":
                    prediction.get(
                        "predicted_response",
                        "",
                    ),
            }
        )

    return (
        predictions,
        personalized_turns,
        fallback_turns,
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Tune BPR personalization "
            "on MusicCRS hybrid predictions"
        )
    )

    parser.add_argument(
        "--predictions",
        default="predictions_hybrid_full.json",
        help="Existing hybrid predictions",
    )

    parser.add_argument(
        "--ground_truth",
        default="ground_truth.json",
    )

    parser.add_argument(
        "--dialogue_dataset",
        default=(
            "talkpl-ai/"
            "TalkPlayData-Challenge-Dataset"
        ),
    )

    parser.add_argument(
        "--user_embeddings",
        default=(
            "talkpl-ai/"
            "TalkPlayData-Challenge-User-Embeddings"
        ),
    )

    parser.add_argument(
        "--track_embeddings",
        default=(
            "talkpl-ai/"
            "TalkPlayData-Challenge-Track-Embeddings"
        ),
    )

    parser.add_argument(
        "--split",
        default="test",
    )

    parser.add_argument(
        "--catalog_size",
        type=int,
        default=47071,
    )

    parser.add_argument(
        "--rrf_k",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--output",
        default="results_bpr_tuning.json",
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    print()
    print("=" * 75)
    print("MusicCRS BPR PERSONALIZATION TUNING")
    print("=" * 75)

    print(
        f"Predictions : {args.predictions}"
    )

    print(
        f"Ground truth: {args.ground_truth}"
    )

    print(
        f"RRF k       : {args.rrf_k}"
    )

    # -----------------------------------------------------
    # Existing hybrid predictions
    # -----------------------------------------------------

    print()
    print("Loading existing hybrid predictions...")

    with open(
        args.predictions,
        "r",
        encoding="utf-8",
    ) as f:

        hybrid_predictions = json.load(f)

    print(
        f"Loaded {len(hybrid_predictions)} "
        "hybrid predictions"
    )

    # -----------------------------------------------------
    # Ground truth
    # -----------------------------------------------------

    print("Loading ground truth...")

    with open(
        args.ground_truth,
        "r",
        encoding="utf-8",
    ) as f:

        ground_truth = json.load(f)

    print(
        f"Loaded {len(ground_truth)} "
        "ground-truth records"
    )

    # -----------------------------------------------------
    # Session-user mapping
    # -----------------------------------------------------

    session_users = load_session_users(
        dataset_name=args.dialogue_dataset,
        split=args.split,
    )

    # -----------------------------------------------------
    # User embeddings
    # -----------------------------------------------------

    user_embeddings = load_user_embeddings(
        dataset_name=args.user_embeddings,
    )

    # -----------------------------------------------------
    # Track embeddings
    # -----------------------------------------------------

    track_embeddings = load_track_embeddings(
        dataset_name=args.track_embeddings,
    )

    # -----------------------------------------------------
    # Report warm/cold coverage
    # -----------------------------------------------------

    warm_sessions = 0
    cold_sessions = 0

    for session_id, user_id in (
        session_users.items()
    ):

        if user_id in user_embeddings:
            warm_sessions += 1
        else:
            cold_sessions += 1

    print()
    print("=" * 75)
    print("BPR COVERAGE")
    print("=" * 75)

    print(
        f"Warm sessions : {warm_sessions}"
    )

    print(
        f"Cold sessions : {cold_sessions}"
    )

    print(
        f"Warm coverage : "
        f"{warm_sessions / len(session_users):.2%}"
    )

    # -----------------------------------------------------
    # Tune BPR weight
    # -----------------------------------------------------

    tuning_results = []

    print()
    print("=" * 75)
    print("STARTING BPR WEIGHT TUNING")
    print("=" * 75)

    for bpr_weight in BPR_WEIGHTS:

        print()
        print("-" * 65)

        print(
            f"Testing BPR weight = "
            f"{bpr_weight:.2f}"
        )

        print("-" * 65)

        (
            predictions,
            personalized_turns,
            fallback_turns,
        ) = build_predictions(
            hybrid_predictions=hybrid_predictions,
            session_users=session_users,
            user_embeddings=user_embeddings,
            track_embeddings=track_embeddings,
            bpr_weight=bpr_weight,
            rrf_k=args.rrf_k,
        )

        # -------------------------------------------------
        # Evaluate
        # -------------------------------------------------

        results = evaluate(
            predictions=predictions,
            ground_truth=ground_truth,
            catalog_size=args.catalog_size,
        )

        experiment = {
            "bpr_weight":
                bpr_weight,

            "rrf_k":
                args.rrf_k,

            "personalized_turns":
                personalized_turns,

            "fallback_turns":
                fallback_turns,

            **results,
        }

        tuning_results.append(
            experiment
        )

        print(
            f"Personalized turns : "
            f"{personalized_turns}"
        )

        print(
            f"Fallback turns     : "
            f"{fallback_turns}"
        )

        print(
            f"nDCG@1             : "
            f"{results['ndcg@1']:.6f}"
        )

        print(
            f"nDCG@10            : "
            f"{results['ndcg@10']:.6f}"
        )

        print(
            f"nDCG@20            : "
            f"{results['ndcg@20']:.6f}"
        )

        print(
            f"Catalog diversity  : "
            f"{results['catalog_diversity']:.6f}"
        )

        print(
            f"Final score        : "
            f"{results['final_score']:.6f}"
        )

    # -----------------------------------------------------
    # Sort by final score
    # -----------------------------------------------------

    tuning_results.sort(
        key=lambda result: result[
            "final_score"
        ],
        reverse=True,
    )

    # -----------------------------------------------------
    # Save tuning results
    # -----------------------------------------------------

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            tuning_results,
            f,
            indent=2,
        )

    # -----------------------------------------------------
    # Print comparison table
    # -----------------------------------------------------

    print()
    print("=" * 100)
    print("BPR TUNING RESULTS")
    print("=" * 100)

    print(
        f"{'Weight':<10}"
        f"{'nDCG@1':<13}"
        f"{'nDCG@10':<13}"
        f"{'nDCG@20':<13}"
        f"{'Diversity':<13}"
        f"{'Final':<13}"
    )

    print("-" * 100)

    for result in tuning_results:

        print(
            f"{result['bpr_weight']:<10.2f}"
            f"{result['ndcg@1']:<13.6f}"
            f"{result['ndcg@10']:<13.6f}"
            f"{result['ndcg@20']:<13.6f}"
            f"{result['catalog_diversity']:<13.6f}"
            f"{result['final_score']:<13.6f}"
        )

    # -----------------------------------------------------
    # Best result
    # -----------------------------------------------------

    best = tuning_results[0]

    print()
    print("=" * 75)
    print("BEST BPR CONFIGURATION")
    print("=" * 75)

    print(
        f"BPR weight        : "
        f"{best['bpr_weight']}"
    )

    print(
        f"nDCG@1           : "
        f"{best['ndcg@1']:.6f}"
    )

    print(
        f"nDCG@10          : "
        f"{best['ndcg@10']:.6f}"
    )

    print(
        f"nDCG@20          : "
        f"{best['ndcg@20']:.6f}"
    )

    print(
        f"Catalog diversity: "
        f"{best['catalog_diversity']:.6f}"
    )

    print(
        f"FINAL SCORE      : "
        f"{best['final_score']:.6f}"
    )

    print()
    print(
        f"Results saved to : "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
