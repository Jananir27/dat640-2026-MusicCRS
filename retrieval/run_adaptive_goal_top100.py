"""Generate top-100 candidates using the existing adaptive goal-aware pipeline.

Purpose:
    Produce a larger candidate pool for later diversity-aware reranking.

Output:
    predictions_adaptive_goal_top100_full.json

This reuses the existing adaptive query construction and HybridRetriever.
"""

import argparse
import json
import time

from datasets import load_dataset

from .data_loader import MusicCatalogLoader
from .hybrid import HybridRetriever
from .run_adaptive_goal_hybrid import build_adaptive_query


DEFAULT_DATASET = (
    "talkpl-ai/"
    "TalkPlayData-Challenge-Dataset"
)

NUM_TURNS = 8


# ============================================================
# Build queries
# ============================================================

def build_queries(dataset, catalog):

    queries = []
    metadata = []

    specificity_counts = {
        "HH": 0,
        "HL": 0,
        "LH": 0,
        "LL": 0,
    }

    for item in dataset:

        goal = (
            item.get("conversation_goal")
            or {}
        )

        specificity = (
            goal.get("specificity")
            or "LL"
        )

        if specificity not in specificity_counts:
            specificity = "LL"

        for turn_number in range(
            1,
            NUM_TURNS + 1,
        ):

            query = build_adaptive_query(
                item=item,
                target_turn_number=turn_number,
                catalog=catalog,
            )

            queries.append(
                query
            )

            metadata.append(
                {
                    "session_id":
                        item["session_id"],

                    "turn_number":
                        turn_number,

                    "specificity":
                        specificity,
                }
            )

            specificity_counts[
                specificity
            ] += 1

    return (
        queries,
        metadata,
        specificity_counts,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate adaptive goal-aware "
            "top-100 candidates"
        )
    )

    parser.add_argument(
        "--dialogue_dataset",
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--split",
        default="test",
    )

    parser.add_argument(
        "--max_sessions",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--candidate_k",
        type=int,
        default=100,
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
        "--output",
        default=(
            "predictions_adaptive_goal_top100_full.json"
        ),
    )

    args = parser.parse_args()

    if args.candidate_k < args.topk:

        raise ValueError(
            "candidate_k must be >= topk"
        )

    print()
    print("=" * 85)
    print(
        "MusicCRS ADAPTIVE GOAL-AWARE "
        "TOP-100 CANDIDATE GENERATION"
    )
    print("=" * 85)

    print(
        f"Split        : {args.split}"
    )

    print(
        f"Sessions     : "
        f"{args.max_sessions or 'ALL'}"
    )

    print(
        f"Top-k        : {args.topk}"
    )

    print(
        f"Candidate-k  : "
        f"{args.candidate_k}"
    )

    print(
        f"Dense weight : "
        f"{args.dense_weight}"
    )

    print(
        f"RRF k        : {args.rrf_k}"
    )

    print(
        f"Output       : {args.output}"
    )

    # ========================================================
    # Load dataset
    # ========================================================

    print()
    print("Loading dialogue dataset...")

    dataset = load_dataset(
        args.dialogue_dataset,
        split=args.split,
    )

    if args.max_sessions is not None:

        count = min(
            args.max_sessions,
            len(dataset),
        )

        dataset = dataset.select(
            range(count)
        )

    print(
        f"Loaded {len(dataset)} sessions."
    )

    # ========================================================
    # Catalog
    # ========================================================

    print()
    print("Loading music catalog...")

    catalog = MusicCatalogLoader()

    # ========================================================
    # Build adaptive queries
    # ========================================================

    print()
    print(
        "Building adaptive queries..."
    )

    (
        queries,
        metadata,
        specificity_counts,
    ) = build_queries(
        dataset,
        catalog,
    )

    print(
        f"Queries generated: "
        f"{len(queries)}"
    )

    print()
    print("Turns by specificity:")

    for spec in [
        "HH",
        "HL",
        "LH",
        "LL",
    ]:

        print(
            f"  {spec}: "
            f"{specificity_counts[spec]}"
        )

    # ========================================================
    # Retriever
    # ========================================================

    print()
    print("=" * 85)
    print("INITIALIZING RETRIEVER")
    print("=" * 85)

    retriever = HybridRetriever(
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
        dense_weight=args.dense_weight,
    )

    # ========================================================
    # Retrieve
    # ========================================================

    print()
    print("=" * 85)
    print(
        "GENERATING TOP-100 CANDIDATES"
    )
    print("=" * 85)

    start_time = time.time()

    rankings = (
        retriever.batch_text_to_item_retrieval(
            queries,
            args.topk,
        )
    )

    runtime = (
        time.time()
        - start_time
    )

    # ========================================================
    # Build predictions
    # ========================================================

    predictions = []

    for info, tracks in zip(
        metadata,
        rankings,
    ):

        predictions.append(
            {
                "session_id":
                    info["session_id"],

                "turn_number":
                    info["turn_number"],

                "predicted_track_ids":
                    tracks,

                "predicted_response":
                    "",
            }
        )

    # ========================================================
    # Validation
    # ========================================================

    print()
    print("Validating output...")

    lengths = [
        len(
            prediction[
                "predicted_track_ids"
            ]
        )
        for prediction in predictions
    ]

    if lengths:

        print(
            f"Minimum candidates: "
            f"{min(lengths)}"
        )

        print(
            f"Maximum candidates: "
            f"{max(lengths)}"
        )

        print(
            f"Average candidates: "
            f"{sum(lengths) / len(lengths):.2f}"
        )

    # ========================================================
    # Save
    # ========================================================

    print()
    print("Saving candidate file...")

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

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 85)
    print(
        "TOP-100 CANDIDATE GENERATION COMPLETED"
    )
    print("=" * 85)

    print(
        f"Sessions     : "
        f"{len(dataset)}"
    )

    print(
        f"Turns        : "
        f"{len(predictions)}"
    )

    print(
        f"Candidates   : "
        f"{args.topk} per turn"
    )

    print(
        f"Output       : "
        f"{args.output}"
    )

    print(
        f"Runtime      : "
        f"{runtime / 60:.2f} minutes"
    )

    print("=" * 85)


if __name__ == "__main__":
    main()
