"""Run specificity-aware query optimized hybrid retrieval.

Uses:
    - query_optimizer.py
    - HybridRetriever
    - conversation_goal
    - specificity-aware query construction

Pipeline:
    Dialogue
        -> specificity-aware query optimization
        -> BM25 + Dense retrieval
        -> weighted RRF
        -> top-k predictions

Example:
    python -m retrieval.run_query_optimized_hybrid \
        --split test \
        --max_sessions 100 \
        --topk 20 \
        --candidate_k 50 \
        --dense_weight 0.25 \
        --rrf_k 60 \
        --output predictions_query_optimized_test100.json
"""

import argparse
import json
import time

from datasets import load_dataset

from .hybrid import HybridRetriever
from .query_optimizer import optimize_query


NUM_TURNS = 8

DEFAULT_DATASET = (
    "talkpl-ai/"
    "TalkPlayData-Challenge-Dataset"
)


# ============================================================
# Build optimized queries
# ============================================================

def build_queries(
    dataset,
):
    """Build optimized retrieval query for every session/turn."""

    queries = []
    metadata = []

    specificity_counts = {
        "HH": 0,
        "HL": 0,
        "LH": 0,
        "LL": 0,
    }

    for item in dataset:

        conversation_goal = (
            item.get("conversation_goal")
            or {}
        )

        specificity = (
            conversation_goal.get(
                "specificity",
                "LL",
            )
        )

        if specificity not in specificity_counts:
            specificity = "LL"

        for turn_number in range(
            1,
            NUM_TURNS + 1,
        ):

            query = optimize_query(
                conversations=item[
                    "conversations"
                ],
                target_turn_number=turn_number,
                specificity=specificity,
                conversation_goal=conversation_goal,
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
            "Run specificity-aware "
            "query optimized hybrid retrieval"
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
        default=20,
    )

    parser.add_argument(
        "--candidate_k",
        type=int,
        default=50,
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
        required=True,
    )

    args = parser.parse_args()

    print()
    print("=" * 75)
    print(
        "MusicCRS QUERY-OPTIMIZED "
        "HYBRID RETRIEVAL"
    )
    print("=" * 75)

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
    # Dataset
    # ========================================================

    print()
    print("Loading dialogue dataset...")

    dataset = load_dataset(
        args.dialogue_dataset,
        split=args.split,
    )

    if args.max_sessions is not None:

        max_sessions = min(
            args.max_sessions,
            len(dataset),
        )

        dataset = dataset.select(
            range(max_sessions)
        )

    print(
        f"Loaded {len(dataset)} sessions."
    )

    # ========================================================
    # Queries
    # ========================================================

    print()
    print(
        "Building optimized queries..."
    )

    (
        queries,
        metadata,
        specificity_counts,
    ) = build_queries(
        dataset
    )

    print(
        f"Queries generated: "
        f"{len(queries)}"
    )

    print()
    print(
        "Turns by specificity:"
    )

    for specificity in [
        "HH",
        "HL",
        "LH",
        "LL",
    ]:

        print(
            f"  {specificity}: "
            f"{specificity_counts[specificity]}"
        )

    # ========================================================
    # Example queries
    # ========================================================

    print()
    print("=" * 75)
    print("QUERY EXAMPLES")
    print("=" * 75)

    shown = set()

    for query, meta in zip(
        queries,
        metadata,
    ):

        specificity = meta[
            "specificity"
        ]

        if specificity in shown:
            continue

        print()
        print(
            f"[{specificity}] "
            f"{query}"
        )

        shown.add(
            specificity
        )

        if len(shown) == 4:
            break

    # ========================================================
    # Retriever
    # ========================================================

    print()
    print("=" * 75)
    print("INITIALIZING RETRIEVER")
    print("=" * 75)

    retriever = HybridRetriever(
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
        dense_weight=args.dense_weight,
    )

    # ========================================================
    # Retrieval
    # ========================================================

    print()
    print("=" * 75)
    print("RUNNING RETRIEVAL")
    print("=" * 75)

    start_time = time.time()

    ranked_results = (
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
    # Predictions
    # ========================================================

    predictions = []

    for meta, tracks in zip(
        metadata,
        ranked_results,
    ):

        predictions.append(
            {
                "session_id":
                    meta["session_id"],

                "turn_number":
                    meta["turn_number"],

                "predicted_track_ids":
                    tracks,

                "predicted_response":
                    "",
            }
        )

    # ========================================================
    # Save
    # ========================================================

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

    # ========================================================
    # Final report
    # ========================================================

    print()
    print("=" * 75)
    print(
        "QUERY-OPTIMIZED HYBRID COMPLETED"
    )
    print("=" * 75)

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
        f"{runtime / 60:.2f} minutes"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()
