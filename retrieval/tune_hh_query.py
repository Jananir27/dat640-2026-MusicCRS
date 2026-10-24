"""Tune HH query construction for MusicCRS adaptive hybrid retrieval.

Only HH query construction is changed.

HL, LH and LL continue to use the existing successful
build_adaptive_query() implementation.

HH variants tested:
    baseline        current x3 + goal x1
    current4        current x4
    current4_goal1  current x4 + goal x1
    current2_goal1  current x2 + goal x1
    current3_goal2  current x3 + goal x2

Designed for a small test set first (e.g. 100 sessions).
"""

import argparse
import json
import time

from datasets import load_dataset

from .data_loader import MusicCatalogLoader
from .hybrid import HybridRetriever
from .run_adaptive_goal_hybrid import (
    build_adaptive_query,
    get_current_user_message,
    get_goal_text,
    repeat_text,
)
from .evaluation.evaluate import evaluate


NUM_TURNS = 8

DEFAULT_DATASET = (
    "talkpl-ai/"
    "TalkPlayData-Challenge-Dataset"
)


# ============================================================
# HH query variants
# ============================================================

def build_hh_variant(
    current_request: str,
    goal_text: str,
    variant: str,
) -> str:
    """Build one HH query variant."""

    parts = []

    # --------------------------------------------------------
    # Existing baseline
    # current x3 + goal x1
    # --------------------------------------------------------

    if variant == "baseline":

        parts.extend(
            repeat_text(
                "CURRENT REQUEST",
                current_request,
                3,
            )
        )

        if goal_text:
            parts.append(
                "GOAL:\n"
                + goal_text
            )

    # --------------------------------------------------------
    # Current request x4 only
    # --------------------------------------------------------

    elif variant == "current4":

        parts.extend(
            repeat_text(
                "CURRENT REQUEST",
                current_request,
                4,
            )
        )

    # --------------------------------------------------------
    # Current x4 + goal x1
    # --------------------------------------------------------

    elif variant == "current4_goal1":

        parts.extend(
            repeat_text(
                "CURRENT REQUEST",
                current_request,
                4,
            )
        )

        if goal_text:
            parts.append(
                "GOAL:\n"
                + goal_text
            )

    # --------------------------------------------------------
    # Current x2 + goal x1
    # --------------------------------------------------------

    elif variant == "current2_goal1":

        parts.extend(
            repeat_text(
                "CURRENT REQUEST",
                current_request,
                2,
            )
        )

        if goal_text:
            parts.append(
                "GOAL:\n"
                + goal_text
            )

    # --------------------------------------------------------
    # Current x3 + goal x2
    # --------------------------------------------------------

    elif variant == "current3_goal2":

        parts.extend(
            repeat_text(
                "CURRENT REQUEST",
                current_request,
                3,
            )
        )

        parts.extend(
            repeat_text(
                "GOAL",
                goal_text,
                2,
            )
        )

    else:

        raise ValueError(
            f"Unknown HH variant: {variant}"
        )

    return "\n\n".join(parts)


# ============================================================
# Build queries
# ============================================================

def build_query_sets(
    dataset,
    catalog,
):
    """Build common non-HH queries and HH variant queries."""

    variants = [
        "baseline",
        "current4",
        "current4_goal1",
        "current2_goal1",
        "current3_goal2",
    ]

    # Non-HH queries only need retrieval once.
    common_queries = []
    common_metadata = []

    # Each HH variant gets its own query list.
    hh_queries = {
        variant: []
        for variant in variants
    }

    hh_metadata = []

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

        specificity_counts[
            specificity
        ] += 1

        for turn_number in range(
            1,
            NUM_TURNS + 1,
        ):

            # =================================================
            # HH
            # =================================================

            if specificity == "HH":

                current_request = (
                    get_current_user_message(
                        item["conversations"],
                        turn_number,
                    )
                )

                goal_text = get_goal_text(
                    goal
                )

                for variant in variants:

                    query = build_hh_variant(
                        current_request=
                            current_request,
                        goal_text=
                            goal_text,
                        variant=
                            variant,
                    )

                    hh_queries[
                        variant
                    ].append(
                        query
                    )

                hh_metadata.append(
                    {
                        "session_id":
                            item["session_id"],

                        "turn_number":
                            turn_number,

                        "specificity":
                            "HH",
                    }
                )

            # =================================================
            # HL / LH / LL
            # =================================================

            else:

                query = build_adaptive_query(
                    item=item,
                    target_turn_number=
                        turn_number,
                    catalog=catalog,
                )

                common_queries.append(
                    query
                )

                common_metadata.append(
                    {
                        "session_id":
                            item["session_id"],

                        "turn_number":
                            turn_number,

                        "specificity":
                            specificity,
                    }
                )

    return (
        variants,
        common_queries,
        common_metadata,
        hh_queries,
        hh_metadata,
        specificity_counts,
    )


# ============================================================
# Convert retrieval output to predictions
# ============================================================

def make_prediction_dict(
    metadata,
    rankings,
):
    """Create dictionary indexed by session + turn."""

    output = {}

    for info, tracks in zip(
        metadata,
        rankings,
    ):

        key = (
            info["session_id"],
            info["turn_number"],
        )

        output[key] = {
            "session_id":
                info["session_id"],

            "turn_number":
                info["turn_number"],

            "predicted_track_ids":
                tracks,

            "predicted_response":
                "",
        }

    return output


# ============================================================
# Merge common + HH
# ============================================================

def merge_predictions(
    common_predictions,
    hh_predictions,
    ground_truth,
):
    """Merge predictions and preserve GT ordering."""

    merged = {}

    merged.update(
        common_predictions
    )

    merged.update(
        hh_predictions
    )

    predictions = []

    for gt in ground_truth:

        key = (
            gt["session_id"],
            gt["turn_number"],
        )

        if key not in merged:

            raise KeyError(
                "Missing prediction for "
                f"{key}"
            )

        predictions.append(
            merged[key]
        )

    return predictions


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Tune HH query formulation "
            "for MusicCRS"
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
        default=100,
    )

    parser.add_argument(
        "--ground_truth",
        default="ground_truth_test100.json",
    )

    parser.add_argument(
        "--catalog_size",
        type=int,
        default=47071,
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
        default="results_hh_query_tuning.json",
    )

    parser.add_argument(
        "--best_predictions",
        default="predictions_hh_query_best.json",
    )

    args = parser.parse_args()

    print()
    print("=" * 90)
    print("MusicCRS HH QUERY TUNING")
    print("=" * 90)

    print(
        f"Sessions     : "
        f"{args.max_sessions}"
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
        f"RRF k        : "
        f"{args.rrf_k}"
    )

    # ========================================================
    # Ground truth
    # ========================================================

    with open(
        args.ground_truth,
        "r",
        encoding="utf-8",
    ) as f:

        ground_truth = json.load(f)

    print(
        f"Ground truth : "
        f"{len(ground_truth)} turns"
    )

    # ========================================================
    # Dataset
    # ========================================================

    print()
    print("Loading dataset...")

    dataset = load_dataset(
        args.dialogue_dataset,
        split=args.split,
    )

    if args.max_sessions:

        count = min(
            args.max_sessions,
            len(dataset),
        )

        dataset = dataset.select(
            range(count)
        )

    print(
        f"Sessions loaded: "
        f"{len(dataset)}"
    )

    # ========================================================
    # Catalog
    # ========================================================

    print()
    print("Loading catalog...")

    catalog = MusicCatalogLoader()

    # ========================================================
    # Queries
    # ========================================================

    print()
    print("Building queries...")

    (
        variants,
        common_queries,
        common_metadata,
        hh_queries,
        hh_metadata,
        specificity_counts,
    ) = build_query_sets(
        dataset,
        catalog,
    )

    print()
    print("Sessions by specificity:")

    for spec, count in (
        specificity_counts.items()
    ):

        print(
            f"  {spec}: {count}"
        )

    print()
    print(
        f"Non-HH turns : "
        f"{len(common_queries)}"
    )

    print(
        f"HH turns     : "
        f"{len(hh_metadata)}"
    )

    # ========================================================
    # Retriever
    # ========================================================

    print()
    print("Initializing retriever...")

    retriever = HybridRetriever(
        candidate_k=
            args.candidate_k,

        dense_weight=
            args.dense_weight,

        rrf_k=
            args.rrf_k,
    )

    # ========================================================
    # Retrieve non-HH ONCE
    # ========================================================

    print()
    print("=" * 90)
    print(
        "RETRIEVING COMMON HL/LH/LL QUERIES"
    )
    print("=" * 90)

    start = time.time()

    common_rankings = (
        retriever.batch_text_to_item_retrieval(
            common_queries,
            args.topk,
        )
    )

    common_predictions = (
        make_prediction_dict(
            common_metadata,
            common_rankings,
        )
    )

    print(
        f"Common retrieval: "
        f"{(time.time() - start) / 60:.2f} min"
    )

    # ========================================================
    # HH variants
    # ========================================================

    experiments = []

    best_score = -1.0
    best_variant = None
    best_metrics = None
    best_predictions = None

    print()
    print("=" * 90)
    print("TESTING HH QUERY VARIANTS")
    print("=" * 90)

    for variant in variants:

        print()
        print("-" * 90)

        print(
            f"Variant: {variant}"
        )

        print("-" * 90)

        variant_start = time.time()

        hh_rankings = (
            retriever.batch_text_to_item_retrieval(
                hh_queries[variant],
                args.topk,
            )
        )

        hh_predictions = (
            make_prediction_dict(
                hh_metadata,
                hh_rankings,
            )
        )

        predictions = merge_predictions(
            common_predictions,
            hh_predictions,
            ground_truth,
        )

        metrics = evaluate(
            predictions=predictions,
            ground_truth=ground_truth,
            catalog_size=
                args.catalog_size,
        )

        runtime = (
            time.time()
            - variant_start
        )

        print(
            f"nDCG@1     : "
            f"{metrics['ndcg@1']:.6f}"
        )

        print(
            f"nDCG@10    : "
            f"{metrics['ndcg@10']:.6f}"
        )

        print(
            f"nDCG@20    : "
            f"{metrics['ndcg@20']:.6f}"
        )

        print(
            f"Diversity  : "
            f"{metrics['catalog_diversity']:.6f}"
        )

        print(
            f"Final      : "
            f"{metrics['final_score']:.6f}"
        )

        print(
            f"Runtime    : "
            f"{runtime / 60:.2f} min"
        )

        experiments.append(
            {
                "variant":
                    variant,

                "runtime_seconds":
                    runtime,

                **metrics,
            }
        )

        if (
            metrics["final_score"]
            > best_score
        ):

            best_score = (
                metrics["final_score"]
            )

            best_variant = variant

            best_metrics = dict(
                metrics
            )

            best_predictions = (
                predictions
            )

            print(
                "*** NEW BEST ***"
            )

    # ========================================================
    # Save best predictions
    # ========================================================

    with open(
        args.best_predictions,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            best_predictions,
            f,
            indent=2,
        )

    # ========================================================
    # Save tuning results
    # ========================================================

    output = {
        "best_variant":
            best_variant,

        "best_metrics":
            best_metrics,

        "experiments":
            experiments,
    }

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 90)
    print("BEST HH QUERY CONFIGURATION")
    print("=" * 90)

    print(
        f"Variant    : "
        f"{best_variant}"
    )

    print()

    print(
        f"nDCG@1     : "
        f"{best_metrics['ndcg@1']:.6f}"
    )

    print(
        f"nDCG@10    : "
        f"{best_metrics['ndcg@10']:.6f}"
    )

    print(
        f"nDCG@20    : "
        f"{best_metrics['ndcg@20']:.6f}"
    )

    print(
        f"Diversity  : "
        f"{best_metrics['catalog_diversity']:.6f}"
    )

    print(
        f"FINAL      : "
        f"{best_metrics['final_score']:.6f}"
    )

    print()

    print(
        f"Best predictions: "
        f"{args.best_predictions}"
    )

    print(
        f"Results         : "
        f"{args.output}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
