"""Analyze retrieval performance by conversation specificity.

Compares two existing prediction files:
    1. Goal-aware Hybrid
    2. Adaptive Goal-aware Hybrid

No retrieval is performed.

For each specificity group (HH, HL, LH, LL), the script computes:
    - number of sessions / turns
    - nDCG@1
    - nDCG@10
    - nDCG@20
    - catalog diversity
    - final score

This helps determine which retrieval strategy should be used for each
specificity group in a selective ensemble.
"""

import argparse
import json
from collections import defaultdict

from datasets import load_dataset

from .evaluation.evaluate import evaluate


DEFAULT_DATASET = (
    "talkpl-ai/TalkPlayData-Challenge-Dataset"
)

SPECIFICITY_TYPES = [
    "HH",
    "HL",
    "LH",
    "LL",
]


# ============================================================
# Load JSON
# ============================================================

def load_json(path: str) -> list[dict]:
    """Load JSON file."""

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# Session -> specificity
# ============================================================

def load_session_specificity(
    dataset_name: str,
    split: str,
) -> dict[str, str]:
    """Map every session ID to its specificity type."""

    print()
    print("Loading dialogue dataset...")

    dataset = load_dataset(
        dataset_name,
        split=split,
    )

    mapping = {}

    counts = defaultdict(int)

    for item in dataset:

        session_id = item["session_id"]

        goal = (
            item.get("conversation_goal")
            or {}
        )

        specificity = goal.get(
            "specificity",
            "UNKNOWN",
        )

        mapping[session_id] = specificity

        counts[specificity] += 1

    print(
        f"Loaded {len(mapping)} sessions."
    )

    print()
    print("Session distribution:")

    for specificity in SPECIFICITY_TYPES:

        print(
            f"  {specificity}: "
            f"{counts[specificity]}"
        )

    if counts["UNKNOWN"]:

        print(
            f"  UNKNOWN: "
            f"{counts['UNKNOWN']}"
        )

    return mapping


# ============================================================
# Group records
# ============================================================

def group_by_specificity(
    records: list[dict],
    session_specificity: dict[str, str],
) -> dict[str, list[dict]]:
    """Group prediction/ground-truth records by specificity."""

    groups = defaultdict(list)

    for record in records:

        session_id = record[
            "session_id"
        ]

        specificity = (
            session_specificity.get(
                session_id,
                "UNKNOWN",
            )
        )

        groups[specificity].append(
            record
        )

    return groups


# ============================================================
# Evaluate one method
# ============================================================

def evaluate_groups(
    predictions: list[dict],
    ground_truth: list[dict],
    session_specificity: dict[str, str],
    catalog_size: int,
) -> dict[str, dict]:
    """Evaluate predictions separately for each specificity."""

    prediction_groups = (
        group_by_specificity(
            predictions,
            session_specificity,
        )
    )

    ground_truth_groups = (
        group_by_specificity(
            ground_truth,
            session_specificity,
        )
    )

    results = {}

    for specificity in SPECIFICITY_TYPES:

        preds = prediction_groups[
            specificity
        ]

        gt = ground_truth_groups[
            specificity
        ]

        if not gt:
            continue

        metrics = evaluate(
            predictions=preds,
            ground_truth=gt,
            catalog_size=catalog_size,
        )

        metrics["turns"] = len(gt)

        # Every MusicCRS session has 8 turns.
        metrics["sessions"] = (
            len(gt) // 8
        )

        results[specificity] = metrics

    return results


# ============================================================
# Print comparison
# ============================================================

def print_comparison(
    goal_results: dict,
    adaptive_results: dict,
) -> None:
    """Print side-by-side comparison."""

    print()
    print("=" * 112)
    print("PER-SPECIFICITY COMPARISON")
    print("=" * 112)

    print(
        f"{'Spec':<7}"
        f"{'Sessions':<10}"
        f"{'Method':<15}"
        f"{'nDCG@1':<13}"
        f"{'nDCG@10':<13}"
        f"{'nDCG@20':<13}"
        f"{'Diversity':<13}"
        f"{'Final':<13}"
    )

    print("-" * 112)

    for specificity in SPECIFICITY_TYPES:

        goal = goal_results.get(
            specificity
        )

        adaptive = adaptive_results.get(
            specificity
        )

        if (
            goal is None
            or adaptive is None
        ):
            continue

        print(
            f"{specificity:<7}"
            f"{goal['sessions']:<10}"
            f"{'Goal-aware':<15}"
            f"{goal['ndcg@1']:<13.6f}"
            f"{goal['ndcg@10']:<13.6f}"
            f"{goal['ndcg@20']:<13.6f}"
            f"{goal['catalog_diversity']:<13.6f}"
            f"{goal['final_score']:<13.6f}"
        )

        print(
            f"{'':<7}"
            f"{'':<10}"
            f"{'Adaptive':<15}"
            f"{adaptive['ndcg@1']:<13.6f}"
            f"{adaptive['ndcg@10']:<13.6f}"
            f"{adaptive['ndcg@20']:<13.6f}"
            f"{adaptive['catalog_diversity']:<13.6f}"
            f"{adaptive['final_score']:<13.6f}"
        )

        print("-" * 112)


# ============================================================
# Print differences
# ============================================================

def print_differences(
    goal_results: dict,
    adaptive_results: dict,
) -> None:
    """Show adaptive - goal-aware differences."""

    print()
    print("=" * 95)
    print("ADAPTIVE - GOAL-AWARE DIFFERENCE")
    print("=" * 95)

    print(
        f"{'Spec':<8}"
        f"{'Δ nDCG@1':<16}"
        f"{'Δ nDCG@10':<16}"
        f"{'Δ nDCG@20':<16}"
        f"{'Δ Diversity':<16}"
        f"{'Δ Final':<16}"
    )

    print("-" * 95)

    for specificity in SPECIFICITY_TYPES:

        goal = goal_results.get(
            specificity
        )

        adaptive = adaptive_results.get(
            specificity
        )

        if (
            goal is None
            or adaptive is None
        ):
            continue

        delta_1 = (
            adaptive["ndcg@1"]
            - goal["ndcg@1"]
        )

        delta_10 = (
            adaptive["ndcg@10"]
            - goal["ndcg@10"]
        )

        delta_20 = (
            adaptive["ndcg@20"]
            - goal["ndcg@20"]
        )

        delta_diversity = (
            adaptive["catalog_diversity"]
            - goal["catalog_diversity"]
        )

        delta_final = (
            adaptive["final_score"]
            - goal["final_score"]
        )

        print(
            f"{specificity:<8}"
            f"{delta_1:<+16.6f}"
            f"{delta_10:<+16.6f}"
            f"{delta_20:<+16.6f}"
            f"{delta_diversity:<+16.6f}"
            f"{delta_final:<+16.6f}"
        )


# ============================================================
# Recommendation
# ============================================================

def print_recommendations(
    goal_results: dict,
    adaptive_results: dict,
) -> None:
    """Print which method wins for each specificity."""

    print()
    print("=" * 72)
    print("PRELIMINARY ROUTING RECOMMENDATION")
    print("=" * 72)

    for specificity in SPECIFICITY_TYPES:

        goal = goal_results.get(
            specificity
        )

        adaptive = adaptive_results.get(
            specificity
        )

        if (
            goal is None
            or adaptive is None
        ):
            continue

        # Use final score only as a convenient per-group
        # diagnostic here. We will still evaluate the actual
        # combined predictions globally afterward.
        if (
            adaptive["final_score"]
            > goal["final_score"]
        ):

            winner = "ADAPTIVE"

        elif (
            adaptive["final_score"]
            < goal["final_score"]
        ):

            winner = "GOAL-AWARE"

        else:

            winner = "TIE"

        print(
            f"{specificity}: {winner}"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Compare Goal-aware and Adaptive "
            "MusicCRS predictions by specificity"
        )
    )

    parser.add_argument(
        "--goal_predictions",
        default=(
            "predictions_"
            "goal_aware_hybrid_full.json"
        ),
    )

    parser.add_argument(
        "--adaptive_predictions",
        default=(
            "predictions_"
            "adaptive_goal_full.json"
        ),
    )

    parser.add_argument(
        "--ground_truth",
        default="ground_truth.json",
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
        "--catalog_size",
        type=int,
        default=47071,
    )

    parser.add_argument(
        "--output",
        default=(
            "results_specificity_"
            "analysis.json"
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 72)
    print("MusicCRS SPECIFICITY ANALYSIS")
    print("=" * 72)

    print(
        f"Goal predictions    : "
        f"{args.goal_predictions}"
    )

    print(
        f"Adaptive predictions: "
        f"{args.adaptive_predictions}"
    )

    print(
        f"Ground truth        : "
        f"{args.ground_truth}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    goal_predictions = load_json(
        args.goal_predictions
    )

    adaptive_predictions = load_json(
        args.adaptive_predictions
    )

    ground_truth = load_json(
        args.ground_truth
    )

    print()
    print(
        f"Goal predictions    : "
        f"{len(goal_predictions)}"
    )

    print(
        f"Adaptive predictions: "
        f"{len(adaptive_predictions)}"
    )

    print(
        f"Ground truth        : "
        f"{len(ground_truth)}"
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if (
        len(goal_predictions)
        != len(ground_truth)
    ):
        raise ValueError(
            "Goal-aware prediction count "
            "does not match ground truth."
        )

    if (
        len(adaptive_predictions)
        != len(ground_truth)
    ):
        raise ValueError(
            "Adaptive prediction count "
            "does not match ground truth."
        )

    # --------------------------------------------------------
    # Session specificity
    # --------------------------------------------------------

    session_specificity = (
        load_session_specificity(
            dataset_name=(
                args.dialogue_dataset
            ),
            split=args.split,
        )
    )

    # --------------------------------------------------------
    # Goal-aware
    # --------------------------------------------------------

    print()
    print(
        "Evaluating Goal-aware "
        "by specificity..."
    )

    goal_results = evaluate_groups(
        predictions=goal_predictions,
        ground_truth=ground_truth,
        session_specificity=(
            session_specificity
        ),
        catalog_size=args.catalog_size,
    )

    # --------------------------------------------------------
    # Adaptive
    # --------------------------------------------------------

    print(
        "Evaluating Adaptive "
        "by specificity..."
    )

    adaptive_results = evaluate_groups(
        predictions=adaptive_predictions,
        ground_truth=ground_truth,
        session_specificity=(
            session_specificity
        ),
        catalog_size=args.catalog_size,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_comparison(
        goal_results,
        adaptive_results,
    )

    print_differences(
        goal_results,
        adaptive_results,
    )

    print_recommendations(
        goal_results,
        adaptive_results,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {
        "goal_aware":
            goal_results,

        "adaptive":
            adaptive_results,
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

    print()
    print(
        f"Analysis saved to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
