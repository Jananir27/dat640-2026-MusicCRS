"""Build a specificity-routed ensemble from existing MusicCRS predictions.

Routing:
    HH -> Goal-aware
    HL -> Adaptive
    LH -> Adaptive
    LL -> Goal-aware

No retrieval is performed.

The script:
    1. Loads Goal-aware predictions
    2. Loads Adaptive predictions
    3. Loads dialogue specificity
    4. Selects the appropriate ranking for each session
    5. Saves combined predictions
    6. Evaluates globally
"""

import argparse
import json
from collections import Counter

from datasets import load_dataset

from .evaluation.evaluate import evaluate


DEFAULT_DATASET = (
    "talkpl-ai/TalkPlayData-Challenge-Dataset"
)


# ============================================================
# Load JSON
# ============================================================

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Build specificity-based MusicCRS ensemble"
    )

    parser.add_argument(
        "--goal_predictions",
        default="predictions_goal_aware_hybrid_full.json",
    )

    parser.add_argument(
        "--adaptive_predictions",
        default="predictions_adaptive_goal_full.json",
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
        default="predictions_specificity_ensemble.json",
    )

    parser.add_argument(
        "--results",
        default="results_specificity_ensemble.json",
    )

    args = parser.parse_args()

    print()
    print("=" * 75)
    print("MusicCRS SPECIFICITY ENSEMBLE")
    print("=" * 75)

    # --------------------------------------------------------
    # Load prediction files
    # --------------------------------------------------------

    print("\nLoading predictions...")

    goal_predictions = load_json(
        args.goal_predictions
    )

    adaptive_predictions = load_json(
        args.adaptive_predictions
    )

    ground_truth = load_json(
        args.ground_truth
    )

    print(
        f"Goal-aware predictions : "
        f"{len(goal_predictions)}"
    )

    print(
        f"Adaptive predictions   : "
        f"{len(adaptive_predictions)}"
    )

    print(
        f"Ground truth           : "
        f"{len(ground_truth)}"
    )

    # --------------------------------------------------------
    # Build lookup tables
    # --------------------------------------------------------

    goal_lookup = {
        (
            p["session_id"],
            p["turn_number"],
        ): p
        for p in goal_predictions
    }

    adaptive_lookup = {
        (
            p["session_id"],
            p["turn_number"],
        ): p
        for p in adaptive_predictions
    }

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dialogue dataset...")

    dataset = load_dataset(
        args.dialogue_dataset,
        split=args.split,
    )

    # --------------------------------------------------------
    # Session -> specificity
    # --------------------------------------------------------

    session_specificity = {}

    for item in dataset:

        goal = (
            item.get("conversation_goal")
            or {}
        )

        session_specificity[
            item["session_id"]
        ] = goal.get(
            "specificity",
            "LL",
        )

    # --------------------------------------------------------
    # Routing
    # --------------------------------------------------------

    routing = {
        "HH": "goal",
        "HL": "adaptive",
        "LH": "adaptive",
        "LL": "goal",
    }

    print("\nRouting configuration:")

    for spec, method in routing.items():
        print(
            f"  {spec} -> "
            f"{method.upper()}"
        )

    # --------------------------------------------------------
    # Build ensemble
    # --------------------------------------------------------

    ensemble = []

    method_counts = Counter()
    specificity_counts = Counter()

    missing = 0

    for gt in ground_truth:

        session_id = gt["session_id"]
        turn_number = gt["turn_number"]

        key = (
            session_id,
            turn_number,
        )

        specificity = (
            session_specificity.get(
                session_id,
                "LL",
            )
        )

        method = routing.get(
            specificity,
            "goal",
        )

        specificity_counts[
            specificity
        ] += 1

        if method == "adaptive":

            prediction = (
                adaptive_lookup.get(key)
            )

        else:

            prediction = (
                goal_lookup.get(key)
            )

        if prediction is None:

            missing += 1
            continue

        # Make a clean copy
        ensemble.append(
            {
                "session_id":
                    prediction["session_id"],

                "turn_number":
                    prediction["turn_number"],

                "predicted_track_ids":
                    prediction[
                        "predicted_track_ids"
                    ],

                "predicted_response":
                    prediction.get(
                        "predicted_response",
                        "",
                    ),
            }
        )

        method_counts[
            method
        ] += 1

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("ENSEMBLE SUMMARY")
    print("=" * 75)

    print(
        f"Combined predictions : "
        f"{len(ensemble)}"
    )

    print(
        f"Missing predictions  : "
        f"{missing}"
    )

    print("\nTurns by specificity:")

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

    print("\nTurns by method:")

    print(
        f"  Goal-aware : "
        f"{method_counts['goal']}"
    )

    print(
        f"  Adaptive   : "
        f"{method_counts['adaptive']}"
    )

    if missing:

        raise RuntimeError(
            f"{missing} predictions are missing."
        )

    if len(ensemble) != len(
        ground_truth
    ):

        raise RuntimeError(
            "Ensemble prediction count "
            "does not match ground truth."
        )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            ensemble,
            f,
            indent=2,
        )

    print(
        f"\nPredictions saved to: "
        f"{args.output}"
    )

    # --------------------------------------------------------
    # Global evaluation
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("GLOBAL EVALUATION")
    print("=" * 75)

    results = evaluate(
        predictions=ensemble,
        ground_truth=ground_truth,
        catalog_size=args.catalog_size,
    )

    with open(
        args.results,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print(
        f"nDCG@1            : "
        f"{results['ndcg@1']:.6f}"
    )

    print(
        f"nDCG@10           : "
        f"{results['ndcg@10']:.6f}"
    )

    print(
        f"nDCG@20           : "
        f"{results['ndcg@20']:.6f}"
    )

    print(
        f"Catalog diversity : "
        f"{results['catalog_diversity']:.6f}"
    )

    print(
        f"FINAL SCORE       : "
        f"{results['final_score']:.6f}"
    )

    print()
    print(
        f"Results saved to: "
        f"{args.results}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()
