"""Tune diversity-aware reranking for MusicCRS.

Starting point:
    predictions_exact_specificity_best.json

Goal:
    Increase catalog diversity while preserving nDCG.

Strategy:
    1. Calculate how frequently every track is recommended.
    2. Protect the first N positions of every ranking.
    3. Rerank only the remaining positions.
    4. Give a novelty bonus to tracks recommended less frequently.
    5. Evaluate multiple protected-prefix and novelty-strength settings.

No BM25 retrieval.
No dense retrieval.
No GPU required.
"""

import argparse
import json
import math
from collections import Counter

from .evaluation.evaluate import evaluate


# ============================================================
# Utilities
# ============================================================

def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# Track frequency
# ============================================================

def calculate_track_frequency(predictions):
    """Count how often each track appears in predictions."""

    frequency = Counter()

    for prediction in predictions:

        for track_id in prediction[
            "predicted_track_ids"
        ]:

            frequency[
                track_id
            ] += 1

    return frequency


# ============================================================
# Novelty score
# ============================================================

def novelty_score(
    track_id,
    frequency,
    total_turns,
):
    """Calculate inverse-frequency novelty.

    Common tracks receive scores near 0.
    Rare tracks receive larger scores.
    """

    count = frequency.get(
        track_id,
        0,
    )

    return math.log(
        (total_turns + 1)
        / (count + 1)
    )


# ============================================================
# Diversity reranking
# ============================================================

def rerank_prediction(
    track_ids,
    frequency,
    total_turns,
    protect_top,
    novelty_weight,
):
    """Rerank lower part of one recommendation list."""

    if (
        novelty_weight == 0.0
        or protect_top >= len(track_ids)
    ):
        return list(track_ids)

    protected = list(
        track_ids[:protect_top]
    )

    remaining = list(
        track_ids[protect_top:]
    )

    remaining_count = len(
        remaining
    )

    scored = []

    for local_rank, track_id in enumerate(
        remaining,
        start=1,
    ):

        # ----------------------------------------------------
        # Base ranking score
        #
        # Keeps original ordering important.
        # Best remaining candidate gets score near 1.
        # ----------------------------------------------------

        base_score = (
            remaining_count
            - local_rank
            + 1
        ) / remaining_count

        # ----------------------------------------------------
        # Novelty
        # ----------------------------------------------------

        novelty = novelty_score(
            track_id,
            frequency,
            total_turns,
        )

        # Normalize roughly to 0-1.
        normalized_novelty = (
            novelty
            / math.log(
                total_turns + 1
            )
        )

        final_score = (
            base_score
            + novelty_weight
            * normalized_novelty
        )

        scored.append(
            (
                track_id,
                final_score,
                local_rank,
            )
        )

    scored.sort(
        key=lambda item: (
            -item[1],
            item[2],
        )
    )

    reranked = [
        track_id
        for track_id, _, _ in scored
    ]

    return (
        protected
        + reranked
    )


# ============================================================
# Apply globally
# ============================================================

def apply_diversity_reranking(
    predictions,
    frequency,
    protect_top,
    novelty_weight,
):
    """Apply diversity reranking to all predictions."""

    total_turns = len(
        predictions
    )

    output = []

    changed_turns = 0

    for prediction in predictions:

        original = prediction[
            "predicted_track_ids"
        ]

        reranked = rerank_prediction(
            track_ids=original,
            frequency=frequency,
            total_turns=total_turns,
            protect_top=protect_top,
            novelty_weight=novelty_weight,
        )

        if reranked != original:
            changed_turns += 1

        new_prediction = dict(
            prediction
        )

        new_prediction[
            "predicted_track_ids"
        ] = reranked

        output.append(
            new_prediction
        )

    return (
        output,
        changed_turns,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Tune diversity-aware reranking "
            "for MusicCRS"
        )
    )

    parser.add_argument(
        "--predictions",
        default=(
            "predictions_exact_specificity_best.json"
        ),
    )

    parser.add_argument(
        "--ground_truth",
        default="ground_truth.json",
    )

    parser.add_argument(
        "--catalog_size",
        type=int,
        default=47071,
    )

    parser.add_argument(
        "--output",
        default=(
            "results_diversity_tuning.json"
        ),
    )

    parser.add_argument(
        "--best_predictions",
        default=(
            "predictions_diversity_best.json"
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 95)
    print(
        "MusicCRS DIVERSITY-AWARE RERANKING"
    )
    print("=" * 95)

    # ========================================================
    # Load
    # ========================================================

    predictions = load_json(
        args.predictions
    )

    ground_truth = load_json(
        args.ground_truth
    )

    print()
    print(
        f"Predictions : "
        f"{len(predictions)}"
    )

    print(
        f"Ground truth: "
        f"{len(ground_truth)}"
    )

    if (
        len(predictions)
        != len(ground_truth)
    ):

        raise ValueError(
            "Prediction and ground-truth "
            "counts do not match."
        )

    # ========================================================
    # Baseline
    # ========================================================

    baseline = evaluate(
        predictions=predictions,
        ground_truth=ground_truth,
        catalog_size=args.catalog_size,
    )

    print()
    print("=" * 95)
    print("CURRENT BEST BASELINE")
    print("=" * 95)

    print(
        f"nDCG@1     : "
        f"{baseline['ndcg@1']:.6f}"
    )

    print(
        f"nDCG@10    : "
        f"{baseline['ndcg@10']:.6f}"
    )

    print(
        f"nDCG@20    : "
        f"{baseline['ndcg@20']:.6f}"
    )

    print(
        f"Diversity  : "
        f"{baseline['catalog_diversity']:.6f}"
    )

    print(
        f"FINAL      : "
        f"{baseline['final_score']:.6f}"
    )

    # ========================================================
    # Frequency
    # ========================================================

    print()
    print(
        "Calculating recommendation "
        "frequency..."
    )

    frequency = (
        calculate_track_frequency(
            predictions
        )
    )

    print(
        f"Unique recommended tracks: "
        f"{len(frequency)}"
    )

    most_common = (
        frequency.most_common(10)
    )

    print()
    print(
        "10 most frequently "
        "recommended tracks:"
    )

    for track_id, count in most_common:

        print(
            f"  {track_id}: "
            f"{count}"
        )

    # ========================================================
    # Search grid
    # ========================================================

    protect_values = [
        5,
        8,
        10,
        12,
        15,
    ]

    novelty_weights = [
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.50,
    ]

    total_experiments = (
        len(protect_values)
        * len(novelty_weights)
    )

    print()
    print("=" * 95)
    print("TUNING")
    print("=" * 95)

    print(
        f"Experiments: "
        f"{total_experiments}"
    )

    best_score = (
        baseline["final_score"]
    )

    best_metrics = dict(
        baseline
    )

    best_predictions = list(
        predictions
    )

    best_protect_top = None
    best_novelty_weight = 0.0
    best_changed_turns = 0

    experiments = []

    experiment_number = 0

    # ========================================================
    # Tune
    # ========================================================

    for protect_top in protect_values:

        for novelty_weight in (
            novelty_weights
        ):

            experiment_number += 1

            (
                candidate_predictions,
                changed_turns,
            ) = apply_diversity_reranking(
                predictions=predictions,
                frequency=frequency,
                protect_top=protect_top,
                novelty_weight=novelty_weight,
            )

            metrics = evaluate(
                predictions=
                    candidate_predictions,

                ground_truth=
                    ground_truth,

                catalog_size=
                    args.catalog_size,
            )

            experiments.append(
                {
                    "protect_top":
                        protect_top,

                    "novelty_weight":
                        novelty_weight,

                    "changed_turns":
                        changed_turns,

                    **metrics,
                }
            )

            print(
                f"[{experiment_number:02d}/"
                f"{total_experiments}] "
                f"protect={protect_top:2d} "
                f"novelty={novelty_weight:.2f} "
                f"nDCG20="
                f"{metrics['ndcg@20']:.6f} "
                f"div="
                f"{metrics['catalog_diversity']:.6f} "
                f"final="
                f"{metrics['final_score']:.6f}"
            )

            if (
                metrics["final_score"]
                > best_score
            ):

                best_score = (
                    metrics[
                        "final_score"
                    ]
                )

                best_metrics = dict(
                    metrics
                )

                best_predictions = (
                    candidate_predictions
                )

                best_protect_top = (
                    protect_top
                )

                best_novelty_weight = (
                    novelty_weight
                )

                best_changed_turns = (
                    changed_turns
                )

                print()
                print(
                    "*** NEW BEST ***"
                )

                print(
                    f"protect_top    = "
                    f"{best_protect_top}"
                )

                print(
                    f"novelty_weight = "
                    f"{best_novelty_weight}"
                )

                print(
                    f"FINAL          = "
                    f"{best_score:.6f}"
                )

                print()

    # ========================================================
    # Save
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

    results = {
        "baseline":
            baseline,

        "best_configuration": {
            "protect_top":
                best_protect_top,

            "novelty_weight":
                best_novelty_weight,

            "changed_turns":
                best_changed_turns,
        },

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
            results,
            f,
            indent=2,
        )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 95)
    print(
        "BEST DIVERSITY CONFIGURATION"
    )
    print("=" * 95)

    if best_protect_top is None:

        print(
            "No diversity reranking "
            "beat the baseline."
        )

        print()
        print(
            "Keeping original predictions."
        )

    else:

        print(
            f"Protect top    : "
            f"{best_protect_top}"
        )

        print(
            f"Novelty weight : "
            f"{best_novelty_weight}"
        )

        print(
            f"Changed turns  : "
            f"{best_changed_turns}"
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

    print("=" * 95)


if __name__ == "__main__":
    main()
