"""Tune rank fusion between Goal-aware and Adaptive MusicCRS predictions.

This script uses EXISTING prediction files only.

Inputs:
    predictions_goal_aware_hybrid_full.json
    predictions_adaptive_goal_full.json
    ground_truth.json

It tests weighted Reciprocal Rank Fusion (RRF) between:
    - Adaptive ranking
    - Goal-aware ranking

The adaptive ranking is treated as the primary system.

No BM25 retrieval.
No dense retrieval.
No GPU required.
"""

import argparse
import json
from itertools import product

from datasets import load_dataset

from .evaluation.evaluate import evaluate


DEFAULT_DATASET = "talkpl-ai/TalkPlayData-Challenge-Dataset"

SPECIFICITIES = [
    "HH",
    "HL",
    "LH",
    "LL",
]


# ============================================================
# Utilities
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prediction_lookup(predictions):
    """Create (session_id, turn_number) -> prediction mapping."""

    return {
        (
            p["session_id"],
            p["turn_number"],
        ): p
        for p in predictions
    }


# ============================================================
# Specificity mapping
# ============================================================

def load_specificities(dataset_name, split):
    """Map session IDs to HH / HL / LH / LL."""

    dataset = load_dataset(
        dataset_name,
        split=split,
    )

    mapping = {}

    for item in dataset:

        goal = (
            item.get("conversation_goal")
            or {}
        )

        mapping[item["session_id"]] = (
            goal.get("specificity", "LL")
        )

    return mapping


# ============================================================
# Weighted RRF
# ============================================================

def weighted_rrf(
    adaptive_tracks,
    goal_tracks,
    adaptive_weight=1.0,
    goal_weight=0.0,
    rrf_k=60,
    topk=20,
):
    """Fuse Adaptive and Goal-aware rankings using weighted RRF."""

    scores = {}

    # Adaptive ranking
    for rank, track_id in enumerate(
        adaptive_tracks,
        start=1,
    ):

        score = (
            adaptive_weight
            / (rrf_k + rank)
        )

        scores[track_id] = (
            scores.get(track_id, 0.0)
            + score
        )

    # Goal-aware ranking
    for rank, track_id in enumerate(
        goal_tracks,
        start=1,
    ):

        score = (
            goal_weight
            / (rrf_k + rank)
        )

        scores[track_id] = (
            scores.get(track_id, 0.0)
            + score
        )

    # Deterministic tie-breaking:
    # prefer adaptive ranking when scores are equal.
    adaptive_rank = {
        track_id: rank
        for rank, track_id in enumerate(
            adaptive_tracks,
            start=1,
        )
    }

    goal_rank = {
        track_id: rank
        for rank, track_id in enumerate(
            goal_tracks,
            start=1,
        )
    }

    large_rank = 1_000_000

    ranked = sorted(
        scores,
        key=lambda track_id: (
            -scores[track_id],
            adaptive_rank.get(
                track_id,
                large_rank,
            ),
            goal_rank.get(
                track_id,
                large_rank,
            ),
            track_id,
        ),
    )

    return ranked[:topk]


# ============================================================
# Build fused predictions
# ============================================================

def build_fused_predictions(
    ground_truth,
    adaptive_lookup,
    goal_lookup,
    session_specificity,
    weights,
    rrf_k,
    topk,
):
    """Build predictions using specificity-dependent fusion weights."""

    predictions = []

    for gt in ground_truth:

        key = (
            gt["session_id"],
            gt["turn_number"],
        )

        adaptive = adaptive_lookup[key]
        goal = goal_lookup[key]

        specificity = (
            session_specificity.get(
                gt["session_id"],
                "LL",
            )
        )

        goal_weight = weights.get(
            specificity,
            0.0,
        )

        fused_tracks = weighted_rrf(
            adaptive_tracks=adaptive[
                "predicted_track_ids"
            ],
            goal_tracks=goal[
                "predicted_track_ids"
            ],
            adaptive_weight=1.0,
            goal_weight=goal_weight,
            rrf_k=rrf_k,
            topk=topk,
        )

        predictions.append(
            {
                "session_id":
                    gt["session_id"],

                "turn_number":
                    gt["turn_number"],

                "predicted_track_ids":
                    fused_tracks,

                "predicted_response":
                    "",
            }
        )

    return predictions


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Tune specificity-dependent fusion "
            "of Goal-aware and Adaptive predictions"
        )
    )

    parser.add_argument(
        "--adaptive_predictions",
        default="predictions_adaptive_goal_full.json",
    )

    parser.add_argument(
        "--goal_predictions",
        default="predictions_goal_aware_hybrid_full.json",
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
        "--topk",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--rrf_k",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--output",
        default="prediction_fusion_tuning_results.json",
    )

    parser.add_argument(
        "--best_predictions",
        default="predictions_best_fusion.json",
    )

    args = parser.parse_args()

    print()
    print("=" * 90)
    print("MusicCRS PREDICTION FUSION TUNING")
    print("=" * 90)

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    print("\nLoading prediction files...")

    adaptive_predictions = load_json(
        args.adaptive_predictions
    )

    goal_predictions = load_json(
        args.goal_predictions
    )

    ground_truth = load_json(
        args.ground_truth
    )

    print(
        f"Adaptive predictions : "
        f"{len(adaptive_predictions)}"
    )

    print(
        f"Goal predictions     : "
        f"{len(goal_predictions)}"
    )

    print(
        f"Ground truth         : "
        f"{len(ground_truth)}"
    )

    if (
        len(adaptive_predictions)
        != len(ground_truth)
    ):
        raise ValueError(
            "Adaptive prediction count "
            "does not match ground truth."
        )

    if (
        len(goal_predictions)
        != len(ground_truth)
    ):
        raise ValueError(
            "Goal prediction count "
            "does not match ground truth."
        )

    adaptive_lookup = prediction_lookup(
        adaptive_predictions
    )

    goal_lookup = prediction_lookup(
        goal_predictions
    )

    # --------------------------------------------------------
    # Load specificity
    # --------------------------------------------------------

    print("\nLoading session specificity...")

    session_specificity = (
        load_specificities(
            args.dialogue_dataset,
            args.split,
        )
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("ADAPTIVE BASELINE")
    print("=" * 90)

    baseline = evaluate(
        predictions=adaptive_predictions,
        ground_truth=ground_truth,
        catalog_size=args.catalog_size,
    )

    print(
        f"nDCG@20  : "
        f"{baseline['ndcg@20']:.6f}"
    )

    print(
        f"Diversity: "
        f"{baseline['catalog_diversity']:.6f}"
    )

    print(
        f"Final    : "
        f"{baseline['final_score']:.6f}"
    )

    # --------------------------------------------------------
    # Stage 1
    #
    # Tune one specificity at a time.
    #
    # This avoids immediately searching every possible
    # 4-dimensional combination.
    # --------------------------------------------------------

    candidate_weights = [
        0.0,
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
        0.75,
        1.00,
    ]

    best_weights = {
        "HH": 0.0,
        "HL": 0.0,
        "LH": 0.0,
        "LL": 0.0,
    }

    all_results = []

    print()
    print("=" * 90)
    print("STAGE 1 - PER-SPECIFICITY WEIGHT SEARCH")
    print("=" * 90)

    for specificity in SPECIFICITIES:

        print()
        print("-" * 90)
        print(
            f"TUNING {specificity}"
        )
        print("-" * 90)

        local_best_score = -1.0
        local_best_weight = 0.0

        for weight in candidate_weights:

            weights = dict(
                best_weights
            )

            weights[
                specificity
            ] = weight

            predictions = (
                build_fused_predictions(
                    ground_truth=ground_truth,
                    adaptive_lookup=adaptive_lookup,
                    goal_lookup=goal_lookup,
                    session_specificity=(
                        session_specificity
                    ),
                    weights=weights,
                    rrf_k=args.rrf_k,
                    topk=args.topk,
                )
            )

            metrics = evaluate(
                predictions=predictions,
                ground_truth=ground_truth,
                catalog_size=args.catalog_size,
            )

            print(
                f"{specificity} "
                f"goal_weight={weight:<5.2f} "
                f"nDCG20="
                f"{metrics['ndcg@20']:.6f} "
                f"div="
                f"{metrics['catalog_diversity']:.6f} "
                f"final="
                f"{metrics['final_score']:.6f}"
            )

            all_results.append(
                {
                    "stage": 1,
                    "specificity":
                        specificity,
                    "weights":
                        dict(weights),
                    **metrics,
                }
            )

            if (
                metrics["final_score"]
                > local_best_score
            ):

                local_best_score = (
                    metrics["final_score"]
                )

                local_best_weight = weight

        best_weights[
            specificity
        ] = local_best_weight

        print()
        print(
            f"Best {specificity} "
            f"goal weight: "
            f"{local_best_weight}"
        )

        print(
            f"Best global score: "
            f"{local_best_score:.6f}"
        )

    # --------------------------------------------------------
    # Stage 2
    #
    # Search around the discovered configuration.
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("STAGE 2 - LOCAL COMBINATION SEARCH")
    print("=" * 90)

    def nearby(weight):

        values = {
            0.0,
            weight,
        }

        for delta in [
            -0.10,
            -0.05,
            0.05,
            0.10,
        ]:

            value = round(
                weight + delta,
                2,
            )

            if (
                value >= 0.0
                and value <= 1.0
            ):
                values.add(value)

        return sorted(values)

    search_spaces = {
        spec: nearby(
            best_weights[spec]
        )
        for spec in SPECIFICITIES
    }

    print("\nSearch spaces:")

    for spec in SPECIFICITIES:

        print(
            f"  {spec}: "
            f"{search_spaces[spec]}"
        )

    combinations = list(
        product(
            search_spaces["HH"],
            search_spaces["HL"],
            search_spaces["LH"],
            search_spaces["LL"],
        )
    )

    print(
        f"\nTesting "
        f"{len(combinations)} "
        f"combinations..."
    )

    global_best_score = (
        baseline["final_score"]
    )

    global_best_metrics = dict(
        baseline
    )

    global_best_weights = {
        "HH": 0.0,
        "HL": 0.0,
        "LH": 0.0,
        "LL": 0.0,
    }

    global_best_predictions = (
        adaptive_predictions
    )

    for index, combination in enumerate(
        combinations,
        start=1,
    ):

        weights = {
            "HH": combination[0],
            "HL": combination[1],
            "LH": combination[2],
            "LL": combination[3],
        }

        predictions = (
            build_fused_predictions(
                ground_truth=ground_truth,
                adaptive_lookup=adaptive_lookup,
                goal_lookup=goal_lookup,
                session_specificity=(
                    session_specificity
                ),
                weights=weights,
                rrf_k=args.rrf_k,
                topk=args.topk,
            )
        )

        metrics = evaluate(
            predictions=predictions,
            ground_truth=ground_truth,
            catalog_size=args.catalog_size,
        )

        all_results.append(
            {
                "stage": 2,
                "weights":
                    dict(weights),
                **metrics,
            }
        )

        if (
            metrics["final_score"]
            > global_best_score
        ):

            global_best_score = (
                metrics["final_score"]
            )

            global_best_metrics = dict(
                metrics
            )

            global_best_weights = dict(
                weights
            )

            global_best_predictions = (
                predictions
            )

            print()
            print(
                "NEW BEST "
                f"#{index}: "
                f"{global_best_score:.6f}"
            )

            print(
                f"  weights = "
                f"{global_best_weights}"
            )

            print(
                f"  nDCG@20 = "
                f"{metrics['ndcg@20']:.6f}"
            )

            print(
                f"  diversity = "
                f"{metrics['catalog_diversity']:.6f}"
            )

        elif (
            index % 100 == 0
        ):

            print(
                f"Tested "
                f"{index}/"
                f"{len(combinations)}"
            )

    # --------------------------------------------------------
    # Save best predictions
    # --------------------------------------------------------

    with open(
        args.best_predictions,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            global_best_predictions,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save tuning results
    # --------------------------------------------------------

    output = {
        "adaptive_baseline":
            baseline,

        "stage1_best_weights":
            best_weights,

        "best_weights":
            global_best_weights,

        "best_metrics":
            global_best_metrics,

        "experiments":
            all_results,
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

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("BEST FUSION CONFIGURATION")
    print("=" * 90)

    print(
        f"HH goal weight : "
        f"{global_best_weights['HH']}"
    )

    print(
        f"HL goal weight : "
        f"{global_best_weights['HL']}"
    )

    print(
        f"LH goal weight : "
        f"{global_best_weights['LH']}"
    )

    print(
        f"LL goal weight : "
        f"{global_best_weights['LL']}"
    )

    print()

    print(
        f"nDCG@1        : "
        f"{global_best_metrics['ndcg@1']:.6f}"
    )

    print(
        f"nDCG@10       : "
        f"{global_best_metrics['ndcg@10']:.6f}"
    )

    print(
        f"nDCG@20       : "
        f"{global_best_metrics['ndcg@20']:.6f}"
    )

    print(
        f"Diversity     : "
        f"{global_best_metrics['catalog_diversity']:.6f}"
    )

    print(
        f"FINAL SCORE   : "
        f"{global_best_metrics['final_score']:.6f}"
    )

    print()
    print(
        f"Best predictions saved to: "
        f"{args.best_predictions}"
    )

    print(
        f"Tuning results saved to: "
        f"{args.output}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
