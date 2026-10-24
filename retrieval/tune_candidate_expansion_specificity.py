"""Specificity-aware candidate expansion tuning for MusicCRS.

Idea
----
Different conversation-goal specificity classes should not necessarily
receive the same amount of diversity expansion.

HH: highly specific -> preserve relevance
HL: moderately specific
LH: moderately exploratory
LL: vague/exploratory -> allow more diversity

Official score:
    final_score = 0.8 * nDCG@20 + 0.2 * catalog_diversity
"""

import argparse
import itertools
import json
import math
from collections import Counter

from datasets import load_dataset

from .evaluation.evaluate import evaluate


# ============================================================
# Constants
# ============================================================

SPECIFICITIES = [
    "HH",
    "HL",
    "LH",
    "LL",
]

FINAL_TOPK = 20

TARGET_SCORE = 0.2235


# ============================================================
# JSON
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ============================================================
# Dataset specificity
# ============================================================

def load_specificity(
    dataset_name,
    split,
):
    """Return session_id -> specificity."""

    print("Loading dialogue dataset...")

    dataset = load_dataset(
        dataset_name,
        split=split,
    )

    mapping = {}

    counts = Counter()

    for item in dataset:

        goal = (
            item.get("conversation_goal")
            or {}
        )

        specificity = goal.get(
            "specificity",
            "LL",
        )

        if specificity not in SPECIFICITIES:
            specificity = "LL"

        session_id = item["session_id"]

        mapping[session_id] = specificity

        counts[specificity] += 1

    print()
    print("Sessions by specificity:")

    for spec in SPECIFICITIES:
        print(
            f"  {spec}: {counts[spec]}"
        )

    return mapping


# ============================================================
# Prediction helpers
# ============================================================

def build_prediction_map(predictions):
    """Index predictions by session and turn."""

    return {
        (
            prediction["session_id"],
            prediction["turn_number"],
        ): prediction
        for prediction in predictions
    }


def calculate_frequency(predictions):
    """Count how often every track occurs in baseline top-20."""

    frequency = Counter()

    for prediction in predictions:

        for track_id in prediction[
            "predicted_track_ids"
        ]:

            frequency[track_id] += 1

    return frequency


# ============================================================
# Novelty
# ============================================================

def novelty_score(
    track_id,
    frequency,
    total_turns,
):
    """Normalized inverse-frequency novelty."""

    count = frequency.get(
        track_id,
        0,
    )

    numerator = math.log(
        (total_turns + 1)
        / (count + 1)
    )

    denominator = math.log(
        total_turns + 1
    )

    if denominator <= 0:
        return 0.0

    return numerator / denominator


# ============================================================
# Expansion
# ============================================================

def expand_prediction(
    base_tracks,
    reservoir_tracks,
    frequency,
    total_turns,
    protect_top,
    replacements,
    novelty_weight,
    reservoir_start=21,
    reservoir_end=100,
):
    """Replace lower-ranked tracks with novel reservoir candidates."""

    base_tracks = list(base_tracks)

    reservoir_tracks = list(
        reservoir_tracks
    )

    if not base_tracks:
        return []

    final_size = len(base_tracks)

    protect_top = min(
        protect_top,
        final_size,
    )

    replacements = min(
        replacements,
        final_size - protect_top,
    )

    if replacements <= 0:
        return base_tracks

    # --------------------------------------------------------
    # Protected high-ranking tracks
    # --------------------------------------------------------

    protected = base_tracks[
        :protect_top
    ]

    lower_original = base_tracks[
        protect_top:
    ]

    keep_count = (
        len(lower_original)
        - replacements
    )

    original_to_keep = lower_original[
        :keep_count
    ]

    # --------------------------------------------------------
    # Reservoir candidates
    # --------------------------------------------------------

    base_set = set(
        base_tracks
    )

    start_index = max(
        reservoir_start - 1,
        0,
    )

    end_index = min(
        reservoir_end,
        len(reservoir_tracks),
    )

    candidates = []

    seen = set()

    for rank_index in range(
        start_index,
        end_index,
    ):

        track_id = reservoir_tracks[
            rank_index
        ]

        if track_id in base_set:
            continue

        if track_id in seen:
            continue

        seen.add(track_id)

        reservoir_rank = (
            rank_index + 1
        )

        # Retrieval relevance
        relevance = (
            21.0
            / max(
                reservoir_rank,
                21,
            )
        )

        # Catalog novelty
        novelty = novelty_score(
            track_id,
            frequency,
            total_turns,
        )

        score = (
            relevance
            + novelty_weight * novelty
        )

        candidates.append(
            (
                track_id,
                score,
                reservoir_rank,
            )
        )

    candidates.sort(
        key=lambda x: (
            -x[1],
            x[2],
        )
    )

    selected = [
        track_id
        for track_id, _, _
        in candidates[:replacements]
    ]

    # --------------------------------------------------------
    # Final ranking
    # --------------------------------------------------------

    result = (
        list(protected)
        + list(original_to_keep)
        + selected
    )

    result_set = set(result)

    # Fallback if insufficient reservoir candidates
    if len(result) < final_size:

        for track_id in lower_original:

            if track_id in result_set:
                continue

            result.append(track_id)

            result_set.add(track_id)

            if len(result) >= final_size:
                break

    return result[:final_size]


# ============================================================
# Apply specificity configuration
# ============================================================

def apply_specificity_configuration(
    base_predictions,
    reservoir_map,
    specificity_map,
    frequency,
    replacement_config,
    novelty_weight,
    protect_top=10,
):
    """Apply different replacement counts to HH/HL/LH/LL."""

    output = []

    total_turns = len(
        base_predictions
    )

    changed_by_spec = {
        spec: 0
        for spec in SPECIFICITIES
    }

    inserted_by_spec = {
        spec: 0
        for spec in SPECIFICITIES
    }

    for prediction in base_predictions:

        session_id = prediction[
            "session_id"
        ]

        turn_number = prediction[
            "turn_number"
        ]

        specificity = (
            specificity_map.get(
                session_id,
                "LL",
            )
        )

        replacements = (
            replacement_config[
                specificity
            ]
        )

        key = (
            session_id,
            turn_number,
        )

        original = list(
            prediction[
                "predicted_track_ids"
            ]
        )

        reservoir_prediction = (
            reservoir_map.get(key)
        )

        if reservoir_prediction is None:

            reranked = original

        else:

            reranked = expand_prediction(
                base_tracks=original,
                reservoir_tracks=(
                    reservoir_prediction[
                        "predicted_track_ids"
                    ]
                ),
                frequency=frequency,
                total_turns=total_turns,
                protect_top=protect_top,
                replacements=replacements,
                novelty_weight=novelty_weight,
            )

        if reranked != original:

            changed_by_spec[
                specificity
            ] += 1

        original_set = set(original)

        inserted_by_spec[
            specificity
        ] += sum(
            1
            for track_id in reranked
            if track_id
            not in original_set
        )

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
        changed_by_spec,
        inserted_by_spec,
    )


# ============================================================
# Assignment points
# ============================================================

def calculate_points(
    final_score,
):
    baseline = 0.1444
    target = 0.2235

    points = (
        10.0
        * (
            final_score
            - baseline
        )
        / (
            target
            - baseline
        )
    )

    return max(
        0.0,
        min(
            10.0,
            points,
        ),
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Tune specificity-aware candidate "
            "expansion for MusicCRS"
        )
    )

    parser.add_argument(
        "--predictions",
        default=(
            "predictions_diversity_best.json"
        ),
    )

    parser.add_argument(
        "--reservoir",
        default=(
            "predictions_adaptive_goal_top100_full.json"
        ),
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
            "results_candidate_expansion_specificity.json"
        ),
    )

    parser.add_argument(
        "--best_predictions",
        default=(
            "predictions_candidate_expansion_specificity_best.json"
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 100)
    print(
        "SPECIFICITY-AWARE CANDIDATE EXPANSION"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    base_predictions = load_json(
        args.predictions
    )

    reservoir_predictions = load_json(
        args.reservoir
    )

    ground_truth = load_json(
        args.ground_truth
    )

    print()
    print(
        f"Base predictions : "
        f"{len(base_predictions)}"
    )

    print(
        f"Reservoir        : "
        f"{len(reservoir_predictions)}"
    )

    print(
        f"Ground truth     : "
        f"{len(ground_truth)}"
    )

    if (
        len(base_predictions)
        != len(ground_truth)
    ):
        raise ValueError(
            "Predictions and ground truth "
            "have different lengths."
        )

    reservoir_map = (
        build_prediction_map(
            reservoir_predictions
        )
    )

    specificity_map = (
        load_specificity(
            args.dialogue_dataset,
            args.split,
        )
    )

    frequency = calculate_frequency(
        base_predictions
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline = evaluate(
        predictions=base_predictions,
        ground_truth=ground_truth,
        catalog_size=args.catalog_size,
    )

    print()
    print("=" * 100)
    print("INPUT BASELINE")
    print("=" * 100)

    print(
        f"nDCG@20   : "
        f"{baseline['ndcg@20']:.6f}"
    )

    print(
        f"Diversity : "
        f"{baseline['catalog_diversity']:.6f}"
    )

    print(
        f"FINAL     : "
        f"{baseline['final_score']:.6f}"
    )

    # --------------------------------------------------------
    # Search space
    # --------------------------------------------------------
    #
    # HH gets conservative replacement values.
    # LL gets more aggressive replacement values.
    #
    # Current global winner = 7 replacements / novelty 5.
    # Search around that point.
    # --------------------------------------------------------

    replacement_values = {
        "HH": [
            2,
            3,
            4,
            5,
            6,
            7,
        ],

        "HL": [
            4,
            5,
            6,
            7,
            8,
        ],

        "LH": [
            4,
            5,
            6,
            7,
            8,
        ],

        "LL": [
            5,
            6,
            7,
            8,
            9,
            10,
        ],
    }

    novelty_weights = [
        3.0,
        4.0,
        5.0,
        6.0,
        7.5,
    ]

    protect_top = 10

    # --------------------------------------------------------
    # Instead of full Cartesian search:
    #
    # First tune each specificity coordinate while keeping
    # the others at the current global winner (7).
    #
    # Repeat passes until no improvement.
    #
    # This is dramatically cheaper than testing every
    # combination.
    # --------------------------------------------------------

    current_config = {
        "HH": 7,
        "HL": 7,
        "LH": 7,
        "LL": 7,
    }

    current_novelty = 5.0

    (
        current_predictions,
        _,
        _,
    ) = apply_specificity_configuration(
        base_predictions=base_predictions,
        reservoir_map=reservoir_map,
        specificity_map=specificity_map,
        frequency=frequency,
        replacement_config=current_config,
        novelty_weight=current_novelty,
        protect_top=protect_top,
    )

    current_metrics = evaluate(
        predictions=current_predictions,
        ground_truth=ground_truth,
        catalog_size=args.catalog_size,
    )

    best_predictions = (
        current_predictions
    )

    best_metrics = (
        current_metrics
    )

    best_config = dict(
        current_config
    )

    best_novelty = (
        current_novelty
    )

    experiments = []

    print()
    print("=" * 100)
    print("STARTING COORDINATE SEARCH")
    print("=" * 100)

    print()
    print(
        "Initial configuration:"
    )

    print(
        f"  replacements = "
        f"{current_config}"
    )

    print(
        f"  novelty      = "
        f"{current_novelty}"
    )

    print(
        f"  final        = "
        f"{current_metrics['final_score']:.6f}"
    )

    # ========================================================
    # Coordinate passes
    # ========================================================

    max_passes = 3

    experiment_number = 0

    for pass_number in range(
        1,
        max_passes + 1,
    ):

        print()
        print("=" * 100)
        print(
            f"PASS {pass_number}"
        )
        print("=" * 100)

        pass_improved = False

        # ----------------------------------------------------
        # Tune replacement count for each specificity
        # ----------------------------------------------------

        for specificity in SPECIFICITIES:

            print()
            print(
                f"Tuning {specificity} "
                f"replacement count..."
            )

            local_best_score = (
                best_metrics[
                    "final_score"
                ]
            )

            local_best_value = (
                best_config[
                    specificity
                ]
            )

            local_best_predictions = (
                best_predictions
            )

            local_best_metrics = (
                best_metrics
            )

            for value in (
                replacement_values[
                    specificity
                ]
            ):

                candidate_config = dict(
                    best_config
                )

                candidate_config[
                    specificity
                ] = value

                (
                    predictions,
                    changed,
                    inserted,
                ) = (
                    apply_specificity_configuration(
                        base_predictions=base_predictions,
                        reservoir_map=reservoir_map,
                        specificity_map=specificity_map,
                        frequency=frequency,
                        replacement_config=candidate_config,
                        novelty_weight=best_novelty,
                        protect_top=protect_top,
                    )
                )

                metrics = evaluate(
                    predictions=predictions,
                    ground_truth=ground_truth,
                    catalog_size=args.catalog_size,
                )

                experiment_number += 1

                print(
                    f"[{experiment_number:03d}] "
                    f"{specificity}={value} "
                    f"novelty={best_novelty:.1f} "
                    f"nDCG20="
                    f"{metrics['ndcg@20']:.6f} "
                    f"div="
                    f"{metrics['catalog_diversity']:.6f} "
                    f"final="
                    f"{metrics['final_score']:.6f}"
                )

                experiments.append(
                    {
                        "pass":
                            pass_number,

                        "dimension":
                            specificity,

                        "value":
                            value,

                        "config":
                            candidate_config,

                        "novelty_weight":
                            best_novelty,

                        "metrics":
                            metrics,

                        "changed":
                            changed,

                        "inserted":
                            inserted,
                    }
                )

                if (
                    metrics["final_score"]
                    > local_best_score
                ):

                    local_best_score = (
                        metrics[
                            "final_score"
                        ]
                    )

                    local_best_value = (
                        value
                    )

                    local_best_predictions = (
                        predictions
                    )

                    local_best_metrics = (
                        metrics
                    )

            if (
                local_best_score
                > best_metrics[
                    "final_score"
                ]
            ):

                best_config[
                    specificity
                ] = local_best_value

                best_predictions = (
                    local_best_predictions
                )

                best_metrics = (
                    local_best_metrics
                )

                pass_improved = True

                print()
                print(
                    "*** NEW BEST ***"
                )

                print(
                    f"Config    : "
                    f"{best_config}"
                )

                print(
                    f"Novelty   : "
                    f"{best_novelty}"
                )

                print(
                    f"nDCG@20   : "
                    f"{best_metrics['ndcg@20']:.6f}"
                )

                print(
                    f"Diversity : "
                    f"{best_metrics['catalog_diversity']:.6f}"
                )

                print(
                    f"FINAL     : "
                    f"{best_metrics['final_score']:.6f}"
                )

        # ----------------------------------------------------
        # Tune shared novelty weight
        # ----------------------------------------------------

        print()
        print(
            "Tuning shared novelty weight..."
        )

        local_best_score = (
            best_metrics[
                "final_score"
            ]
        )

        local_best_novelty = (
            best_novelty
        )

        local_best_predictions = (
            best_predictions
        )

        local_best_metrics = (
            best_metrics
        )

        for novelty_weight in novelty_weights:

            (
                predictions,
                changed,
                inserted,
            ) = (
                apply_specificity_configuration(
                    base_predictions=base_predictions,
                    reservoir_map=reservoir_map,
                    specificity_map=specificity_map,
                    frequency=frequency,
                    replacement_config=best_config,
                    novelty_weight=novelty_weight,
                    protect_top=protect_top,
                )
            )

            metrics = evaluate(
                predictions=predictions,
                ground_truth=ground_truth,
                catalog_size=args.catalog_size,
            )

            experiment_number += 1

            print(
                f"[{experiment_number:03d}] "
                f"novelty={novelty_weight:.1f} "
                f"nDCG20="
                f"{metrics['ndcg@20']:.6f} "
                f"div="
                f"{metrics['catalog_diversity']:.6f} "
                f"final="
                f"{metrics['final_score']:.6f}"
            )

            experiments.append(
                {
                    "pass":
                        pass_number,

                    "dimension":
                        "novelty",

                    "value":
                        novelty_weight,

                    "config":
                        dict(best_config),

                    "novelty_weight":
                        novelty_weight,

                    "metrics":
                        metrics,

                    "changed":
                        changed,

                    "inserted":
                        inserted,
                }
            )

            if (
                metrics["final_score"]
                > local_best_score
            ):

                local_best_score = (
                    metrics[
                        "final_score"
                    ]
                )

                local_best_novelty = (
                    novelty_weight
                )

                local_best_predictions = (
                    predictions
                )

                local_best_metrics = (
                    metrics
                )

        if (
            local_best_score
            > best_metrics[
                "final_score"
            ]
        ):

            best_novelty = (
                local_best_novelty
            )

            best_predictions = (
                local_best_predictions
            )

            best_metrics = (
                local_best_metrics
            )

            pass_improved = True

            print()
            print(
                "*** NEW BEST NOVELTY ***"
            )

            print(
                f"Config    : "
                f"{best_config}"
            )

            print(
                f"Novelty   : "
                f"{best_novelty}"
            )

            print(
                f"FINAL     : "
                f"{best_metrics['final_score']:.6f}"
            )

        if not pass_improved:

            print()
            print(
                "No improvement in this pass."
            )

            break

    # ========================================================
    # Save
    # ========================================================

    save_json(
        best_predictions,
        args.best_predictions,
    )

    points = calculate_points(
        best_metrics[
            "final_score"
        ]
    )

    output = {
        "best_configuration": {
            "protect_top":
                protect_top,

            "replacement_by_specificity":
                best_config,

            "novelty_weight":
                best_novelty,
        },

        "best_metrics":
            best_metrics,

        "estimated_points":
            points,

        "target_score":
            TARGET_SCORE,

        "experiments":
            experiments,
    }

    save_json(
        output,
        args.output,
    )

    # ========================================================
    # Report
    # ========================================================

    print()
    print("=" * 100)
    print(
        "BEST SPECIFICITY-AWARE CANDIDATE EXPANSION"
    )
    print("=" * 100)

    print(
        f"Protect top : "
        f"{protect_top}"
    )

    print()

    print("Replacement counts:")

    for spec in SPECIFICITIES:

        print(
            f"  {spec}: "
            f"{best_config[spec]}"
        )

    print()

    print(
        f"Novelty weight : "
        f"{best_novelty}"
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
        f"Estimated points : "
        f"{points:.2f}/10"
    )

    if (
        best_metrics[
            "final_score"
        ]
        >= TARGET_SCORE
    ):

        print(
            "TARGET STATUS   : "
            "FULL-MARK TARGET REACHED"
        )

    else:

        gap = (
            TARGET_SCORE
            - best_metrics[
                "final_score"
            ]
        )

        print(
            "TARGET STATUS   : "
            "NOT YET REACHED"
        )

        print(
            f"Remaining gap   : "
            f"{gap:.6f}"
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

    print("=" * 100)


if __name__ == "__main__":
    main()
