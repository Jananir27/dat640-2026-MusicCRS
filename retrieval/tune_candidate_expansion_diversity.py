"""Extended candidate-expansion diversity tuning for MusicCRS.

This script starts from the strongest existing top-20 predictions and
uses a top-100 candidate reservoir to introduce novel tracks into the
lower part of the ranking.

Previous best:
    protect_top    = 10
    replacements   = 5
    novelty_weight = 2.0

    nDCG@20        = 0.101056
    diversity      = 0.653545
    final_score    = 0.211554

Official score:
    final_score = 0.8 * nDCG@20 + 0.2 * catalog_diversity

Target:
    final_score >= 0.2235
"""

import argparse
import json
import math
from collections import Counter

from .evaluation.evaluate import evaluate


# ======================================================================
# Constants
# ======================================================================

FINAL_TOPK = 20

SPECIFIC_TARGET = 0.2235


# ======================================================================
# JSON utilities
# ======================================================================

def load_json(path):
    """Load JSON file."""

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_json(data, path):
    """Save JSON file."""

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
        )


# ======================================================================
# Prediction indexing
# ======================================================================

def prediction_map(predictions):
    """Index predictions by (session_id, turn_number)."""

    return {
        (
            prediction["session_id"],
            prediction["turn_number"],
        ): prediction
        for prediction in predictions
    }


# ======================================================================
# Frequency / novelty
# ======================================================================

def calculate_frequency(predictions):
    """Count frequency of tracks in current recommendations."""

    frequency = Counter()

    for prediction in predictions:

        track_ids = prediction[
            "predicted_track_ids"
        ]

        for track_id in track_ids:
            frequency[track_id] += 1

    return frequency


def novelty_score(
    track_id,
    frequency,
    total_turns,
):
    """Calculate normalized inverse-frequency novelty.

    A track that appears very frequently receives a lower novelty
    score.

    A track that never appears in the baseline receives a score
    close to 1.
    """

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


# ======================================================================
# Reservoir candidate extraction
# ======================================================================

def get_expansion_candidates(
    base_tracks,
    reservoir_tracks,
    reservoir_start,
    reservoir_end,
):
    """Get candidate tracks from the larger reservoir.

    Tracks already present in the original top-20 are excluded.
    """

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

    for track_id in reservoir_tracks[
        start_index:end_index
    ]:

        if track_id in base_set:
            continue

        if track_id in seen:
            continue

        seen.add(
            track_id
        )

        candidates.append(
            track_id
        )

    return candidates


# ======================================================================
# Candidate scoring
# ======================================================================

def expansion_score(
    track_id,
    reservoir_rank,
    frequency,
    total_turns,
    novelty_weight,
):
    """Score one expansion candidate.

    Score combines:

        1. Retrieval relevance represented by reservoir rank.
        2. Global recommendation novelty.

    Earlier reservoir ranks receive stronger relevance scores.
    Rare tracks receive stronger novelty scores.
    """

    relevance_score = (
        21.0
        / float(
            max(
                reservoir_rank,
                21,
            )
        )
    )

    novelty = novelty_score(
        track_id=track_id,
        frequency=frequency,
        total_turns=total_turns,
    )

    score = (
        relevance_score
        + novelty_weight * novelty
    )

    return score


# ======================================================================
# Expand one prediction
# ======================================================================

def expand_one_prediction(
    base_tracks,
    reservoir_tracks,
    frequency,
    total_turns,
    protect_top,
    replacements,
    novelty_weight,
    reservoir_start,
    reservoir_end,
):
    """Expand one top-20 prediction using reservoir candidates.

    Example:

        protect_top = 10
        replacements = 5

    Ranks 1-10 are always preserved.

    Five tracks from the lower portion of the ranking are replaced
    using candidates from reservoir ranks 21-100.
    """

    final_size = len(
        base_tracks
    )

    if final_size == 0:
        return []

    protect_top = min(
        protect_top,
        final_size,
    )

    maximum_replacements = (
        final_size
        - protect_top
    )

    replacements = min(
        replacements,
        maximum_replacements,
    )

    if replacements <= 0:
        return list(
            base_tracks
        )

    # ------------------------------------------------------------------
    # Protect top ranking
    # ------------------------------------------------------------------

    protected_tracks = list(
        base_tracks[:protect_top]
    )

    # ------------------------------------------------------------------
    # Original lower-ranking tracks
    # ------------------------------------------------------------------

    lower_original = list(
        base_tracks[protect_top:]
    )

    keep_original_count = (
        len(lower_original)
        - replacements
    )

    original_to_keep = list(
        lower_original[
            :keep_original_count
        ]
    )

    # ------------------------------------------------------------------
    # Expansion candidates
    # ------------------------------------------------------------------

    expansion_candidates = (
        get_expansion_candidates(
            base_tracks=base_tracks,
            reservoir_tracks=reservoir_tracks,
            reservoir_start=reservoir_start,
            reservoir_end=reservoir_end,
        )
    )

    reservoir_rank_map = {
        track_id: rank
        for rank, track_id in enumerate(
            reservoir_tracks,
            start=1,
        )
    }

    scored_candidates = []

    for track_id in expansion_candidates:

        reservoir_rank = (
            reservoir_rank_map[
                track_id
            ]
        )

        score = expansion_score(
            track_id=track_id,
            reservoir_rank=reservoir_rank,
            frequency=frequency,
            total_turns=total_turns,
            novelty_weight=novelty_weight,
        )

        scored_candidates.append(
            (
                track_id,
                score,
                reservoir_rank,
            )
        )

    # Higher score first.
    # Earlier reservoir rank breaks ties.

    scored_candidates.sort(
        key=lambda item: (
            -item[1],
            item[2],
        )
    )

    selected_new_tracks = [
        track_id
        for track_id, _, _
        in scored_candidates[
            :replacements
        ]
    ]

    # ------------------------------------------------------------------
    # Build final ranking
    # ------------------------------------------------------------------

    result = (
        protected_tracks
        + original_to_keep
        + selected_new_tracks
    )

    result_set = set(
        result
    )

    # ------------------------------------------------------------------
    # Fallback if reservoir cannot provide enough unique tracks
    # ------------------------------------------------------------------

    if len(result) < final_size:

        for track_id in lower_original:

            if track_id in result_set:
                continue

            result.append(
                track_id
            )

            result_set.add(
                track_id
            )

            if len(result) >= final_size:
                break

    # ------------------------------------------------------------------
    # Defensive fill
    # ------------------------------------------------------------------

    if len(result) < final_size:

        for track_id in base_tracks:

            if track_id in result_set:
                continue

            result.append(
                track_id
            )

            result_set.add(
                track_id
            )

            if len(result) >= final_size:
                break

    return result[
        :final_size
    ]


# ======================================================================
# Apply configuration
# ======================================================================

def apply_configuration(
    base_predictions,
    reservoir_map,
    frequency,
    protect_top,
    replacements,
    novelty_weight,
    reservoir_start,
    reservoir_end,
):
    """Apply one candidate-expansion configuration globally."""

    output = []

    changed_turns = 0

    new_track_insertions = 0

    total_turns = len(
        base_predictions
    )

    for prediction in base_predictions:

        session_id = prediction[
            "session_id"
        ]

        turn_number = prediction[
            "turn_number"
        ]

        key = (
            session_id,
            turn_number,
        )

        base_tracks = list(
            prediction[
                "predicted_track_ids"
            ]
        )

        reservoir_prediction = (
            reservoir_map.get(
                key
            )
        )

        if reservoir_prediction is None:

            final_tracks = (
                base_tracks
            )

        else:

            reservoir_tracks = list(
                reservoir_prediction[
                    "predicted_track_ids"
                ]
            )

            final_tracks = (
                expand_one_prediction(
                    base_tracks=base_tracks,
                    reservoir_tracks=reservoir_tracks,
                    frequency=frequency,
                    total_turns=total_turns,
                    protect_top=protect_top,
                    replacements=replacements,
                    novelty_weight=novelty_weight,
                    reservoir_start=reservoir_start,
                    reservoir_end=reservoir_end,
                )
            )

        if final_tracks != base_tracks:
            changed_turns += 1

        base_set = set(
            base_tracks
        )

        new_track_insertions += sum(
            1
            for track_id in final_tracks
            if track_id not in base_set
        )

        new_prediction = dict(
            prediction
        )

        new_prediction[
            "predicted_track_ids"
        ] = final_tracks

        output.append(
            new_prediction
        )

    return (
        output,
        changed_turns,
        new_track_insertions,
    )


# ======================================================================
# Print metrics
# ======================================================================

def print_metrics(metrics):
    """Print evaluation metrics."""

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
        f"FINAL      : "
        f"{metrics['final_score']:.6f}"
    )


# ======================================================================
# Main
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extended candidate-expansion "
            "diversity tuning for MusicCRS"
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
        "--catalog_size",
        type=int,
        default=47071,
    )

    parser.add_argument(
        "--output",
        default=(
            "results_candidate_expansion_extended.json"
        ),
    )

    parser.add_argument(
        "--best_predictions",
        default=(
            "predictions_candidate_expansion_extended_best.json"
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 100)
    print(
        "MusicCRS EXTENDED CANDIDATE-EXPANSION TUNING"
    )
    print("=" * 100)

    # ==================================================================
    # Load files
    # ==================================================================

    print()
    print("Loading files...")

    base_predictions = load_json(
        args.predictions
    )

    reservoir_predictions = load_json(
        args.reservoir
    )

    ground_truth = load_json(
        args.ground_truth
    )

    print(
        f"Base predictions      : "
        f"{len(base_predictions)}"
    )

    print(
        f"Reservoir predictions : "
        f"{len(reservoir_predictions)}"
    )

    print(
        f"Ground truth          : "
        f"{len(ground_truth)}"
    )

    # ==================================================================
    # Validation
    # ==================================================================

    if (
        len(base_predictions)
        != len(ground_truth)
    ):
        raise ValueError(
            "Base predictions and ground truth "
            "have different lengths."
        )

    reservoir_map = prediction_map(
        reservoir_predictions
    )

    missing_reservoir = 0

    for prediction in base_predictions:

        key = (
            prediction["session_id"],
            prediction["turn_number"],
        )

        if key not in reservoir_map:
            missing_reservoir += 1

    print(
        f"Missing reservoir turns : "
        f"{missing_reservoir}"
    )

    # ==================================================================
    # Baseline evaluation
    # ==================================================================

    baseline = evaluate(
        predictions=base_predictions,
        ground_truth=ground_truth,
        catalog_size=args.catalog_size,
    )

    print()
    print("=" * 100)
    print("BASELINE")
    print("=" * 100)

    print_metrics(
        baseline
    )

    print()
    print(
        f"Full-mark target : "
        f"{SPECIFIC_TARGET:.6f}"
    )

    # ==================================================================
    # Frequency
    # ==================================================================

    print()
    print(
        "Calculating baseline track frequency..."
    )

    frequency = calculate_frequency(
        base_predictions
    )

    print(
        f"Unique baseline tracks : "
        f"{len(frequency)}"
    )

    # ==================================================================
    # EXTENDED SEARCH SPACE
    # ==================================================================
    #
    # Previous best:
    #
    #   protect_top    = 10
    #   replacements   = 5
    #   novelty_weight = 2.0
    #
    #   nDCG@20        = 0.101056
    #   diversity      = 0.653545
    #   final          = 0.211554
    #
    # The optimum occurred at the previous upper boundary for
    # replacements and novelty weight.
    #
    # We therefore extend both dimensions.
    #
    # protect_top remains >= 10 to preserve the strongest part
    # of the ranking.
    # ==================================================================

    protect_values = [
        10,
        11,
        12,
        13,
        14,
        15,
    ]

    replacement_values = [
        5,
        6,
        7,
        8,
        9,
        10,
    ]

    novelty_weights = [
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
        7.5,
        10.0,
    ]

    reservoir_start = 21

    reservoir_end = 100

    # ==================================================================
    # Build valid configurations
    # ==================================================================

    valid_experiments = []

    for protect_top in protect_values:

        for replacements in replacement_values:

            # Only positions after protect_top
            # may be replaced.
            if (
                protect_top
                + replacements
                > FINAL_TOPK
            ):
                continue

            for novelty_weight in novelty_weights:

                valid_experiments.append(
                    (
                        protect_top,
                        replacements,
                        novelty_weight,
                    )
                )

    total_experiments = len(
        valid_experiments
    )

    print()
    print("=" * 100)
    print("EXTENDED SEARCH")
    print("=" * 100)

    print(
        f"Experiments       : "
        f"{total_experiments}"
    )

    print(
        f"Reservoir ranks   : "
        f"{reservoir_start}-"
        f"{reservoir_end}"
    )

    print(
        f"Target final score: "
        f"{SPECIFIC_TARGET:.6f}"
    )

    # ==================================================================
    # Initial best = baseline
    # ==================================================================

    best_score = baseline[
        "final_score"
    ]

    best_metrics = dict(
        baseline
    )

    best_predictions = list(
        base_predictions
    )

    best_config = {
        "protect_top": None,
        "replacements": 0,
        "novelty_weight": 0.0,
        "changed_turns": 0,
        "new_track_insertions": 0,
    }

    experiments = []

    # ==================================================================
    # Search
    # ==================================================================

    for experiment_number, configuration in enumerate(
        valid_experiments,
        start=1,
    ):

        (
            protect_top,
            replacements,
            novelty_weight,
        ) = configuration

        (
            candidate_predictions,
            changed_turns,
            new_track_insertions,
        ) = apply_configuration(
            base_predictions=base_predictions,
            reservoir_map=reservoir_map,
            frequency=frequency,
            protect_top=protect_top,
            replacements=replacements,
            novelty_weight=novelty_weight,
            reservoir_start=reservoir_start,
            reservoir_end=reservoir_end,
        )

        metrics = evaluate(
            predictions=candidate_predictions,
            ground_truth=ground_truth,
            catalog_size=args.catalog_size,
        )

        experiment = {
            "protect_top":
                protect_top,

            "replacements":
                replacements,

            "novelty_weight":
                novelty_weight,

            "changed_turns":
                changed_turns,

            "new_track_insertions":
                new_track_insertions,

            **metrics,
        }

        experiments.append(
            experiment
        )

        print(
            f"[{experiment_number:03d}/"
            f"{total_experiments:03d}] "
            f"protect={protect_top:2d} "
            f"replace={replacements:2d} "
            f"novelty={novelty_weight:4.1f} "
            f"nDCG20="
            f"{metrics['ndcg@20']:.6f} "
            f"div="
            f"{metrics['catalog_diversity']:.6f} "
            f"final="
            f"{metrics['final_score']:.6f}"
        )

        # ==============================================================
        # New best
        # ==============================================================

        if (
            metrics["final_score"]
            > best_score
        ):

            best_score = (
                metrics["final_score"]
            )

            best_metrics = dict(
                metrics
            )

            best_predictions = (
                candidate_predictions
            )

            best_config = {
                "protect_top":
                    protect_top,

                "replacements":
                    replacements,

                "novelty_weight":
                    novelty_weight,

                "changed_turns":
                    changed_turns,

                "new_track_insertions":
                    new_track_insertions,
            }

            print()
            print(
                "*** NEW BEST ***"
            )

            print(
                f"Protect top          : "
                f"{protect_top}"
            )

            print(
                f"Replacements         : "
                f"{replacements}"
            )

            print(
                f"Novelty weight       : "
                f"{novelty_weight}"
            )

            print(
                f"Changed turns        : "
                f"{changed_turns}"
            )

            print(
                f"New track insertions : "
                f"{new_track_insertions}"
            )

            print(
                f"nDCG@20              : "
                f"{best_metrics['ndcg@20']:.6f}"
            )

            print(
                f"Diversity             : "
                f"{best_metrics['catalog_diversity']:.6f}"
            )

            print(
                f"FINAL                 : "
                f"{best_score:.6f}"
            )

            # ----------------------------------------------------------
            # Assignment points
            # ----------------------------------------------------------

            baseline_score = 0.1444
            target_score = 0.2235

            points = (
                10.0
                * (
                    best_score
                    - baseline_score
                )
                / (
                    target_score
                    - baseline_score
                )
            )

            points = max(
                0.0,
                min(
                    10.0,
                    points,
                ),
            )

            print(
                f"Estimated points      : "
                f"{points:.2f}/10"
            )

            if (
                best_score
                >= SPECIFIC_TARGET
            ):

                print()
                print(
                    ">>> FULL-MARK TARGET REACHED <<<"
                )

            print()

    # ==================================================================
    # Save best predictions
    # ==================================================================

    save_json(
        best_predictions,
        args.best_predictions,
    )

    # ==================================================================
    # Assignment points
    # ==================================================================

    assignment_baseline = 0.1444

    assignment_target = 0.2235

    estimated_points = (
        10.0
        * (
            best_score
            - assignment_baseline
        )
        / (
            assignment_target
            - assignment_baseline
        )
    )

    estimated_points = max(
        0.0,
        min(
            10.0,
            estimated_points,
        ),
    )

    # ==================================================================
    # Save results
    # ==================================================================

    output_results = {
        "baseline":
            baseline,

        "best_configuration":
            best_config,

        "best_metrics":
            best_metrics,

        "estimated_points":
            estimated_points,

        "target_score":
            SPECIFIC_TARGET,

        "experiments":
            experiments,
    }

    save_json(
        output_results,
        args.output,
    )

    # ==================================================================
    # Final report
    # ==================================================================

    print()
    print("=" * 100)
    print(
        "BEST EXTENDED CANDIDATE-EXPANSION CONFIGURATION"
    )
    print("=" * 100)

    if (
        best_config[
            "protect_top"
        ]
        is None
    ):

        print(
            "No extended configuration "
            "beat the input baseline."
        )

    else:

        print(
            f"Protect top          : "
            f"{best_config['protect_top']}"
        )

        print(
            f"Replacements         : "
            f"{best_config['replacements']}"
        )

        print(
            f"Novelty weight       : "
            f"{best_config['novelty_weight']}"
        )

        print(
            f"Changed turns        : "
            f"{best_config['changed_turns']}"
        )

        print(
            f"New track insertions : "
            f"{best_config['new_track_insertions']}"
        )

    print()

    print_metrics(
        best_metrics
    )

    print()

    print(
        f"Estimated points : "
        f"{estimated_points:.2f}/10"
    )

    print(
        f"Target score     : "
        f"{SPECIFIC_TARGET:.6f}"
    )

    if (
        best_score
        >= SPECIFIC_TARGET
    ):

        print(
            "Target status    : "
            "FULL-MARK TARGET REACHED"
        )

    else:

        difference = (
            SPECIFIC_TARGET
            - best_score
        )

        print(
            "Target status    : "
            "NOT YET REACHED"
        )

        print(
            f"Remaining gap    : "
            f"{difference:.6f}"
        )

    print()

    print(
        f"Best predictions : "
        f"{args.best_predictions}"
    )

    print(
        f"Results          : "
        f"{args.output}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
