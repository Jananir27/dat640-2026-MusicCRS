"""Fine-tune selective candidate expansion for MusicCRS.

The search automatically stops as soon as:

    final_score >= 0.2235

Every new best is saved immediately, so the best result is preserved
even if the program is interrupted.

Official metric:

    final_score =
        0.8 * nDCG@20
        + 0.2 * catalog_diversity
"""

import argparse
import json
import math
import sys
from collections import Counter

from .evaluation.evaluate import evaluate


# =====================================================================
# Constants
# =====================================================================

TARGET_SCORE = 0.2235
BASELINE_SCORE = 0.1444

# Current best from the previous coarse search.
PREVIOUS_BEST_SCORE = 0.22341374098014416


# =====================================================================
# JSON helpers
# =====================================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    """Save JSON while safely converting NumPy scalar types."""

    def json_converter(obj):
        # NumPy integer
        if hasattr(obj, "item"):
            return obj.item()

        raise TypeError(
            f"Object of type {type(obj).__name__} "
            "is not JSON serializable"
        )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            default=json_converter,
        )

# =====================================================================
# Prediction helpers
# =====================================================================

def build_prediction_map(predictions):
    """Index predictions by (session_id, turn_number)."""

    return {
        (
            prediction["session_id"],
            prediction["turn_number"],
        ): prediction
        for prediction in predictions
    }


def calculate_frequency(predictions):
    """Count track frequency in base predictions."""

    frequency = Counter()

    for prediction in predictions:

        for track_id in prediction[
            "predicted_track_ids"
        ]:
            frequency[track_id] += 1

    return frequency


# =====================================================================
# Novelty
# =====================================================================

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


# =====================================================================
# Retrieval relevance
# =====================================================================

def rank_relevance(rank):
    """Convert retrieval rank to relevance."""

    if rank <= 0:
        return 0.0

    return (
        1.0
        / math.log2(rank + 1)
    )


# =====================================================================
# Selective reranking
# =====================================================================

def selective_rerank(
    base_tracks,
    reservoir_tracks,
    frequency,
    total_turns,
    protect_top,
    novelty_weight,
    original_bonus,
    reservoir_start,
    reservoir_end,
):
    """Selectively rerank lower positions.

    The first `protect_top` tracks are preserved.

    Existing lower-ranked tracks compete against tracks from the
    top-100 reservoir.

    Candidate score:

        rank relevance
        + novelty_weight * novelty
        + original_bonus if already in original top-20
    """

    base_tracks = list(
        base_tracks
    )

    reservoir_tracks = list(
        reservoir_tracks
    )

    final_size = len(
        base_tracks
    )

    if final_size == 0:
        return []

    protect_top = min(
        protect_top,
        final_size,
    )

    protected = list(
        base_tracks[:protect_top]
    )

    protected_set = set(
        protected
    )

    remaining_slots = (
        final_size
        - protect_top
    )

    if remaining_slots <= 0:
        return protected

    # -----------------------------------------------------------------
    # Reservoir rank lookup
    # -----------------------------------------------------------------

    reservoir_rank = {}

    for rank, track_id in enumerate(
        reservoir_tracks,
        start=1,
    ):

        if track_id not in reservoir_rank:

            reservoir_rank[
                track_id
            ] = rank

    # -----------------------------------------------------------------
    # Original track rank lookup
    # -----------------------------------------------------------------

    base_set = set(
        base_tracks
    )

    base_rank = {
        track_id: rank
        for rank, track_id in enumerate(
            base_tracks,
            start=1,
        )
    }

    # -----------------------------------------------------------------
    # Candidate pool
    # -----------------------------------------------------------------

    candidates = {}

    # Existing lower-ranked tracks.
    for original_rank, track_id in enumerate(
        base_tracks[protect_top:],
        start=protect_top + 1,
    ):

        if track_id in protected_set:
            continue

        retrieval_rank = (
            reservoir_rank.get(
                track_id,
                original_rank,
            )
        )

        relevance = rank_relevance(
            retrieval_rank
        )

        novelty = novelty_score(
            track_id,
            frequency,
            total_turns,
        )

        score = (
            relevance
            + novelty_weight * novelty
            + original_bonus
        )

        candidates[
            track_id
        ] = {
            "score":
                score,

            "relevance":
                relevance,

            "novelty":
                novelty,

            "rank":
                retrieval_rank,

            "original":
                True,

            "original_rank":
                original_rank,
        }

    # -----------------------------------------------------------------
    # Add reservoir candidates
    # -----------------------------------------------------------------

    start_index = max(
        reservoir_start - 1,
        0,
    )

    end_index = min(
        reservoir_end,
        len(reservoir_tracks),
    )

    for index in range(
        start_index,
        end_index,
    ):

        track_id = (
            reservoir_tracks[index]
        )

        if track_id in protected_set:
            continue

        rank = index + 1

        relevance = rank_relevance(
            rank
        )

        novelty = novelty_score(
            track_id,
            frequency,
            total_turns,
        )

        is_original = (
            track_id in base_set
        )

        score = (
            relevance
            + novelty_weight * novelty
        )

        if is_original:
            score += original_bonus

        candidate = {
            "score":
                score,

            "relevance":
                relevance,

            "novelty":
                novelty,

            "rank":
                rank,

            "original":
                is_original,

            "original_rank":
                base_rank.get(
                    track_id,
                    10**9,
                ),
        }

        previous = (
            candidates.get(
                track_id
            )
        )

        if (
            previous is None
            or candidate["score"]
            > previous["score"]
        ):

            candidates[
                track_id
            ] = candidate

    # -----------------------------------------------------------------
    # Select candidates
    # -----------------------------------------------------------------

    ranked = sorted(
        candidates.items(),
        key=lambda item: (
            -item[1]["score"],
            -int(
                item[1]["original"]
            ),
            item[1]["rank"],
            item[1]["original_rank"],
            item[0],
        ),
    )

    selected = [
        track_id
        for track_id, _
        in ranked[
            :remaining_slots
        ]
    ]

    # -----------------------------------------------------------------
    # Final ordering by retrieval evidence
    # -----------------------------------------------------------------

    selected.sort(
        key=lambda track_id: (
            candidates[
                track_id
            ]["rank"],

            candidates[
                track_id
            ]["original_rank"],
        )
    )

    result = (
        protected
        + selected
    )

    # -----------------------------------------------------------------
    # Defensive fallback
    # -----------------------------------------------------------------

    result_set = set(
        result
    )

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

    return result[:final_size]


# =====================================================================
# Apply one configuration
# =====================================================================

def apply_configuration(
    base_predictions,
    reservoir_map,
    frequency,
    protect_top,
    novelty_weight,
    original_bonus,
    reservoir_start,
    reservoir_end,
):
    """Apply one selective-expansion configuration."""

    output = []

    changed_turns = 0
    inserted_tracks = 0

    total_turns = len(
        base_predictions
    )

    for prediction in base_predictions:

        key = (
            prediction[
                "session_id"
            ],
            prediction[
                "turn_number"
            ],
        )

        original = list(
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

            reranked = original

        else:

            reranked = (
                selective_rerank(
                    base_tracks=original,

                    reservoir_tracks=(
                        reservoir_prediction[
                            "predicted_track_ids"
                        ]
                    ),

                    frequency=frequency,

                    total_turns=(
                        total_turns
                    ),

                    protect_top=(
                        protect_top
                    ),

                    novelty_weight=(
                        novelty_weight
                    ),

                    original_bonus=(
                        original_bonus
                    ),

                    reservoir_start=(
                        reservoir_start
                    ),

                    reservoir_end=(
                        reservoir_end
                    ),
                )
            )

        if reranked != original:
            changed_turns += 1

        original_set = set(
            original
        )

        inserted_tracks += sum(
            1
            for track_id in reranked
            if track_id not in original_set
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
        changed_turns,
        inserted_tracks,
    )


# =====================================================================
# Points
# =====================================================================

def calculate_points(
    final_score,
):
    """Estimate assignment points."""

    points = (
        10.0
        * (
            final_score
            - BASELINE_SCORE
        )
        / (
            TARGET_SCORE
            - BASELINE_SCORE
        )
    )

    return max(
        0.0,
        min(
            10.0,
            points,
        ),
    )


# =====================================================================
# Save current best
# =====================================================================

def save_current_best(
    predictions,
    metrics,
    config,
    experiments,
    args,
    target_reached=False,
):
    """Immediately save the current best result."""

    save_json(
        predictions,
        args.best_predictions,
    )

    points = calculate_points(
        metrics["final_score"]
    )

    result = {
        "best_configuration":
            config,

        "best_metrics":
            metrics,

        "estimated_points":
            points,

        "target_score":
            TARGET_SCORE,

        "target_reached":
            target_reached,

        "remaining_gap":
            max(
                0.0,
                TARGET_SCORE
                - metrics[
                    "final_score"
                ],
            ),

        "experiments":
            experiments,
    }

    save_json(
        result,
        args.output,
    )


# =====================================================================
# Print result
# =====================================================================

def print_best_result(
    config,
    metrics,
    args,
    target_reached,
):
    """Print best configuration."""

    points = calculate_points(
        metrics["final_score"]
    )

    print()
    print("=" * 100)

    if target_reached:

        print(
            "TARGET SCORE REACHED - SEARCH STOPPED"
        )

    else:

        print(
            "BEST FINE-TUNED SELECTIVE EXPANSION"
        )

    print("=" * 100)

    print(
        f"Protect top      : "
        f"{config['protect_top']}"
    )

    print(
        f"Reservoir start  : "
        f"{config['reservoir_start']}"
    )

    print(
        f"Reservoir end    : "
        f"{config['reservoir_end']}"
    )

    print(
        f"Novelty weight   : "
        f"{config['novelty_weight']}"
    )

    print(
        f"Original bonus   : "
        f"{config['original_bonus']}"
    )

    print(
        f"Changed turns    : "
        f"{config['changed_turns']}"
    )

    print(
        f"Inserted tracks  : "
        f"{config['inserted_tracks']}"
    )

    print()

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
        f"{metrics['final_score']:.9f}"
    )

    print()

    print(
        f"Estimated points : "
        f"{points:.2f}/10"
    )

    print(
        f"Target score     : "
        f"{TARGET_SCORE:.6f}"
    )

    if target_reached:

        print(
            "TARGET STATUS    : "
            "FULL-MARK TARGET REACHED"
        )

        print(
            f"Amount above target : "
            f"{metrics['final_score'] - TARGET_SCORE:.9f}"
        )

    else:

        print(
            "TARGET STATUS    : "
            "NOT YET REACHED"
        )

        print(
            f"Remaining gap    : "
            f"{TARGET_SCORE - metrics['final_score']:.9f}"
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


# =====================================================================
# Main
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune selective candidate expansion "
            "and stop automatically at target score"
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
            "results_selective_expansion_fine.json"
        ),
    )

    parser.add_argument(
        "--best_predictions",
        default=(
            "predictions_selective_expansion_fine_best.json"
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 100)
    print(
        "SELECTIVE CANDIDATE EXPANSION - FINE SEARCH"
    )
    print("=" * 100)

    print(
        f"Target: FINAL >= "
        f"{TARGET_SCORE:.6f}"
    )

    print(
        "Search will stop automatically "
        "when the target is reached."
    )

    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------

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

    frequency = (
        calculate_frequency(
            base_predictions
        )
    )

    # -----------------------------------------------------------------
    # Baseline
    # -----------------------------------------------------------------

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

    # =================================================================
    # Fine search
    #
    # Centered around:
    #
    # protect_top     = 12
    # reservoir_start = 11
    # novelty_weight  = 1.50
    # original_bonus  = 0.15
    #
    # Previous score:
    # 0.22341374098014416
    # =================================================================

    protect_values = [
        # Search the known-best region first.
        12,
        11,
        13,
    ]

    reservoir_start_values = [
        # Again, known best first.
        11,
        10,
        12,
        9,
        13,
    ]

    novelty_weights = [
        # Fine values around 1.50 first.
        1.50,
        1.45,
        1.55,
        1.40,
        1.60,
        1.35,
        1.65,
        1.30,
        1.70,
        1.20,
        1.80,
    ]

    original_bonuses = [
        # Fine search around 0.15.
        0.150,
        0.145,
        0.155,
        0.140,
        0.160,
        0.135,
        0.165,
        0.130,
        0.170,
        0.120,
        0.180,
        0.110,
        0.190,
        0.100,
        0.200,
    ]

    reservoir_end = 100

    total_experiments = (
        len(protect_values)
        * len(
            reservoir_start_values
        )
        * len(
            novelty_weights
        )
        * len(
            original_bonuses
        )
    )

    print()
    print("=" * 100)
    print("FINE SEARCH SPACE")
    print("=" * 100)

    print(
        f"Protect values    : "
        f"{protect_values}"
    )

    print(
        f"Reservoir starts  : "
        f"{reservoir_start_values}"
    )

    print(
        f"Novelty weights   : "
        f"{novelty_weights}"
    )

    print(
        f"Original bonuses  : "
        f"{original_bonuses}"
    )

    print(
        f"Experiments       : "
        f"{total_experiments}"
    )

    print()

    print(
        f"Previous best     : "
        f"{PREVIOUS_BEST_SCORE:.9f}"
    )

    print(
        f"Target            : "
        f"{TARGET_SCORE:.9f}"
    )

    print(
        f"Gap               : "
        f"{TARGET_SCORE - PREVIOUS_BEST_SCORE:.9f}"
    )

    # -----------------------------------------------------------------
    # Best values
    # -----------------------------------------------------------------

    best_score = (
        baseline[
            "final_score"
        ]
    )

    best_metrics = dict(
        baseline
    )

    best_predictions = (
        base_predictions
    )

    best_config = None

    experiments = []

    experiment_number = 0

    # =================================================================
    # Search
    # =================================================================

    for protect_top in protect_values:

        for reservoir_start in (
            reservoir_start_values
        ):

            for novelty_weight in (
                novelty_weights
            ):

                for original_bonus in (
                    original_bonuses
                ):

                    experiment_number += 1

                    (
                        predictions,
                        changed_turns,
                        inserted_tracks,
                    ) = apply_configuration(
                        base_predictions=(
                            base_predictions
                        ),

                        reservoir_map=(
                            reservoir_map
                        ),

                        frequency=(
                            frequency
                        ),

                        protect_top=(
                            protect_top
                        ),

                        novelty_weight=(
                            novelty_weight
                        ),

                        original_bonus=(
                            original_bonus
                        ),

                        reservoir_start=(
                            reservoir_start
                        ),

                        reservoir_end=(
                            reservoir_end
                        ),
                    )

                    metrics = evaluate(
                        predictions=(
                            predictions
                        ),

                        ground_truth=(
                            ground_truth
                        ),

                        catalog_size=(
                            args.catalog_size
                        ),
                    )

                    config = {
                        "protect_top":
                            protect_top,

                        "reservoir_start":
                            reservoir_start,

                        "reservoir_end":
                            reservoir_end,

                        "novelty_weight":
                            novelty_weight,

                        "original_bonus":
                            original_bonus,

                        "changed_turns":
                            changed_turns,

                        "inserted_tracks":
                            inserted_tracks,
                    }

                    experiment = {
                        **config,
                        **metrics,
                    }

                    experiments.append(
                        experiment
                    )

                    print(
                        f"[{experiment_number:04d}/"
                        f"{total_experiments:04d}] "

                        f"protect="
                        f"{protect_top:2d} "

                        f"start="
                        f"{reservoir_start:2d} "

                        f"nov="
                        f"{novelty_weight:5.3f} "

                        f"bonus="
                        f"{original_bonus:5.3f} "

                        f"n20="
                        f"{metrics['ndcg@20']:.6f} "

                        f"div="
                        f"{metrics['catalog_diversity']:.6f} "

                        f"final="
                        f"{metrics['final_score']:.9f}"
                    )

                    # =================================================
                    # New best
                    # =================================================

                    if (
                        metrics[
                            "final_score"
                        ]
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
                            predictions
                        )

                        best_config = dict(
                            config
                        )

                        target_reached = (
                            best_score
                            >= TARGET_SCORE
                        )

                        # ---------------------------------------------
                        # SAVE IMMEDIATELY
                        # ---------------------------------------------

                        save_current_best(
                            predictions=(
                                best_predictions
                            ),

                            metrics=(
                                best_metrics
                            ),

                            config=(
                                best_config
                            ),

                            experiments=(
                                experiments
                            ),

                            args=args,

                            target_reached=(
                                target_reached
                            ),
                        )

                        print()
                        print(
                            "*** NEW BEST - SAVED ***"
                        )

                        print(
                            f"FINAL : "
                            f"{best_score:.9f}"
                        )

                        print(
                            f"nDCG20: "
                            f"{best_metrics['ndcg@20']:.6f}"
                        )

                        print(
                            f"DIV   : "
                            f"{best_metrics['catalog_diversity']:.6f}"
                        )

                        # =============================================
                        # TARGET REACHED -> SAVE + STOP
                        # =============================================

                        if target_reached:

                            print()
                            print(
                                "############################################"
                            )

                            print(
                                "### TARGET >= 0.223500 REACHED!          ###"
                            )

                            print(
                                "### BEST FILES HAVE BEEN SAVED.          ###"
                            )

                            print(
                                "### STOPPING SEARCH AUTOMATICALLY.       ###"
                            )

                            print(
                                "############################################"
                            )

                            print_best_result(
                                config=(
                                    best_config
                                ),

                                metrics=(
                                    best_metrics
                                ),

                                args=args,

                                target_reached=True,
                            )

                            # Normal successful termination.
                            return

                        else:

                            print(
                                f"Gap   : "
                                f"{TARGET_SCORE - best_score:.9f}"
                            )

                            print()

    # =================================================================
    # Search completed without reaching target
    # =================================================================

    if best_config is None:

        print()
        print(
            "No configuration improved "
            "the input baseline."
        )

        sys.exit(0)

    # Ensure final best is saved.
    save_current_best(
        predictions=(
            best_predictions
        ),

        metrics=(
            best_metrics
        ),

        config=(
            best_config
        ),

        experiments=(
            experiments
        ),

        args=args,

        target_reached=False,
    )

    print_best_result(
        config=(
            best_config
        ),

        metrics=(
            best_metrics
        ),

        args=args,

        target_reached=False,
    )


if __name__ == "__main__":
    main()
