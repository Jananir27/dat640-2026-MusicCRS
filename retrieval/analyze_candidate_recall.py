"""Analyze candidate recall for MusicCRS retrieval systems.

Compares:
    1. Goal-aware Hybrid
    2. Adaptive Goal-aware Hybrid
    3. TRUE candidate-pool union of Goal-aware + Adaptive

Important:
    Union Recall@K means:
        ground-truth track appears in Goal top-K
        OR
        ground-truth track appears in Adaptive top-K.

This is different from concatenating the two rankings and taking
the first K items.

Reports Recall@1, @5, @10, @20 globally and separately for
HH, HL, LH and LL specificity groups.

No retrieval is performed.
No GPU is required.
"""

import argparse
import json
from collections import defaultdict

from datasets import load_dataset


DEFAULT_DATASET = "talkpl-ai/TalkPlayData-Challenge-Dataset"

SPECIFICITIES = [
    "HH",
    "HL",
    "LH",
    "LL",
]

K_VALUES = [
    1,
    5,
    10,
    20,
]


# ============================================================
# Utilities
# ============================================================

def load_json(path):
    """Load JSON file."""

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def make_lookup(predictions):
    """Create (session_id, turn_number) -> track IDs mapping."""

    return {
        (
            p["session_id"],
            p["turn_number"],
        ): p["predicted_track_ids"]
        for p in predictions
    }


# ============================================================
# Specificity mapping
# ============================================================

def load_specificities(
    dataset_name,
    split,
):
    """Create session_id -> specificity mapping."""

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

        mapping[
            item["session_id"]
        ] = goal.get(
            "specificity",
            "LL",
        )

    return mapping


# ============================================================
# Recall helpers
# ============================================================

def hit_at_k(
    tracks,
    ground_truth_track,
    k,
):
    """Return 1 if ground truth occurs within top-k."""

    return int(
        ground_truth_track
        in tracks[:k]
    )


def union_hit_at_k(
    adaptive_tracks,
    goal_tracks,
    ground_truth_track,
    k,
):
    """Return 1 if either system retrieves GT within top-k.

    This is the TRUE candidate-pool union recall.
    """

    return int(
        ground_truth_track
        in adaptive_tracks[:k]
        or
        ground_truth_track
        in goal_tracks[:k]
    )


def reciprocal_rank(
    tracks,
    ground_truth_track,
):
    """Return reciprocal rank of ground truth."""

    try:

        rank = (
            tracks.index(
                ground_truth_track
            )
            + 1
        )

        return 1.0 / rank

    except ValueError:

        return 0.0


def union_reciprocal_rank(
    adaptive_tracks,
    goal_tracks,
    ground_truth_track,
):
    """Best reciprocal rank achieved by either system."""

    adaptive_rr = reciprocal_rank(
        adaptive_tracks,
        ground_truth_track,
    )

    goal_rr = reciprocal_rank(
        goal_tracks,
        ground_truth_track,
    )

    return max(
        adaptive_rr,
        goal_rr,
    )


# ============================================================
# Unique candidate pool
# ============================================================

def build_candidate_union(
    adaptive_tracks,
    goal_tracks,
):
    """Return unique candidates from both top lists.

    Used only to calculate average candidate-pool size.
    Ranking order is irrelevant here.
    """

    return list(
        dict.fromkeys(
            adaptive_tracks
            + goal_tracks
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze MusicCRS "
            "candidate recall"
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
        "--output",
        default=(
            "results_candidate_recall.json"
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 95)
    print(
        "MusicCRS TRUE CANDIDATE "
        "RECALL ANALYSIS"
    )
    print("=" * 95)

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    print()
    print("Loading predictions...")

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
    # Validation
    # --------------------------------------------------------

    if (
        len(goal_predictions)
        != len(ground_truth)
    ):
        raise ValueError(
            "Goal prediction count "
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

    goal_lookup = make_lookup(
        goal_predictions
    )

    adaptive_lookup = make_lookup(
        adaptive_predictions
    )

    # --------------------------------------------------------
    # Load specificity
    # --------------------------------------------------------

    print()
    print("Loading specificity...")

    specificity_map = (
        load_specificities(
            args.dialogue_dataset,
            args.split,
        )
    )

    # --------------------------------------------------------
    # Statistics containers
    # --------------------------------------------------------

    methods = [
        "goal",
        "adaptive",
        "union",
    ]

    groups = [
        "ALL",
        *SPECIFICITIES,
    ]

    stats = {
        group: {
            method: defaultdict(float)
            for method in methods
        }
        for group in groups
    }

    counts = {
        group: 0
        for group in groups
    }

    # Extra complementarity counts
    complementarity = {
        "both": 0,
        "goal_only": 0,
        "adaptive_only": 0,
        "neither": 0,
    }

    complementarity_by_spec = {
        spec: {
            "both": 0,
            "goal_only": 0,
            "adaptive_only": 0,
            "neither": 0,
        }
        for spec in SPECIFICITIES
    }

    missing = 0

    # --------------------------------------------------------
    # Analyze turns
    # --------------------------------------------------------

    for gt in ground_truth:

        session_id = gt[
            "session_id"
        ]

        turn_number = gt[
            "turn_number"
        ]

        ground_truth_track = gt[
            "ground_truth_track_id"
        ]

        key = (
            session_id,
            turn_number,
        )

        if (
            key not in goal_lookup
            or key not in adaptive_lookup
        ):

            missing += 1
            continue

        goal_tracks = goal_lookup[
            key
        ]

        adaptive_tracks = adaptive_lookup[
            key
        ]

        union_candidates = (
            build_candidate_union(
                adaptive_tracks,
                goal_tracks,
            )
        )

        specificity = (
            specificity_map.get(
                session_id,
                "LL",
            )
        )

        if specificity not in SPECIFICITIES:
            specificity = "LL"

        current_groups = [
            "ALL",
            specificity,
        ]

        # ----------------------------------------------------
        # Goal/adaptive hits @20 for complementarity
        # ----------------------------------------------------

        goal_hit20 = (
            ground_truth_track
            in goal_tracks[:20]
        )

        adaptive_hit20 = (
            ground_truth_track
            in adaptive_tracks[:20]
        )

        if (
            goal_hit20
            and adaptive_hit20
        ):

            status = "both"

        elif goal_hit20:

            status = "goal_only"

        elif adaptive_hit20:

            status = "adaptive_only"

        else:

            status = "neither"

        complementarity[
            status
        ] += 1

        complementarity_by_spec[
            specificity
        ][
            status
        ] += 1

        # ----------------------------------------------------
        # Update metrics
        # ----------------------------------------------------

        for group in current_groups:

            counts[group] += 1

            # ================================================
            # Goal-aware
            # ================================================

            for k in K_VALUES:

                stats[
                    group
                ][
                    "goal"
                ][
                    f"recall@{k}"
                ] += hit_at_k(
                    goal_tracks,
                    ground_truth_track,
                    k,
                )

            stats[
                group
            ][
                "goal"
            ][
                "mrr"
            ] += reciprocal_rank(
                goal_tracks,
                ground_truth_track,
            )

            stats[
                group
            ][
                "goal"
            ][
                "candidate_count"
            ] += len(
                goal_tracks
            )

            # ================================================
            # Adaptive
            # ================================================

            for k in K_VALUES:

                stats[
                    group
                ][
                    "adaptive"
                ][
                    f"recall@{k}"
                ] += hit_at_k(
                    adaptive_tracks,
                    ground_truth_track,
                    k,
                )

            stats[
                group
            ][
                "adaptive"
            ][
                "mrr"
            ] += reciprocal_rank(
                adaptive_tracks,
                ground_truth_track,
            )

            stats[
                group
            ][
                "adaptive"
            ][
                "candidate_count"
            ] += len(
                adaptive_tracks
            )

            # ================================================
            # TRUE UNION
            # ================================================

            for k in K_VALUES:

                stats[
                    group
                ][
                    "union"
                ][
                    f"recall@{k}"
                ] += union_hit_at_k(
                    adaptive_tracks,
                    goal_tracks,
                    ground_truth_track,
                    k,
                )

            stats[
                group
            ][
                "union"
            ][
                "mrr"
            ] += union_reciprocal_rank(
                adaptive_tracks,
                goal_tracks,
                ground_truth_track,
            )

            stats[
                group
            ][
                "union"
            ][
                "candidate_count"
            ] += len(
                union_candidates
            )

    # --------------------------------------------------------
    # Convert totals to averages
    # --------------------------------------------------------

    results = {}

    for group in groups:

        total = counts[group]

        if total == 0:
            continue

        results[group] = {
            "turns": total,
        }

        for method in methods:

            result = {}

            for k in K_VALUES:

                result[
                    f"recall@{k}"
                ] = (
                    stats[
                        group
                    ][
                        method
                    ][
                        f"recall@{k}"
                    ]
                    / total
                )

            result[
                "mrr"
            ] = (
                stats[
                    group
                ][
                    method
                ][
                    "mrr"
                ]
                / total
            )

            result[
                "avg_candidates"
            ] = (
                stats[
                    group
                ][
                    method
                ][
                    "candidate_count"
                ]
                / total
            )

            results[
                group
            ][
                method
            ] = result

    # --------------------------------------------------------
    # Global results
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("GLOBAL CANDIDATE RECALL")
    print("=" * 100)

    print(
        f"{'Method':<15}"
        f"{'R@1':<12}"
        f"{'R@5':<12}"
        f"{'R@10':<12}"
        f"{'R@20':<12}"
        f"{'MRR':<12}"
        f"{'Avg Cand.':<12}"
    )

    print("-" * 100)

    for method in methods:

        r = results[
            "ALL"
        ][
            method
        ]

        print(
            f"{method:<15}"
            f"{r['recall@1']:<12.6f}"
            f"{r['recall@5']:<12.6f}"
            f"{r['recall@10']:<12.6f}"
            f"{r['recall@20']:<12.6f}"
            f"{r['mrr']:<12.6f}"
            f"{r['avg_candidates']:<12.2f}"
        )

    # --------------------------------------------------------
    # Recall by specificity
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("TRUE RECALL@20 BY SPECIFICITY")
    print("=" * 100)

    print(
        f"{'Spec':<10}"
        f"{'Turns':<10}"
        f"{'Goal R@20':<18}"
        f"{'Adaptive R@20':<20}"
        f"{'True Union R@20':<20}"
    )

    print("-" * 100)

    for spec in SPECIFICITIES:

        r = results[
            spec
        ]

        print(
            f"{spec:<10}"
            f"{r['turns']:<10}"
            f"{r['goal']['recall@20']:<18.6f}"
            f"{r['adaptive']['recall@20']:<20.6f}"
            f"{r['union']['recall@20']:<20.6f}"
        )

    # --------------------------------------------------------
    # Complementarity
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("TOP-20 COMPLEMENTARITY")
    print("=" * 100)

    total = counts["ALL"]

    for status in [
        "both",
        "goal_only",
        "adaptive_only",
        "neither",
    ]:

        value = complementarity[
            status
        ]

        percentage = (
            value / total
        )

        print(
            f"{status:<15}: "
            f"{value:<6} "
            f"({percentage:.4%})"
        )

    # --------------------------------------------------------
    # Complementarity by specificity
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("COMPLEMENTARITY BY SPECIFICITY")
    print("=" * 100)

    print(
        f"{'Spec':<8}"
        f"{'Both':<12}"
        f"{'Goal only':<14}"
        f"{'Adaptive only':<17}"
        f"{'Neither':<12}"
    )

    print("-" * 100)

    for spec in SPECIFICITIES:

        c = complementarity_by_spec[
            spec
        ]

        print(
            f"{spec:<8}"
            f"{c['both']:<12}"
            f"{c['goal_only']:<14}"
            f"{c['adaptive_only']:<17}"
            f"{c['neither']:<12}"
        )

    # --------------------------------------------------------
    # Important summary
    # --------------------------------------------------------

    goal_r20 = results[
        "ALL"
    ][
        "goal"
    ][
        "recall@20"
    ]

    adaptive_r20 = results[
        "ALL"
    ][
        "adaptive"
    ][
        "recall@20"
    ]

    union_r20 = results[
        "ALL"
    ][
        "union"
    ][
        "recall@20"
    ]

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        f"Goal-aware Recall@20 : "
        f"{goal_r20:.6f}"
    )

    print(
        f"Adaptive Recall@20   : "
        f"{adaptive_r20:.6f}"
    )

    print(
        f"TRUE Union Recall@20 : "
        f"{union_r20:.6f}"
    )

    print()

    print(
        "Union gain vs Goal-aware : "
        f"{union_r20 - goal_r20:+.6f}"
    )

    print(
        "Union gain vs Adaptive   : "
        f"{union_r20 - adaptive_r20:+.6f}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {
        "results":
            results,

        "complementarity":
            complementarity,

        "complementarity_by_specificity":
            complementarity_by_spec,

        "missing_predictions":
            missing,
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
        f"Results saved to: "
        f"{args.output}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
