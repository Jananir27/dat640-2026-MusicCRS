"""Rerank union of Goal-aware and Adaptive candidate lists.

Uses existing prediction files:
    predictions_goal_aware_hybrid_full.json
    predictions_adaptive_goal_full.json

Pipeline:
    Goal-aware top-20 ----\
                           -> union candidates
    Adaptive top-20 ------/
                               |
                               v
                    specificity-aware scoring
                               |
                               v
                            top-20

Signals:
    1. Goal-aware reciprocal rank
    2. Adaptive reciprocal rank
    3. Exact metadata matching

No BM25 retrieval.
No dense retrieval.
No GPU required.
"""

import argparse
import itertools
import json

from datasets import load_dataset

from .data_loader import MusicCatalogLoader
from .exact_match import ExactMetadataMatcher
from .evaluation.evaluate import evaluate


DEFAULT_DATASET = (
    "talkpl-ai/"
    "TalkPlayData-Challenge-Dataset"
)

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
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def prediction_map(predictions):
    """Index predictions by session + turn."""

    return {
        (
            p["session_id"],
            p["turn_number"],
        ): p
        for p in predictions
    }


# ============================================================
# Dataset information
# ============================================================

def load_session_info(
    dataset_name,
    split,
):
    """Load specificity and conversation information."""

    print("Loading dialogue dataset...")

    dataset = load_dataset(
        dataset_name,
        split=split,
    )

    info = {}

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

        info[item["session_id"]] = {
            "specificity":
                specificity,

            "conversations":
                item["conversations"],

            "conversation_goal":
                goal,
        }

    return info


def get_current_user_message(
    conversations,
    turn_number,
):
    """Get user message for target turn."""

    for message in conversations:

        if (
            message.get("turn_number")
            == turn_number
            and message.get("role")
            == "user"
        ):

            return str(
                message.get(
                    "content",
                    "",
                )
            ).strip()

    return ""


# ============================================================
# Candidate union
# ============================================================

def make_union(
    goal_tracks,
    adaptive_tracks,
):
    """Create order-preserving union."""

    seen = set()
    union = []

    for track_id in (
        list(goal_tracks)
        + list(adaptive_tracks)
    ):

        if track_id not in seen:

            seen.add(
                track_id
            )

            union.append(
                track_id
            )

    return union


# ============================================================
# Precompute turn information
# ============================================================

def precompute_turn_data(
    goal_predictions,
    adaptive_predictions,
    session_info,
    matcher,
):
    """Precompute ranks, union candidates and exact scores."""

    print()
    print(
        "Precomputing union candidates "
        "and exact scores..."
    )

    goal_map = prediction_map(
        goal_predictions
    )

    adaptive_map = prediction_map(
        adaptive_predictions
    )

    turn_data = {}

    keys = list(
        goal_map.keys()
    )

    total = len(keys)

    for index, key in enumerate(
        keys,
        start=1,
    ):

        if key not in adaptive_map:
            raise KeyError(
                f"Missing adaptive prediction: {key}"
            )

        session_id, turn_number = key

        goal_tracks = goal_map[
            key
        ]["predicted_track_ids"]

        adaptive_tracks = adaptive_map[
            key
        ]["predicted_track_ids"]

        union_tracks = make_union(
            goal_tracks,
            adaptive_tracks,
        )

        goal_ranks = {
            track_id: rank
            for rank, track_id in enumerate(
                goal_tracks,
                start=1,
            )
        }

        adaptive_ranks = {
            track_id: rank
            for rank, track_id in enumerate(
                adaptive_tracks,
                start=1,
            )
        }

        info = session_info.get(
            session_id,
            {
                "specificity": "LL",
                "conversations": [],
            },
        )

        specificity = info[
            "specificity"
        ]

        query = get_current_user_message(
            info.get(
                "conversations",
                [],
            ),
            turn_number,
        )

        exact_scores = {}

        for track_id in union_tracks:

            exact_scores[
                track_id
            ] = matcher.score_track(
                query,
                track_id,
            )

        turn_data[key] = {
            "specificity":
                specificity,

            "union":
                union_tracks,

            "goal_ranks":
                goal_ranks,

            "adaptive_ranks":
                adaptive_ranks,

            "exact_scores":
                exact_scores,
        }

        if index % 1000 == 0:

            print(
                f"Processed "
                f"{index}/{total}"
            )

    print(
        f"Prepared {len(turn_data)} turns."
    )

    return turn_data


# ============================================================
# Reranking
# ============================================================

def rerank_turn(
    data,
    adaptive_weight,
    exact_weight,
    rrf_k=60,
    topk=20,
):
    """Rerank one union candidate list.

    Score:

        goal reciprocal rank
        +
        adaptive_weight * adaptive reciprocal rank
        +
        exact_weight * normalized exact score
    """

    scored = []

    union_tracks = data[
        "union"
    ]

    goal_ranks = data[
        "goal_ranks"
    ]

    adaptive_ranks = data[
        "adaptive_ranks"
    ]

    exact_scores = data[
        "exact_scores"
    ]

    for original_rank, track_id in enumerate(
        union_tracks,
        start=1,
    ):

        score = 0.0

        # ----------------------------------------------------
        # Goal-aware ranking signal
        # ----------------------------------------------------

        goal_rank = goal_ranks.get(
            track_id
        )

        if goal_rank is not None:

            score += (
                1.0
                / (
                    rrf_k
                    + goal_rank
                )
            )

        # ----------------------------------------------------
        # Adaptive ranking signal
        # ----------------------------------------------------

        adaptive_rank = (
            adaptive_ranks.get(
                track_id
            )
        )

        if adaptive_rank is not None:

            score += (
                adaptive_weight
                / (
                    rrf_k
                    + adaptive_rank
                )
            )

        # ----------------------------------------------------
        # Exact metadata signal
        #
        # Exact matcher scores can be much larger than RRF,
        # so normalize to roughly the same scale.
        # ----------------------------------------------------

        exact_score = (
            exact_scores.get(
                track_id,
                0.0,
            )
        )

        if exact_score > 0:

            normalized_exact = (
                exact_score
                / 15.0
            )

            score += (
                exact_weight
                * normalized_exact
                / rrf_k
            )

        scored.append(
            (
                track_id,
                score,
                original_rank,
            )
        )

    scored.sort(
        key=lambda x: (
            -x[1],
            x[2],
        )
    )

    return [
        track_id
        for track_id, _, _ in scored[
            :topk
        ]
    ]


# ============================================================
# Build predictions
# ============================================================

def build_predictions(
    ground_truth,
    turn_data,
    adaptive_weights,
    exact_weights,
    rrf_k,
    topk,
):
    """Build predictions using specificity-aware weights."""

    predictions = []

    changed = {
        spec: 0
        for spec in SPECIFICITIES
    }

    for gt in ground_truth:

        key = (
            gt["session_id"],
            gt["turn_number"],
        )

        if key not in turn_data:

            raise KeyError(
                f"Missing turn data: {key}"
            )

        data = turn_data[
            key
        ]

        specificity = data[
            "specificity"
        ]

        ranking = rerank_turn(
            data=data,

            adaptive_weight=
                adaptive_weights[
                    specificity
                ],

            exact_weight=
                exact_weights[
                    specificity
                ],

            rrf_k=rrf_k,
            topk=topk,
        )

        # Compare against original goal ranking.
        original_goal = list(
            data["goal_ranks"].keys()
        )[:topk]

        if ranking != original_goal:

            changed[
                specificity
            ] += 1

        predictions.append(
            {
                "session_id":
                    gt["session_id"],

                "turn_number":
                    gt["turn_number"],

                "predicted_track_ids":
                    ranking,

                "predicted_response":
                    "",
            }
        )

    return (
        predictions,
        changed,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Tune specificity-aware "
            "Goal + Adaptive union reranking"
        )
    )

    parser.add_argument(
        "--goal_predictions",
        default=(
            "predictions_goal_aware_hybrid_full.json"
        ),
    )

    parser.add_argument(
        "--adaptive_predictions",
        default=(
            "predictions_adaptive_goal_full.json"
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
        default=(
            "results_union_reranking.json"
        ),
    )

    parser.add_argument(
        "--best_predictions",
        default=(
            "predictions_union_reranked_best.json"
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 95)
    print(
        "MusicCRS UNION CANDIDATE RERANKING"
    )
    print("=" * 95)

    # ========================================================
    # Load files
    # ========================================================

    print()
    print("Loading prediction files...")

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

    # ========================================================
    # Dataset
    # ========================================================

    session_info = load_session_info(
        args.dialogue_dataset,
        args.split,
    )

    # ========================================================
    # Exact matcher
    # ========================================================

    print()
    print("Loading music catalog...")

    catalog = MusicCatalogLoader()

    matcher = ExactMetadataMatcher(
        catalog=catalog
    )

    # ========================================================
    # Precompute
    # ========================================================

    turn_data = precompute_turn_data(
        goal_predictions=
            goal_predictions,

        adaptive_predictions=
            adaptive_predictions,

        session_info=
            session_info,

        matcher=
            matcher,
    )

    # ========================================================
    # Baseline
    # ========================================================

    print()
    print("=" * 95)
    print("GOAL-AWARE BASELINE")
    print("=" * 95)

    baseline = evaluate(
        predictions=
            goal_predictions,

        ground_truth=
            ground_truth,

        catalog_size=
            args.catalog_size,
    )

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
        f"Final      : "
        f"{baseline['final_score']:.6f}"
    )

    # ========================================================
    # Tuning grids
    # ========================================================

    # Adaptive contribution.
    adaptive_grid = [
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
    ]

    # Exact metadata contribution.
    exact_grid = [
        0.0,
        0.025,
        0.05,
        0.075,
        0.10,
    ]

    # --------------------------------------------------------
    # To avoid an enormous 25^4 search, tune one specificity
    # group at a time using coordinate search.
    # --------------------------------------------------------

    adaptive_weights = {
        "HH": 0.5,
        "HL": 0.5,
        "LH": 0.5,
        "LL": 0.5,
    }

    exact_weights = {
        "HH": 0.05,
        "HL": 0.10,
        "LH": 0.10,
        "LL": 0.10,
    }

    best_predictions = None
    best_metrics = None

    tuning_history = []

    # ========================================================
    # Coordinate tuning
    # ========================================================

    print()
    print("=" * 95)
    print(
        "SPECIFICITY-AWARE COORDINATE SEARCH"
    )
    print("=" * 95)

    # Two passes allow groups to adjust after the others change.
    for pass_number in range(
        1,
        3,
    ):

        print()
        print(
            f"PASS {pass_number}"
        )

        for specificity in SPECIFICITIES:

            local_best_score = -1.0
            local_best_adaptive = (
                adaptive_weights[
                    specificity
                ]
            )

            local_best_exact = (
                exact_weights[
                    specificity
                ]
            )

            local_best_metrics = None
            local_best_predictions = None

            print()
            print("-" * 95)
            print(
                f"Tuning {specificity}"
            )
            print("-" * 95)

            for (
                adaptive_weight,
                exact_weight,
            ) in itertools.product(
                adaptive_grid,
                exact_grid,
            ):

                trial_adaptive = dict(
                    adaptive_weights
                )

                trial_exact = dict(
                    exact_weights
                )

                trial_adaptive[
                    specificity
                ] = adaptive_weight

                trial_exact[
                    specificity
                ] = exact_weight

                (
                    predictions,
                    changed,
                ) = build_predictions(
                    ground_truth=
                        ground_truth,

                    turn_data=
                        turn_data,

                    adaptive_weights=
                        trial_adaptive,

                    exact_weights=
                        trial_exact,

                    rrf_k=
                        args.rrf_k,

                    topk=
                        args.topk,
                )

                metrics = evaluate(
                    predictions=
                        predictions,

                    ground_truth=
                        ground_truth,

                    catalog_size=
                        args.catalog_size,
                )

                tuning_history.append(
                    {
                        "pass":
                            pass_number,

                        "specificity":
                            specificity,

                        "adaptive_weight":
                            adaptive_weight,

                        "exact_weight":
                            exact_weight,

                        "adaptive_weights":
                            dict(
                                trial_adaptive
                            ),

                        "exact_weights":
                            dict(
                                trial_exact
                            ),

                        "changed_turns":
                            dict(changed),

                        **metrics,
                    }
                )

                if (
                    metrics[
                        "final_score"
                    ]
                    > local_best_score
                ):

                    local_best_score = (
                        metrics[
                            "final_score"
                        ]
                    )

                    local_best_adaptive = (
                        adaptive_weight
                    )

                    local_best_exact = (
                        exact_weight
                    )

                    local_best_metrics = (
                        dict(metrics)
                    )

                    local_best_predictions = (
                        predictions
                    )

            # -----------------------------------------------
            # Lock best values for this specificity.
            # -----------------------------------------------

            adaptive_weights[
                specificity
            ] = local_best_adaptive

            exact_weights[
                specificity
            ] = local_best_exact

            best_metrics = (
                local_best_metrics
            )

            best_predictions = (
                local_best_predictions
            )

            print(
                f"{specificity} best:"
            )

            print(
                f"  adaptive weight = "
                f"{local_best_adaptive}"
            )

            print(
                f"  exact weight    = "
                f"{local_best_exact}"
            )

            print(
                f"  final score     = "
                f"{local_best_score:.6f}"
            )

    # ========================================================
    # Final evaluation
    # ========================================================

    (
        best_predictions,
        changed,
    ) = build_predictions(
        ground_truth=
            ground_truth,

        turn_data=
            turn_data,

        adaptive_weights=
            adaptive_weights,

        exact_weights=
            exact_weights,

        rrf_k=
            args.rrf_k,

        topk=
            args.topk,
    )

    best_metrics = evaluate(
        predictions=
            best_predictions,

        ground_truth=
            ground_truth,

        catalog_size=
            args.catalog_size,
    )

    # ========================================================
    # Save predictions
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
    # Save results
    # ========================================================

    results = {
        "baseline":
            baseline,

        "best_adaptive_weights":
            adaptive_weights,

        "best_exact_weights":
            exact_weights,

        "best_metrics":
            best_metrics,

        "changed_turns":
            changed,

        "tuning_history":
            tuning_history,
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
    # Report
    # ========================================================

    print()
    print("=" * 95)
    print(
        "BEST UNION RERANKING CONFIGURATION"
    )
    print("=" * 95)

    print()
    print("Adaptive weights:")

    for spec in SPECIFICITIES:

        print(
            f"  {spec}: "
            f"{adaptive_weights[spec]}"
        )

    print()
    print("Exact weights:")

    for spec in SPECIFICITIES:

        print(
            f"  {spec}: "
            f"{exact_weights[spec]}"
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
        f"Changed turns: {changed}"
    )

    print()
    print(
        f"Predictions: "
        f"{args.best_predictions}"
    )

    print(
        f"Results    : "
        f"{args.output}"
    )

    print("=" * 95)


if __name__ == "__main__":
    main()
