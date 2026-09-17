"""Tune exact metadata reranking weights by goal specificity.

Starting point:
    predictions_best_fusion.json

Specificity groups:
    HH, HL, LH, LL

For each group, tests different ExactMetadataMatcher weights while
leaving the other groups unchanged.

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


DEFAULT_DATASET = "talkpl-ai/TalkPlayData-Challenge-Dataset"

SPECIFICITIES = ["HH", "HL", "LH", "LL"]

# Small weights are most interesting because 0.05 already worked for HH.
WEIGHTS = [
    0.0,
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
]


# ============================================================
# Utilities
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_session_info(dataset_name, split):
    """Load specificity and conversation for every session."""

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

        info[item["session_id"]] = {
            "specificity": goal.get(
                "specificity",
                "LL",
            ),
            "conversations": item[
                "conversations"
            ],
        }

    return info


def get_user_message(
    conversations,
    turn_number,
):
    """Get user message corresponding to target turn."""

    for message in conversations:

        if (
            message["turn_number"] == turn_number
            and message["role"] == "user"
        ):
            return str(
                message["content"]
            ).strip()

    return ""


# ============================================================
# Precompute exact scores
# ============================================================

def precompute_exact_scores(
    predictions,
    session_info,
    matcher,
):
    """Calculate exact scores once.

    This avoids repeatedly calling the matcher during tuning.
    """

    print()
    print("Precomputing exact metadata scores...")

    cache = {}

    total = len(predictions)

    for index, prediction in enumerate(
        predictions,
        start=1,
    ):

        session_id = prediction[
            "session_id"
        ]

        turn_number = prediction[
            "turn_number"
        ]

        info = session_info.get(
            session_id
        )

        if info is None:
            continue

        query = get_user_message(
            info["conversations"],
            turn_number,
        )

        track_scores = {}

        for track_id in prediction[
            "predicted_track_ids"
        ]:

            track_scores[track_id] = (
                matcher.score_track(
                    query,
                    track_id,
                )
            )

        cache[
            (
                session_id,
                turn_number,
            )
        ] = track_scores

        if index % 1000 == 0:
            print(
                f"Processed "
                f"{index}/{total}"
            )

    print(
        f"Exact-score cache created "
        f"for {len(cache)} turns."
    )

    return cache


# ============================================================
# Reranking
# ============================================================

def rerank_tracks(
    track_ids,
    exact_scores,
    weight,
):
    """Rerank one existing candidate list."""

    if weight == 0.0:
        return list(track_ids)

    candidate_count = len(
        track_ids
    )

    scored = []

    for rank, track_id in enumerate(
        track_ids,
        start=1,
    ):

        # Same base-ranking logic as ExactMetadataMatcher.rerank().
        base_score = (
            candidate_count
            - rank
            + 1
        ) / candidate_count

        exact_score = exact_scores.get(
            track_id,
            0.0,
        )

        final_score = (
            base_score
            + weight * exact_score
        )

        scored.append(
            (
                track_id,
                final_score,
                rank,
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
        for track_id, _, _ in scored
    ]


def apply_weights(
    predictions,
    session_info,
    score_cache,
    weights,
):
    """Apply specificity-dependent exact weights."""

    output = []

    changed_by_spec = {
        spec: 0
        for spec in SPECIFICITIES
    }

    for prediction in predictions:

        session_id = prediction[
            "session_id"
        ]

        turn_number = prediction[
            "turn_number"
        ]

        original = prediction[
            "predicted_track_ids"
        ]

        info = session_info.get(
            session_id
        )

        if info is None:
            specificity = "LL"
        else:
            specificity = info[
                "specificity"
            ]

        if specificity not in SPECIFICITIES:
            specificity = "LL"

        weight = weights[
            specificity
        ]

        scores = score_cache.get(
            (
                session_id,
                turn_number,
            ),
            {},
        )

        reranked = rerank_tracks(
            original,
            scores,
            weight,
        )

        if reranked != original:
            changed_by_spec[
                specificity
            ] += 1

        output.append(
            {
                "session_id":
                    session_id,

                "turn_number":
                    turn_number,

                "predicted_track_ids":
                    reranked,

                "predicted_response":
                    prediction.get(
                        "predicted_response",
                        "",
                    ),
            }
        )

    return (
        output,
        changed_by_spec,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Tune exact metadata weights "
            "by specificity"
        )
    )

    parser.add_argument(
        "--predictions",
        default="predictions_best_fusion.json",
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
            "results_exact_specificity_tuning.json"
        ),
    )

    parser.add_argument(
        "--best_predictions",
        default=(
            "predictions_exact_specificity_best.json"
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 90)
    print(
        "MusicCRS EXACT-MATCH "
        "SPECIFICITY TUNING"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

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

    if len(predictions) != len(
        ground_truth
    ):
        raise ValueError(
            "Prediction and ground-truth "
            "counts do not match."
        )

    session_info = load_session_info(
        args.dialogue_dataset,
        args.split,
    )

    # --------------------------------------------------------
    # Matcher
    # --------------------------------------------------------

    print()
    print("Loading catalog...")

    catalog = MusicCatalogLoader()

    matcher = ExactMetadataMatcher(
        catalog=catalog
    )

    score_cache = (
        precompute_exact_scores(
            predictions,
            session_info,
            matcher,
        )
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline = evaluate(
        predictions=predictions,
        ground_truth=ground_truth,
        catalog_size=args.catalog_size,
    )

    print()
    print("=" * 90)
    print("BASELINE")
    print("=" * 90)

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

    # --------------------------------------------------------
    # Tune
    # --------------------------------------------------------

    # HH already has a proven best value of 0.05.
    # Keep it fixed and tune HL/LH/LL.
    hh_weight = 0.05

    combinations = list(
        itertools.product(
            WEIGHTS,
            WEIGHTS,
            WEIGHTS,
        )
    )

    print()
    print(
        f"Testing {len(combinations)} "
        f"combinations..."
    )

    print(
        f"HH fixed at: {hh_weight}"
    )

    best_score = baseline[
        "final_score"
    ]

    best_weights = {
        "HH": 0.0,
        "HL": 0.0,
        "LH": 0.0,
        "LL": 0.0,
    }

    best_metrics = dict(
        baseline
    )

    best_predictions = list(
        predictions
    )

    best_changed = {}

    tuning_results = []

    for index, (
        hl_weight,
        lh_weight,
        ll_weight,
    ) in enumerate(
        combinations,
        start=1,
    ):

        weights = {
            "HH": hh_weight,
            "HL": hl_weight,
            "LH": lh_weight,
            "LL": ll_weight,
        }

        (
            candidate_predictions,
            changed,
        ) = apply_weights(
            predictions,
            session_info,
            score_cache,
            weights,
        )

        metrics = evaluate(
            predictions=candidate_predictions,
            ground_truth=ground_truth,
            catalog_size=args.catalog_size,
        )

        tuning_results.append(
            {
                "weights": dict(
                    weights
                ),
                "changed_turns":
                    dict(changed),
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

            best_weights = dict(
                weights
            )

            best_metrics = dict(
                metrics
            )

            best_predictions = (
                candidate_predictions
            )

            best_changed = dict(
                changed
            )

            print()
            print(
                f"NEW BEST #{index}: "
                f"{best_score:.6f}"
            )

            print(
                f"  weights = "
                f"{best_weights}"
            )

            print(
                f"  nDCG@1  = "
                f"{best_metrics['ndcg@1']:.6f}"
            )

            print(
                f"  nDCG@10 = "
                f"{best_metrics['ndcg@10']:.6f}"
            )

            print(
                f"  nDCG@20 = "
                f"{best_metrics['ndcg@20']:.6f}"
            )

        if index % 50 == 0:
            print(
                f"Tested "
                f"{index}/"
                f"{len(combinations)}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

    output = {
        "baseline":
            baseline,

        "best_weights":
            best_weights,

        "best_metrics":
            best_metrics,

        "best_changed_turns":
            best_changed,

        "experiments":
            tuning_results,
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
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "BEST SPECIFICITY-AWARE "
        "EXACT CONFIGURATION"
    )
    print("=" * 90)

    print(
        f"HH weight : "
        f"{best_weights['HH']}"
    )

    print(
        f"HL weight : "
        f"{best_weights['HL']}"
    )

    print(
        f"LH weight : "
        f"{best_weights['LH']}"
    )

    print(
        f"LL weight : "
        f"{best_weights['LL']}"
    )

    print()

    print(
        f"nDCG@1    : "
        f"{best_metrics['ndcg@1']:.6f}"
    )

    print(
        f"nDCG@10   : "
        f"{best_metrics['ndcg@10']:.6f}"
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

    print()

    print(
        f"Changed turns: "
        f"{best_changed}"
    )

    print()
    print(
        f"Best predictions: "
        f"{args.best_predictions}"
    )

    print(
        f"Results: "
        f"{args.output}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
