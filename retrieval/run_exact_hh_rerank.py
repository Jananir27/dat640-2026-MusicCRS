"""Tune exact metadata reranking for HH MusicCRS sessions.

Uses existing prediction results and reranks ONLY HH sessions using
ExactMetadataMatcher.

No BM25 retrieval.
No dense retrieval.
No GPU required.

Input:
    predictions_best_fusion.json

Output:
    predictions_exact_hh_best.json
    results_exact_hh_tuning.json
"""

import argparse
import json

from datasets import load_dataset

from .data_loader import MusicCatalogLoader
from .exact_match import ExactMetadataMatcher
from .evaluation.evaluate import evaluate


DEFAULT_DATASET = "talkpl-ai/TalkPlayData-Challenge-Dataset"


# ============================================================
# Utilities
# ============================================================

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Build session information
# ============================================================

def load_session_info(
    dataset_name: str,
    split: str,
):
    """Load specificity and conversations for every session."""

    print("Loading dialogue dataset...")

    dataset = load_dataset(
        dataset_name,
        split=split,
    )

    session_info = {}

    for item in dataset:

        goal = (
            item.get("conversation_goal")
            or {}
        )

        session_info[
            item["session_id"]
        ] = {
            "specificity": goal.get(
                "specificity",
                "LL",
            ),
            "conversations": item[
                "conversations"
            ],
        }

    return session_info


# ============================================================
# Current user message
# ============================================================

def get_current_user_message(
    conversations,
    turn_number,
):
    """Return user message for target turn."""

    for message in conversations:

        if (
            message["turn_number"]
            == turn_number
            and message["role"]
            == "user"
        ):

            return str(
                message["content"]
            ).strip()

    return ""


# ============================================================
# Rerank predictions
# ============================================================

def rerank_hh_predictions(
    predictions,
    session_info,
    matcher,
    exact_weight,
):
    """Rerank only HH predictions."""

    reranked_predictions = []

    hh_turns = 0
    changed_turns = 0
    tracks_moved_to_first = 0

    for prediction in predictions:

        session_id = prediction[
            "session_id"
        ]

        turn_number = prediction[
            "turn_number"
        ]

        original_tracks = prediction[
            "predicted_track_ids"
        ]

        info = session_info.get(
            session_id
        )

        # --------------------------------------------
        # Non-HH -> leave untouched
        # --------------------------------------------

        if (
            info is None
            or info["specificity"] != "HH"
        ):

            reranked_predictions.append(
                {
                    "session_id":
                        session_id,

                    "turn_number":
                        turn_number,

                    "predicted_track_ids":
                        list(original_tracks),

                    "predicted_response":
                        prediction.get(
                            "predicted_response",
                            "",
                        ),
                }
            )

            continue

        hh_turns += 1

        # --------------------------------------------
        # Current user query
        # --------------------------------------------

        query = get_current_user_message(
            info["conversations"],
            turn_number,
        )

        # --------------------------------------------
        # Exact reranking
        # --------------------------------------------

        reranked_tracks = matcher.rerank(
            query=query,
            track_ids=original_tracks,
            topk=len(original_tracks),
            exact_weight=exact_weight,
        )

        if reranked_tracks != original_tracks:

            changed_turns += 1

        if (
            reranked_tracks
            and original_tracks
            and reranked_tracks[0]
            != original_tracks[0]
        ):
            tracks_moved_to_first += 1

        reranked_predictions.append(
            {
                "session_id":
                    session_id,

                "turn_number":
                    turn_number,

                "predicted_track_ids":
                    reranked_tracks,

                "predicted_response":
                    prediction.get(
                        "predicted_response",
                        "",
                    ),
            }
        )

    stats = {
        "hh_turns": hh_turns,
        "changed_turns": changed_turns,
        "tracks_moved_to_first":
            tracks_moved_to_first,
    }

    return (
        reranked_predictions,
        stats,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Tune exact metadata reranking "
            "for HH MusicCRS sessions"
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
        default="results_exact_hh_tuning.json",
    )

    parser.add_argument(
        "--best_predictions",
        default="predictions_exact_hh_best.json",
    )

    args = parser.parse_args()

    print()
    print("=" * 85)
    print("MusicCRS HH EXACT-MATCH RERANKING")
    print("=" * 85)

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    print()
    print("Loading predictions...")

    predictions = load_json(
        args.predictions
    )

    ground_truth = load_json(
        args.ground_truth
    )

    print(
        f"Predictions : {len(predictions)}"
    )

    print(
        f"Ground truth: {len(ground_truth)}"
    )

    if len(predictions) != len(
        ground_truth
    ):
        raise ValueError(
            "Prediction count does not "
            "match ground truth."
        )

    # --------------------------------------------------------
    # Session metadata
    # --------------------------------------------------------

    print()

    session_info = load_session_info(
        args.dialogue_dataset,
        args.split,
    )

    # --------------------------------------------------------
    # Catalog / matcher
    # --------------------------------------------------------

    print()
    print("Loading music catalog...")

    catalog = MusicCatalogLoader()

    matcher = ExactMetadataMatcher(
        catalog=catalog
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    print()
    print("=" * 85)
    print("BASELINE")
    print("=" * 85)

    baseline = evaluate(
        predictions=predictions,
        ground_truth=ground_truth,
        catalog_size=args.catalog_size,
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
        f"Final score: "
        f"{baseline['final_score']:.6f}"
    )

    # --------------------------------------------------------
    # Exact-match weights
    # --------------------------------------------------------

    weights = [
        0.0,
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
        0.75,
        1.00,
    ]

    tuning_results = []

    best_score = baseline[
        "final_score"
    ]

    best_weight = 0.0

    best_metrics = dict(
        baseline
    )

    best_predictions = predictions

    # --------------------------------------------------------
    # Tune
    # --------------------------------------------------------

    print()
    print("=" * 85)
    print("TUNING EXACT-MATCH WEIGHT")
    print("=" * 85)

    for weight in weights:

        print()
        print("-" * 85)

        print(
            f"Testing exact_weight = "
            f"{weight:.2f}"
        )

        print("-" * 85)

        (
            candidate_predictions,
            stats,
        ) = rerank_hh_predictions(
            predictions=predictions,
            session_info=session_info,
            matcher=matcher,
            exact_weight=weight,
        )

        metrics = evaluate(
            predictions=candidate_predictions,
            ground_truth=ground_truth,
            catalog_size=args.catalog_size,
        )

        print(
            f"HH turns             : "
            f"{stats['hh_turns']}"
        )

        print(
            f"Changed HH turns     : "
            f"{stats['changed_turns']}"
        )

        print(
            f"New track moved to #1: "
            f"{stats['tracks_moved_to_first']}"
        )

        print(
            f"nDCG@1               : "
            f"{metrics['ndcg@1']:.6f}"
        )

        print(
            f"nDCG@10              : "
            f"{metrics['ndcg@10']:.6f}"
        )

        print(
            f"nDCG@20              : "
            f"{metrics['ndcg@20']:.6f}"
        )

        print(
            f"Catalog diversity    : "
            f"{metrics['catalog_diversity']:.6f}"
        )

        print(
            f"FINAL SCORE          : "
            f"{metrics['final_score']:.6f}"
        )

        tuning_results.append(
            {
                "exact_weight":
                    weight,

                "hh_turns":
                    stats["hh_turns"],

                "changed_turns":
                    stats["changed_turns"],

                "tracks_moved_to_first":
                    stats[
                        "tracks_moved_to_first"
                    ],

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

            best_weight = weight

            best_metrics = dict(
                metrics
            )

            best_predictions = (
                candidate_predictions
            )

            print()
            print(
                "*** NEW BEST ***"
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
            best_predictions,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output = {
        "baseline":
            baseline,

        "best_weight":
            best_weight,

        "best_metrics":
            best_metrics,

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
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 85)
    print("BEST EXACT-MATCH CONFIGURATION")
    print("=" * 85)

    print(
        f"Exact weight : "
        f"{best_weight}"
    )

    print()

    print(
        f"nDCG@1      : "
        f"{best_metrics['ndcg@1']:.6f}"
    )

    print(
        f"nDCG@10     : "
        f"{best_metrics['ndcg@10']:.6f}"
    )

    print(
        f"nDCG@20     : "
        f"{best_metrics['ndcg@20']:.6f}"
    )

    print(
        f"Diversity   : "
        f"{best_metrics['catalog_diversity']:.6f}"
    )

    print(
        f"FINAL SCORE : "
        f"{best_metrics['final_score']:.6f}"
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

    print("=" * 85)


if __name__ == "__main__":
    main()
