"""Evaluate recommendation predictions against ground truth.

Expects predictions as a list of dicts with `session_id`, `turn_number`,
`predicted_track_ids`, `predicted_response` — matching the RecSys 2026
MusicCRS challenge inference format — and ground truth as a list of
dicts with `session_id`, `turn_number`, `ground_truth_track_id`.
"""
import argparse
import json

from .metrics import compute_ndcg_metrics
from .diversity import compute_catalog_diversity

DEFAULT_K_VALUES = [1, 10, 20]
FINAL_SCORE_NDCG_K = 20
FINAL_SCORE_NDCG_WEIGHT = 0.8
FINAL_SCORE_DIVERSITY_WEIGHT = 0.2


def evaluate(
    predictions: list[dict],
    ground_truth: list[dict],
    catalog_size: int,
    k_values: list[int] = DEFAULT_K_VALUES,
) -> dict:
    """Computes macro-averaged nDCG and diversity metrics.

    Args:
        predictions: List of prediction entries (see module docstring).
        ground_truth: List of ground truth entries (see module docstring).
        catalog_size: Total number of tracks in the catalog.
        k_values: k values to compute nDCG for.

    Returns:
        A dictionary of macro-averaged metrics.
    """
    preds_by_key = {
        (p["session_id"], p["turn_number"]): p for p in predictions
    }

    per_turn_scores = []
    all_recommended_track_ids = []
    for gt in ground_truth:
        key = (gt["session_id"], gt["turn_number"])
        pred = preds_by_key[key]
        scores = compute_ndcg_metrics(
            preds=pred["predicted_track_ids"],
            gold=[gt["ground_truth_track_id"]],
            k_values=k_values,
        )
        per_turn_scores.append(scores)
        all_recommended_track_ids.extend(pred["predicted_track_ids"])

    macro_results = {
        metric: sum(s[metric] for s in per_turn_scores) / len(per_turn_scores)
        for metric in per_turn_scores[0]
    }
    macro_results["catalog_diversity"] = compute_catalog_diversity(
        all_recommended_track_ids, catalog_size
    )
    macro_results["final_score"] = (
        FINAL_SCORE_NDCG_WEIGHT * macro_results[f"ndcg@{FINAL_SCORE_NDCG_K}"]
        + FINAL_SCORE_DIVERSITY_WEIGHT * macro_results["catalog_diversity"]
    )
    return macro_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MusicCRS predictions")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON")
    parser.add_argument("--ground_truth", required=True, help="Path to ground truth JSON")
    parser.add_argument("--catalog_size", type=int, required=True)
    parser.add_argument("--output", default=None, help="Path to write results JSON")
    args = parser.parse_args()

    predictions = json.load(open(args.predictions))
    ground_truth = json.load(open(args.ground_truth))
    results = evaluate(predictions, ground_truth, args.catalog_size)

    print(json.dumps(results, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
