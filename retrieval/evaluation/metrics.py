"""Retrieval evaluation metrics.

nDCG is the primary metric used to score recommendations against a
ground-truth track, as in the RecSys 2026 MusicCRS challenge.
"""
import numpy as np


def get_ndcg(gold, preds, k: int) -> float:
    """Returns the normalized discounted cumulative gain at k.

    Args:
        gold: Collection of ground truth items.
        preds: Sequence of predicted items.
        k: Number of predictions to consider.

    Returns:
        nDCG score between 0 and 1.
    """
    preds = preds[:k]
    dcg = 0.0
    for i, pred in enumerate(preds, start=1):
        rel = 1 if pred in gold else 0
        dcg += rel / np.log2(i + 1)
    n_rel = min(len(gold), k)
    idcg = sum(1 / np.log2(i + 1) for i in range(1, n_rel + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_ndcg_metrics(preds, gold, k_values: list[int]) -> dict[str, float]:
    """Computes nDCG@k for each k in `k_values`.

    Args:
        preds: The list of retrieved items for the example.
        gold: The list of gold (i.e. relevant) items for the example.
        k_values: The list of k values to compute nDCG for.

    Returns:
        A dictionary mapping "ndcg@{k}" to its score.
    """
    if len(preds) != len(set(preds)):
        raise ValueError("Predictions should be unique. Duplicates detected.")
    if len(gold) != len(set(gold)):
        raise ValueError("Gold item list should be unique. Duplicates detected.")
    return {f"ndcg@{k}": get_ndcg(gold=gold, preds=preds, k=k) for k in k_values}
