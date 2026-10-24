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


def graded_retrieval_metrics(preds: list[str], judgments: dict[str, int], k: int = 5) -> dict[str, float]:
    """Part 4: graded 0–2 nDCG with exponential gains and binary Precision@k.

    Precision counts grades 1 and 2, always divides by k (short lists are not
    rewarded). Unjudged returned items are an error, not invented negatives.
    The ideal ranking includes all judged items in this query's pool.
    """
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if len(preds) != len(set(preds)):
        raise ValueError("Predictions must be unique")
    if any(type(grade) is not int or grade not in (0, 1, 2) for grade in judgments.values()):
        raise ValueError("Judgments must be integer grades 0, 1, or 2")
    if any(track_id not in judgments for track_id in preds[:k]):
        raise ValueError("Returned top-k tracks must be judged before scoring")
    grades = [judgments[track_id] for track_id in preds[:k]]
    dcg = sum((2 ** grade - 1) / np.log2(rank + 2) for rank, grade in enumerate(grades))
    ideal = sorted(judgments.values(), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / np.log2(rank + 2) for rank, grade in enumerate(ideal))
    return {f"ndcg@{k}": float(dcg / idcg) if idcg else 0.0,
            f"precision@{k}": sum(grade > 0 for grade in grades) / k}


def latency_summary(milliseconds: list[float]) -> dict[str, float]:
    """Observed wall-clock response times; model initialization is reported separately."""
    times = np.asarray(milliseconds, dtype=float)
    if times.ndim != 1 or times.size == 0 or not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("Latency requires a nonempty list of finite, nonnegative milliseconds")
    return {"mean_ms": float(times.mean()), "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95))}
