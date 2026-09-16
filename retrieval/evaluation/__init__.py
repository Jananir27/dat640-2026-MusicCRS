"""Evaluation metrics for the retrieval module.

Provides nDCG and item (catalog) diversity, adapted from the RecSys 2026
MusicCRS challenge evaluator, plus a combined final score
(0.8 * nDCG@20 + 0.2 * catalog diversity) used to rank submissions.
"""
from .metrics import compute_ndcg_metrics, get_ndcg
from .diversity import compute_catalog_diversity
from .make_ground_truth import make_ground_truth

__all__ = [
    "compute_ndcg_metrics",
    "get_ndcg",
    "compute_catalog_diversity",
    "make_ground_truth",
]
