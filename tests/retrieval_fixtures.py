"""Explicit synthetic settings/rankings; never used by the running app."""

from retrieval.base import RetrievalModule


def fixture_artifact():
    return {
        "schema_version": 1,
        "hybrid": {"candidate_k": 50, "reservoir_k": 100, "dense_weight": 0.25,
                   "rrf_k": 60, "fusion_rrf_k": 60},
        "fusion_weights": dict.fromkeys(("HH", "HL", "LH", "LL"), 0.3),
        "exact_weights": dict.fromkeys(("HH", "HL", "LH", "LL"), 0.05),
        "diversity": {"protect_top": 10, "novelty_weight": 0.5},
        "selective": {"protect_top": 12, "novelty_weight": 1.35, "original_bonus": 0.14,
                      "reservoir_start": 11, "reservoir_end": 100},
        "diversity_statistics": {"total_turns": 100, "frequency": {str(i): 100 for i in range(1, 21)}},
        "selective_statistics": {"total_turns": 100, "frequency": {str(i): 100 for i in range(1, 21)}},
    }


class FixedDenseRetriever(RetrievalModule):
    """Deterministic stand-in for Qwen; makes no semantic-quality claim."""

    def __init__(self, track_ids):
        self.track_ids = list(track_ids)
        self.calls = []

    def text_to_item_retrieval(self, query, topk):
        self.calls.append((query, topk))
        return self.track_ids[:topk]

    def batch_text_to_item_retrieval(self, queries, topk):
        return [self.text_to_item_retrieval(query, topk) for query in queries]
