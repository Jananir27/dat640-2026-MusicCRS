"""Enhanced BM25 retrieval using richer track metadata.

Experiment 1A extends the original BM25 corpus with track tags while
preserving the same retrieval interface as the baseline.
"""

from .bm25 import BM25Retriever


ENHANCED_CORPUS_TYPES = [
    "track_name",
    "artist_name",
    "album_name",
    "tag_list",
]


class EnhancedBM25Retriever(BM25Retriever):
    """BM25 retriever using track metadata and descriptive tags."""

    def __init__(
        self,
        dataset_name=None,
        split_types=None,
        cache_dir="./cache",
    ):
        kwargs = {
            "corpus_types": ENHANCED_CORPUS_TYPES,
            "cache_dir": cache_dir,
        }

        if dataset_name is not None:
            kwargs["dataset_name"] = dataset_name

        if split_types is not None:
            kwargs["split_types"] = split_types

        super().__init__(**kwargs)