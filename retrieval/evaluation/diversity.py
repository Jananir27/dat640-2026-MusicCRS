"""Item diversity metrics for recommendations."""
from collections.abc import Sequence


def compute_catalog_diversity(
    recommended_track_ids: Sequence[str], catalog_size: int
) -> float:
    """Catalog (item) diversity: (# unique recommended tracks) / (catalog size)."""
    if catalog_size <= 0:
        return 0.0
    return len(set(recommended_track_ids)) / float(catalog_size)
