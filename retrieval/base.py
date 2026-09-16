"""Abstract interface for retrieval modules.

Concrete retrieval methods (e.g., BM25, dense/embedding-based retrieval)
should implement this interface so they can be swapped interchangeably.
"""
from abc import ABC, abstractmethod


class RetrievalModule(ABC):
    """Base class for text-to-item retrieval over the track catalog."""

    @abstractmethod
    def text_to_item_retrieval(self, query: str, topk: int) -> list[str]:
        """Retrieve top-k track IDs for a single natural language query.

        Args:
            query: The user text query to match against the catalog.
            topk: Number of items to retrieve.

        Returns:
            A list of track IDs ordered by decreasing relevance score.
        """
        raise NotImplementedError

    @abstractmethod
    def batch_text_to_item_retrieval(
        self, queries: list[str], topk: int
    ) -> list[list[str]]:
        """Retrieve top-k track IDs for multiple queries in batch.

        Args:
            queries: List of user text queries.
            topk: Number of items to retrieve per query.

        Returns:
            A list of lists of track IDs, one list per query, each ordered
            by decreasing relevance score.
        """
        raise NotImplementedError
