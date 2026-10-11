"""Regression coverage for a BM25 build interrupted before all files exist."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from retrieval.bm25 import BM25Retriever
from tests.test_music_intelligence import FakeCatalog


class BM25CacheTests(unittest.TestCase):
    def test_incomplete_cache_rebuilds_and_complete_cache_is_reused(self):
        with tempfile.TemporaryDirectory() as cache:
            index = Path(cache) / "bm25" / "track_name_artist_name_album_name"
            index.mkdir(parents=True)
            args = {"catalog": FakeCatalog(), "cache_dir": cache}
            retriever = BM25Retriever(**args)
            expected = retriever.text_to_item_retrieval("Metallica", topk=1)
            self.assertEqual(expected, ["2"])
            # Also handle a partially populated cache, not only an empty folder.
            (index / "track_ids.json").unlink()
            rebuilt = BM25Retriever(**args)
            self.assertEqual(rebuilt.text_to_item_retrieval("Metallica", topk=1), expected)
            with patch.object(BM25Retriever, "_build_index", side_effect=AssertionError("Unnecessary rebuild")):
                cached = BM25Retriever(**args)
                self.assertEqual(cached.text_to_item_retrieval("Metallica", topk=1), expected)
