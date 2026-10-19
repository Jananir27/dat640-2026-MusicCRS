"""SQLite import, persistence, and database-backed playlist validation."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from musiccrs.music_service import ConversationState, MusicService
from retrieval.data_loader import MusicCatalogLoader


class CatalogDatabaseTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "catalog.sqlite3"
        self.rows = [
            {"track_id": "b", "track_name": ["Déjà Vu", "Deja Vu"],
             "artist_name": ["Artist One", "Guest"], "album_name": ["Album"],
             "tag_list": ["pop", "chill"], "release_date": None, "duration": 200000},
            {"track_id": "a", "track_name": ["Other"], "artist_name": ["Artist Two"],
             "album_name": ["Second Album"], "tag_list": [], "popularity": 80},
        ]

    def load(self, rows=None):
        with patch("retrieval.data_loader.load_dataset", return_value={"all_tracks": rows or self.rows}):
            return MusicCatalogLoader(database_path=self.path)

    def test_first_import_preserves_metadata_and_order_then_reopens_offline(self):
        first = self.load()
        self.assertEqual(list(first.metadata_dict), ["b", "a"])
        self.assertEqual(first.id_to_metadata("b"), self.rows[0])
        with patch("retrieval.data_loader.load_dataset", side_effect=AssertionError("Must reuse SQLite")):
            reopened = MusicCatalogLoader(database_path=self.path)
        self.assertEqual(reopened.metadata_dict, first.metadata_dict)
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM tracks").fetchone()[0], 2)

    def test_playlist_validation_checks_sqlite_not_a_fabricated_index_entry(self):
        catalog = self.load()
        catalog.metadata_dict["invented"] = {"track_id": "invented"}
        service = MusicService(catalog=catalog)
        state = ConversationState(playlist_ids=["a"])
        self.assertIsNone(service.get_track("invented"))
        with self.assertRaises(ValueError):
            service.add_tracks(state, ["b", "invented"])
        self.assertEqual(state.playlist_ids, ["a"])
        self.assertEqual(service.add_tracks(state, ["b", "b"]), ["a", "b"])
        with self.assertRaises(KeyError):
            catalog.id_to_metadata("' OR 1=1 --")

    def test_record_reads_are_safe_from_multiple_chat_threads(self):
        catalog = self.load()
        with ThreadPoolExecutor(max_workers=4) as pool:
            records = list(pool.map(catalog.id_to_metadata, ["b", "a"] * 10))
        self.assertEqual(records, self.rows * 10)

    def test_failed_import_is_not_reused_as_a_partial_catalog(self):
        def interrupted():
            yield {"track_id": "partial", "track_name": ["Partial"]}
            raise RuntimeError("Interrupted dataset read")

        with self.assertRaisesRegex(RuntimeError, "Interrupted"):
            self.load(interrupted())
        retried = self.load()
        self.assertEqual(set(retried.metadata_dict), {"a", "b"})
        with self.assertRaises(KeyError):
            retried.id_to_metadata("partial")

    def test_existing_database_cannot_be_silently_replaced_by_another_dataset(self):
        self.load()
        with self.assertRaisesRegex(ValueError, "different dataset"):
            MusicCatalogLoader(dataset_name="different/dataset", database_path=self.path)


if __name__ == "__main__":
    unittest.main()
