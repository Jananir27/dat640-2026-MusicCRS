"""Offline smoke tests for the shared MusicCRS service and R5–R8 handlers."""

import unittest

from musiccrs.music_intelligence import MusicIntelligence
from musiccrs.music_service import ConversationState, MusicService


class FakeCatalog:
    def __init__(self):
        rows = [
            ("Love", "Artist One", ["pop", "love"], "2000-01-01", 70),
            ("Love", "Metallica", ["metal"], "2001-01-01", 80),
            ("Lover", "Artist Three", ["pop", "love"], "2002-01-01", 60),
            ("Pop Song", "Artist Four", ["pop"], "2003-01-01", 50),
            ("Rock Song", "Artist Five", ["rock"], None, 40),
            ("Dance Song", "Artist Six", ["dance"], "2005-01-01", 30),
            ("Calm Song", "Artist Seven", ["calm"], "2006-01-01", 20),
            ("Bright Song", "Artist Eight", ["pop"], "2007-01-01", 10),
        ]
        self.metadata_dict = {
            str(index): {
                "track_id": str(index),
                "track_name": [name],
                "artist_name": [artist],
                "album_name": [f"Album {index}"],
                "tag_list": tags,
                "release_date": release_date,
                "popularity": popularity,
                "duration": 300_000,
            }
            for index, (name, artist, tags, release_date, popularity)
            in enumerate(rows, start=1)
        }


class FakeRetriever:
    def text_to_item_retrieval(self, query, topk):
        return list(FakeCatalog().metadata_dict)[:topk]


class MusicIntelligenceTests(unittest.TestCase):
    def setUp(self):
        self.service = MusicService(catalog=FakeCatalog(), retriever=FakeRetriever())
        self.handler = MusicIntelligence(self.service)

    def test_playlist_ids_are_unique_and_unknown_add_is_atomic(self):
        state = ConversationState()
        self.assertEqual(self.service.add_tracks(state, ["1", "1"]), ["1"])
        with self.assertRaises(ValueError):
            self.service.add_tracks(state, ["999"])
        self.assertEqual(state.playlist_ids, ["1"])

    def test_track_question_disambiguates_then_answers(self):
        state = ConversationState()
        response, changed = self.handler.handle("Who performs Love?", state)
        self.assertIn("Which track", response)
        self.assertFalse(changed)
        self.assertEqual(len(state.pending_disambiguation_ids), 2)

        response, changed = self.handler.handle("option 2", state)
        self.assertIn("Love is by", response)
        self.assertFalse(changed)
        self.assertFalse(state.pending_disambiguation_ids)

    def test_recommendations_and_positional_selection(self):
        state = ConversationState(playlist_ids=["1"])
        response, changed = self.handler.handle(
            "Recommend something based on my playlist",
            state,
        )
        self.assertIn("related to your playlist", response)
        self.assertFalse(changed)
        self.assertEqual(len(state.pending_recommendation_ids), 5)

        response, changed = self.handler.handle("Add the first two", state)
        self.assertTrue(changed)
        self.assertEqual(state.playlist_ids[:3], ["1", "2", "3"])
        self.assertFalse(state.pending_recommendation_ids)

    def test_natural_language_exclusion_and_rejection(self):
        state = ConversationState(playlist_ids=["1"])
        self.handler.handle("/recommend", state)
        response, changed = self.handler.handle(
            "Add them except the one by Metallica",
            state,
        )
        self.assertTrue(changed)
        self.assertNotIn("2", state.playlist_ids)
        self.assertEqual(set(state.playlist_ids), {"1", "3", "4", "5", "6"})

        self.handler.handle("/recommend", state)
        response, changed = self.handler.handle("No, don't add any", state)
        self.assertFalse(changed)
        self.assertIn("won't add", response)
        self.assertFalse(state.pending_recommendation_ids)

    def test_playlist_generation_replaces_only_when_description_is_present(self):
        state = ConversationState(playlist_ids=["1"])
        original = list(state.playlist_ids)

        response, changed = self.handler.handle("Create a playlist", state)
        self.assertFalse(changed)
        self.assertEqual(state.playlist_ids, original)

        response, changed = self.handler.handle(
            "Create a playlist for a short commute",
            state,
        )
        self.assertTrue(changed)
        self.assertIn("created a playlist", response)
        self.assertGreater(len(state.playlist_ids), 0)
        self.assertLessEqual(len(state.playlist_ids), 8)
        self.assertEqual(len(state.playlist_ids), len(set(state.playlist_ids)))


if __name__ == "__main__":
    unittest.main()
