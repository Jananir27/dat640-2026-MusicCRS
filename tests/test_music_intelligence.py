"""Offline smoke tests for the shared MusicCRS service and R5–R8 handlers."""

import unittest
from unittest.mock import patch

from musiccrs.music_intelligence import MusicIntelligence
from musiccrs.r5_nlu import IntentPrediction
from musiccrs.music_service import ConversationState, MusicService


class FakeCatalog:
    def id_to_metadata(self, track_id):
        return self.metadata_dict[track_id]

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

    def test_r5_contextual_paraphrases_preserve_title_positions(self):
        state = ConversationState(last_question_artist="Artist One")
        canonicalize = self.handler._canonicalize_r5_message

        self.assertEqual(
            canonicalize("When did it drop?", state), "When was this song released?"
        )
        self.assertEqual(
            canonicalize("When did Numb drop?", state), "When was Numb released?"
        )
        self.assertEqual(
            canonicalize("Anything else by them?", state),
            "Anything else by this artist?",
        )
        self.assertEqual(canonicalize("Who made Numb?", state), "Who performs Numb?")

        # These catalog names stay intact when they appear in a title slot.
        self.assertEqual(canonicalize("What album is Drop on?", state), "What album is Drop on?")
        self.assertEqual(canonicalize("What album is Them on?", state), "What album is Them on?")

    def test_release_drop_followup_uses_remembered_track(self):
        state = ConversationState(last_question_track_id="1")
        response, changed = self.handler.handle("When did it drop?", state)
        self.assertEqual(response, "Love was released on 2000-01-01.")
        self.assertFalse(changed)

    def test_artist_pronoun_followup_uses_context_without_fuzzy_dependency(self):
        class ArtistTracksRouter:
            @staticmethod
            def predict(_message):
                return IntentPrediction("artist_tracks", 0.9, 0.5, True)

        handler = MusicIntelligence(self.service, r5_router=ArtistTracksRouter())
        state = ConversationState(
            last_question_track_id="1",
            last_question_artist="Artist One",
        )
        response = handler._handle_question("Anything else by them?", state)
        self.assertIn("other tracks by Artist One", response)

    def test_unmatched_explicit_title_does_not_fall_back_to_previous_track(self):
        class AlbumRouter:
            @staticmethod
            def predict(_message):
                return IntentPrediction("album", 0.9, 0.5, True)

        handler = MusicIntelligence(self.service, r5_router=AlbumRouter())
        state = ConversationState(last_question_track_id="1")
        response = handler._handle_question("What album is Them on?", state)
        self.assertEqual(response, "Which song or track should I look up?")

    def test_recommendations_and_positional_selection(self):
        state = ConversationState(playlist_ids=["1"])
        response, changed = self.handler.handle(
            "Recommend something based on my playlist",
            state,
        )
        self.assertIn("related to your playlist", response)
        self.assertIn("shares genres: pop", response)
        self.assertNotIn("shares tags:", response)
        self.assertFalse(changed)
        self.assertEqual(len(state.pending_recommendation_ids), 5)
        expected_added = state.pending_recommendation_ids[:2]

        response, changed = self.handler.handle("Add the first two", state)
        self.assertTrue(changed)
        self.assertEqual(state.playlist_ids[:3], ["1", *expected_added])
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

    def test_playlist_length_cues_do_not_bias_retrieval(self):
        state = ConversationState()
        with patch.object(
            self.service,
            "recommend_for_playlist_description",
            wraps=self.service.recommend_for_playlist_description,
        ) as retrieve:
            response, changed = self.handler.handle(
                "Make a short commute playlist", state,
            )

        self.assertTrue(changed)
        self.assertIn("playlist for ‘short commute’", response)
        self.assertEqual(retrieve.call_args.args[0], "commute")
        self.assertEqual(retrieve.call_args.kwargs["request"], "commute")


if __name__ == "__main__":
    unittest.main()
