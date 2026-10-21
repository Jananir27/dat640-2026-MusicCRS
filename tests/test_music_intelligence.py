"""Offline smoke tests for the shared MusicCRS service and R5–R8 handlers."""

import unittest
from unittest.mock import patch

from musiccrs.music_intelligence import MusicIntelligence
from musiccrs.r5_nlu import CatalogEntities, IntentPrediction
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

    def test_polite_r5_followup_is_canonicalized_before_entity_matching(self):
        state = ConversationState(last_question_track_id="1")
        canonical = self.handler._canonicalize_r5_message(
            "Do you know when it was released?", state
        )
        self.assertEqual(canonical, "when this song was released?")
        self.assertEqual(self.service.get_r5_entities().mentions(canonical, "title"), [])

    def test_artist_and_album_questions_do_not_select_short_catalog_collisions(self):
        catalog = FakeCatalog()
        catalog.metadata_dict["9"] = {
            "track_id": "9", "track_name": ["Park"], "artist_name": ["Other"],
            "album_name": ["You"], "tag_list": ["country"],
            "release_date": None, "popularity": 1, "duration": 180_000,
        }
        catalog.metadata_dict["10"] = {
            "track_id": "10", "track_name": ["Numb"],
            "artist_name": ["Linkin Park"], "album_name": ["Meteora"],
            "tag_list": ["rock"], "release_date": "2003-03-25",
            "popularity": 90, "duration": 180_000,
        }
        handler = MusicIntelligence(MusicService(catalog=catalog))
        response, _ = handler.handle("What other songs does Linkin Park have?", ConversationState())
        self.assertIn("Numb — Linkin Park", response)
        response, _ = handler.handle("Could you list the tracks on the album Meteora?", ConversationState())
        self.assertIn("Tracks from Meteora: Numb", response)
        response, _ = handler.handle("What genre is Linkin Park?", ConversationState())
        self.assertIn("tagged rock", response)

    def test_conversational_recommendation_and_playlist_requests_are_routed(self):
        self.assertTrue(self.handler._is_recommendation_request(
            "What else would sound good with the songs I have?"
        ))
        self.assertTrue(self.handler._is_recommendation_request(
            "Could you recommend a few more like these?"
        ))
        for message, description in [
            ("I want some sad love songs. Could you make a playlist?", "sad love songs"),
            ("Build a high-energy gym playlist for me.", "high-energy gym"),
            ("Could you put together some mellow jazz for a quiet evening?", "mellow jazz quiet evening"),
            ("I am going for a run. Give me a playlist that will keep me moving.", "running workout"),
        ]:
            with self.subTest(message=message):
                self.assertTrue(self.handler._is_playlist_creation(message))
                self.assertEqual(self.handler._playlist_description(message), description)

    def test_release_verb_is_not_masked_as_a_track_title(self):
        class Catalog:
            metadata_dict = {
                "numb": {
                    "track_name": ["Numb"],
                    "artist_name": ["Linkin Park"],
                    "album_name": ["Meteora"],
                },
                "released": {
                    "track_name": ["Released"],
                    "artist_name": ["Other Artist"],
                    "album_name": ["Other Album"],
                },
            }

        entities = CatalogEntities(Catalog())
        message = "When was Numb by Linkin Park released?"
        title_mentions = entities.mentions(message, "title")
        artist_mentions = entities.mentions(message, "artist")
        album_mentions = entities.mentions(message, "album")

        self.assertEqual([mention.name for mention in title_mentions], ["numb"])
        self.assertEqual([mention.name for mention in artist_mentions], ["linkin park"])
        self.assertEqual(
            entities.mask(message, title_mentions, artist_mentions, album_mentions),
            "when was this song by this artist released",
        )

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
        self.assertTrue(3 <= len(state.pending_recommendation_ids) <= 5)
        expected_added = state.pending_recommendation_ids[:2]

        response, changed = self.handler.handle("Add the first two", state)
        self.assertTrue(changed)
        self.assertEqual(state.playlist_ids[:3], ["1", *expected_added])
        self.assertFalse(state.pending_recommendation_ids)

    def test_conversational_recommendations_are_not_misread_as_playlist_view(self):
        state = ConversationState()
        response, changed = self.handler.handle(
            "Could you suggest a few songs based on what is in my playlist?", state
        )
        self.assertIn("Add at least one song", response)
        self.assertFalse(changed)

        state.playlist_ids = ["1"]
        response, changed = self.handler.handle(
            "I would like a few more recommendations based on my playlist.", state
        )
        self.assertIn("related to your playlist", response)
        self.assertFalse(changed)

    def test_recommendation_selection_combines_ordinal_and_numeric_positions(self):
        for message, expected in [
            ("Add first, second and fourth", ["2", "3", "5"]),
            ("Add the first, the second, and the fourth songs", ["2", "3", "5"]),
            ("Could you add 1st, second and 4 to my playlist?", ["2", "3", "5"]),
            ("Add 1, 2 and 4", ["2", "3", "5"]),
            ("Add fourth, 2nd and first", ["5", "3", "2"]),
            ("Add first, last and first", ["2", "6"]),
            ("Add the first two", ["2", "3"]),
            ("Add the first four", ["2", "3", "4", "5"]),
            ("the second one", ["3"]),
        ]:
            with self.subTest(message=message):
                state = ConversationState(
                    playlist_ids=["1"], pending_recommendation_ids=["2", "3", "4", "5", "6"],
                )
                response, changed = self.handler.handle(message, state)
                self.assertTrue(changed)
                self.assertEqual(state.playlist_ids, ["1", *expected])
                self.assertIn(f"Added {len(expected)} song(s)", response)
                self.assertFalse(state.pending_recommendation_ids)

    def test_invalid_recommendation_position_preserves_playlist_and_all_suggestions(self):
        for message in [
            "Add first, second and ninth", "Add 1, 2 and 9", "Add first and 0",
            "Add the ninth", "Add the first nine", "Add 0", "Add 2 and 99th",
        ]:
            with self.subTest(message=message):
                pending = ["2", "3", "4", "5", "6"]
                state = ConversationState(playlist_ids=["1"], pending_recommendation_ids=pending[:])
                response, changed = self.handler.handle(message, state)
                self.assertFalse(changed)
                self.assertIn("1 to 5", response)
                self.assertEqual(state.playlist_ids, ["1"])
                self.assertEqual(state.pending_recommendation_ids, pending)
                response, changed = self.handler.handle("Add first, second and fourth", state)
                self.assertTrue(changed)
                self.assertEqual(state.playlist_ids, ["1", "2", "3", "5"])

    def test_disambiguation_requires_one_valid_position_and_preserves_question(self):
        state = ConversationState(
            pending_disambiguation_ids=["1", "2"], pending_disambiguation_action="artist",
        )
        for message, expected in [
            ("first and ninth", "1 to 2"), ("first and second", "choose one"),
        ]:
            response, changed = self.handler.handle(message, state)
            self.assertFalse(changed)
            self.assertIn(expected, response)
            self.assertEqual(state.pending_disambiguation_ids, ["1", "2"])
            self.assertEqual(state.pending_disambiguation_action, "artist")
        response, changed = self.handler.handle("the second one", state)
        self.assertEqual(response, "Love is by Metallica.")
        self.assertFalse(changed)

    def test_pending_selection_does_not_consume_numbers_in_a_song_title(self):
        self.service.catalog.metadata_dict["9"] = {
            "track_id": "9", "track_name": ["The Second Coming"],
            "artist_name": ["Artist Nine"], "tag_list": [],
        }
        state = ConversationState(playlist_ids=["1"], pending_recommendation_ids=["2", "3"])
        response, changed = self.handler.handle("Add The Second Coming", state)
        self.assertTrue(changed)
        self.assertEqual(state.playlist_ids, ["1", "9"])
        self.assertIn("The Second Coming", response)

    def test_natural_language_exclusion_and_rejection(self):
        state = ConversationState(playlist_ids=["1"])
        self.handler.handle("/recommend", state)
        response, changed = self.handler.handle(
            "Add them except the one by Artist Three",
            state,
        )
        self.assertTrue(changed)
        self.assertNotIn("3", state.playlist_ids)
        self.assertEqual(set(state.playlist_ids), {"1", "4", "8"})

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
        self.assertLessEqual(len(state.playlist_ids), 5)
        self.assertEqual(len(state.playlist_ids), len(set(state.playlist_ids)))

    def test_natural_add_and_put_together_playlist_commands(self):
        state = ConversationState()
        response, changed = self.handler.handle(
            "Could you add Pop Song to my playlist?", state
        )
        self.assertTrue(changed)
        self.assertIn("Added Pop Song", response)

        response, changed = self.handler.handle(
            "Please put together a playlist for the gym, with upbeat songs.", state
        )
        self.assertTrue(changed)
        self.assertIn("playlist for ‘gym upbeat’", response)
        self.assertEqual(len(state.playlist_ids), len(set(state.playlist_ids)))

    def test_underspecified_playlist_request_preserves_current_contents(self):
        state = ConversationState(playlist_ids=["1"])
        response, changed = self.handler.handle("Could you make me a playlist?", state)
        self.assertIn("What mood, genre, or activity", response)
        self.assertFalse(changed)
        self.assertEqual(state.playlist_ids, ["1"])

    def test_playlist_length_cues_do_not_bias_retrieval(self):
        state = ConversationState()
        with patch.object(
            self.service,
            "recommend_for_playlist_description",
            wraps=self.service.recommend_for_playlist_description,
        ) as retrieve:
            response, changed = self.handler.handle(
                "Make me a short playlist for my commute.", state,
            )

        self.assertTrue(changed)
        self.assertIn("playlist for ‘short commute’", response)
        self.assertEqual(retrieve.call_args.args[0], "easy listening relaxed music")
        self.assertEqual(
            retrieve.call_args.kwargs["request"], "easy listening relaxed music"
        )


class PerformerLookupTests(unittest.TestCase):
    """Regression coverage for real question/name collisions in the catalog."""

    def setUp(self):
        catalog = FakeCatalog()
        rows = [
            ("billie", "Billie Jean", ["Michael Jackson"], "Thriller"),
            ("cover", "Billie Jean", ["Kodaline"], "Forever Young"),
            ("survive", "I Will Survive", ["Cake"], "Fashion Nugget"),
            ("survive-cover", "I Will Survive", ["Demi Lovato"], "Soundtrack"),
            # These collide with the verb and the misspelled artist slot.
            ("collision", "Micheal", ["Sango"], "Jackson"),
            ("savage", "No Heart", ["21 Savage", "Metro Boomin"], "Savage Mode"),
            ("sequel-collision", "Unrelated", ["Other Artist"], "II"),
            ("stand", "Stand By Me", ["Ben E. King"], "Don't Play That Song"),
        ]
        catalog.metadata_dict = {
            track_id: {
                "track_id": track_id, "track_name": [title],
                "artist_name": artists, "album_name": [album],
                "release_date": "2000-01-01", "tag_list": [],
            }
            for track_id, title, artists, album in rows
        }
        self.handler = MusicIntelligence(MusicService(catalog=catalog))

    def test_performer_questions_offer_catalog_versions_instead_of_false_misses(self):
        for message, names in [
            ("who sang billie jean", ["Michael Jackson", "Kodaline"]),
            ("who said billie jean", ["Michael Jackson", "Kodaline"]),
            ("who sang I will survive", ["Cake", "Demi Lovato"]),
        ]:
            with self.subTest(message=message):
                state = ConversationState(playlist_ids=["stand"])
                response, changed = self.handler.handle(message, state)
                self.assertIn("Which track", response)
                for name in names:
                    self.assertIn(name, response)
                self.assertFalse(changed)
                self.assertEqual(state.playlist_ids, ["stand"])

    def test_artist_typo_is_resolved_inside_the_artist_slot(self):
        response, changed = self.handler.handle(
            "who sang billie jean by micheal jackson", ConversationState()
        )
        self.assertEqual(response, "Billie Jean is by Michael Jackson.")
        self.assertFalse(changed)

    def test_ambiguous_versions_are_ranked_by_catalog_popularity(self):
        catalog = self.handler.service.catalog
        catalog.metadata_dict["billie"]["popularity"] = 30
        catalog.metadata_dict["cover"]["popularity"] = 70
        state = ConversationState()
        response, _ = self.handler.handle("Who sang Billie Jean?", state)
        self.assertEqual(state.pending_disambiguation_ids, ["cover", "billie"])
        self.assertIn("1. Billie Jean — Kodaline", response)

    def test_album_names_and_missing_spaces_are_supported(self):
        for message in ["who sang savage mode", "who sang savagemode"]:
            with self.subTest(message=message):
                response, changed = self.handler.handle(message, ConversationState())
                self.assertIn("‘Savage Mode’ is an album", response)
                self.assertIn("21 Savage", response)
                self.assertIn("Metro Boomin", response)
                self.assertFalse(changed)

    def test_missing_sequel_is_not_replaced_by_the_original_or_previous_track(self):
        state = ConversationState(last_question_track_id="billie")
        response, changed = self.handler.handle("who sang savageMode II", state)
        self.assertIn("couldn't find a track or album matching ‘savageMode II’", response)
        self.assertFalse(changed)

    def test_by_can_be_part_of_a_real_title(self):
        response, changed = self.handler.handle("Who sang Stand By Me?", ConversationState())
        self.assertEqual(response, "Stand By Me is by Ben E. King.")
        self.assertFalse(changed)

    def test_performer_pronoun_uses_the_remembered_track(self):
        state = ConversationState(last_question_track_id="billie")
        response, changed = self.handler.handle("Who sang it?", state)
        self.assertEqual(response, "Billie Jean is by Michael Jackson.")
        self.assertFalse(changed)

    def test_unknown_explicit_artist_is_not_ignored(self):
        response, changed = self.handler.handle(
            "Who sang Billie Jean by Zzzzz Qqqqq?", ConversationState()
        )
        self.assertIn("couldn't find the artist", response)
        self.assertFalse(changed)


if __name__ == "__main__":
    unittest.main()
