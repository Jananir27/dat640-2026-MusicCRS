"""Regression cases at the R1/R2 constraint and conversation boundary."""
from copy import deepcopy
import unittest

from musiccrs.feedback_workflow import remember_r1_exclusions
from musiccrs.music_intelligence import MusicIntelligence
from musiccrs.music_service import ConversationState, MusicService
from musiccrs.relevance_feedback import FeedbackState, artist_keys
from musiccrs.request_constraints import ConstraintError, resolve_exclusions, split_request_exclusions
from musiccrs.llm_recommendations import recommend_with_llm
from tests.test_music_intelligence import FakeCatalog, FakeRetriever
from tests.test_r2_workflow import FixedParser, indexed_service


class CatalogConstraintTests(unittest.TestCase):
    def setUp(self):
        self.catalog = FakeCatalog().metadata_dict
        additions = [
            ("lp", "Numb", "Linkin Park"), ("lp-remaster", "Numb - 2003 Remastered", "Linkin Park"),
            ("21", "Numb", "21 Savage"), ("stand", "Stand By Me", "Ben E. King"),
            ("earth", "September", "Earth, Wind & Fire"), ("no", "Don't Speak", "No Doubt"),
            ("surprises", "No Surprises", "Radiohead"), ("beyonce", "Halo", "Beyoncé"),
        ]
        for t, title, artist in additions:
            self.catalog[t] = {"track_id": t, "track_name": [title], "artist_name": [artist], "tag_list": ["pop"]}

    def test_negative_syntax_and_positive_names_are_separated(self):
        for request in ("Recommend metal, no Metallica", "Recommend metal not by Metallica",
                        "Recommend metal but not Metallica", "Recommend metal without Metallica"):
            with self.subTest(request=request):
                positive, negative = split_request_exclusions(request)
                self.assertEqual(positive, "Recommend metal")
                self.assertEqual(negative, "exclude Metallica")
        for request in ('Recommend "No Surprises" by Radiohead', "Recommend No Doubt",
                        "Recommend music like No Doubt", "Recommend music similar to Without Me"):
            self.assertEqual(split_request_exclusions(request), (request, None))

    def test_lists_preserve_literal_band_names(self):
        for text in ("Earth, Wind & Fire and Metallica", "Metallica, Earth, Wind & Fire",
                     "Earth, Wind & Fire; Metallica", "Earth, Wind & Fire without Metallica"):
            with self.subTest(text=text):
                result = resolve_exclusions(text, self.catalog)
                self.assertEqual(result.artists, {"earth wind fire", "metallica"})

    def test_repeated_negative_markers_and_numeric_artist_names(self):
        result = resolve_exclusions("artist 21 Savage and without Linkin Park", self.catalog)
        self.assertEqual(result.artists, {"21 savage", "linkin park"})

    def test_bare_exact_artist_name_has_priority_over_a_same_named_track(self):
        self.catalog['song'] = {"track_id": "song", "track_name": ["Daft Punk"], "artist_name": ["Pentatonix"]}
        self.catalog['band'] = {"track_id": "band", "track_name": ["One More Time"], "artist_name": ["Daft Punk"]}
        self.assertEqual(resolve_exclusions("Daft Punk", self.catalog).artists, {"daft punk"})
        self.assertEqual(resolve_exclusions("track Daft Punk", self.catalog).track_ids, {"song"})

    def test_track_identity_excludes_remasters_without_excluding_another_artist(self):
        result = resolve_exclusions("Numb by Linkin Park", self.catalog)
        self.assertEqual(result.track_ids, {"lp", "lp-remaster"})
        with self.assertRaisesRegex(ConstraintError, "Which artist"):
            resolve_exclusions("Numb", self.catalog)

    def test_by_in_literal_titles_and_accents_are_resolved(self):
        self.assertEqual(resolve_exclusions("Stand By Me", self.catalog).track_ids, {"stand"})
        self.assertEqual(resolve_exclusions("Beyonce", self.catalog).artists, {"beyoncé"})

    def test_quoted_exclusion_titles_are_kept_intact(self):
        self.assertEqual(resolve_exclusions('"No Surprises" and Metallica', self.catalog).track_ids,
                         {"surprises"})

    def test_unknown_and_ambiguous_constraints_do_not_apply_partial_updates(self):
        service = MusicService(FakeCatalog(), FakeRetriever())
        # Replace the catalog before calling the pure constraint resolver; no model loads.
        service.catalog.metadata_dict = self.catalog
        state = ConversationState()
        for tracks in (["Numb"], ["Stand By Me", "Missing Song"]):
            preferences = FixedParser(excluded_artists=["Metallica"], excluded_tracks=tracks).parse("pop")
            with self.assertRaises(ConstraintError):
                remember_r1_exclusions(service, state, preferences)
            self.assertEqual(state.feedback, FeedbackState())

    def test_positive_seed_does_not_disambiguate_a_bare_negative_title(self):
        service = MusicService(FakeCatalog(), FakeRetriever())
        service.catalog.metadata_dict = self.catalog
        prefs = FixedParser(excluded_tracks=["Numb"]).parse("rock")
        state = ConversationState()
        with self.assertRaises(ConstraintError):
            remember_r1_exclusions(service, state, prefs,
                                   "Recommend music like Numb by Linkin Park but exclude Numb")
        self.assertEqual(state.feedback.excluded_track_ids, set())
        remember_r1_exclusions(service, state, prefs,
                               "Recommend music like Numb by Linkin Park but exclude Numb by 21 Savage")
        self.assertEqual(state.feedback.excluded_track_ids, {"21"})

    def test_featured_artist_credits_are_blocked_without_splitting_band_names(self):
        feedback = FeedbackState(excluded_artists={"metallica"})
        for credit in ("Guest feat. Metallica", "Guest (feat. Metallica)", "Guest featuring Metallica", "Guest ft. Metallica"):
            with self.subTest(credit=credit):
                row = {"track_id": "x", "artist_name": [credit]}
                self.assertTrue(feedback.excludes(row))
        self.assertEqual(artist_keys({"artist_name": ["Earth, Wind & Fire"]}), {"earth wind fire"})
        self.assertEqual(artist_keys({"artist_name": ["Man With A Mission"]}), {"man with a mission"})


class IntegratedBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.service = MusicService(FakeCatalog(), FakeRetriever())
        self.chat = MusicIntelligence(self.service)
        self.state = ConversationState()

    def test_standard_route_enforces_no_and_multiple_exclusions(self):
        self.chat.handle("Recommend metal, no Metallica", self.state)
        self.assertEqual(self.state.last_recommendation_ids, [])
        self.assertIn("metallica", self.state.feedback.excluded_artists)
        self.chat.handle("Recommend pop excluding Metallica and Artist Three", self.state)
        self.assertNotIn("3", self.state.last_recommendation_ids)
        self.assertIn("artist three", self.state.feedback.excluded_artists)

    def test_invalid_constraint_and_feedback_in_one_request_are_atomic(self):
        self.chat.handle("Recommend pop music", self.state)
        response, _ = self.chat.handle("I like track 2 and recommend rock without Missing Artist", self.state)
        self.assertIn("could not identify", response)
        self.assertEqual(self.state.feedback, FeedbackState())

    def test_literal_title_containing_by_can_receive_positive_feedback(self):
        self.service.catalog.metadata_dict["3"]["track_name"] = ["Stand By Me"]
        self.chat.handle("Recommend pop music", self.state)
        self.chat.handle("I like Stand By Me", self.state)
        self.assertEqual(self.state.feedback.votes, {"3": 1})

    def test_named_artist_does_not_become_a_genre_constraint(self):
        self.service.catalog.metadata_dict["6"].update(artist_name=["Daft Punk"], tag_list=["electronic"])
        for request in ("Recommend Daft Punk", "Recommend music similar to Daft Punk"):
            genres, _ = self.service.extract_request_preferences(request)
            self.assertEqual(set(genres), set())
        genres, _ = self.service.extract_request_preferences("Recommend rock like Daft Punk")
        self.assertEqual(set(genres), {"rock"})

    def test_composite_genre_tags_survive_the_entire_seed_free_route(self):
        self.service.catalog.metadata_dict['2']['tag_list'] = ['R&B/Soul', 'mellow']
        self.chat.handle("Recommend mellow R&B music", self.state)
        self.assertEqual(self.state.last_recommendation_ids, ['2'])

    def test_avoiding_a_song_style_records_negative_feedback_without_banning_its_artist(self):
        self.chat.handle("Recommend pop music but avoid the style of Lover by Artist Three", self.state)
        self.assertEqual(self.state.feedback.votes, {'3': -1})
        self.assertEqual(self.state.feedback.disliked_artists, set())
        self.assertNotIn('3', self.state.last_recommendation_ids)

    def test_configured_advanced_route_works_without_playlist_seeds(self):
        parser = FixedParser(genre=["pop"])
        chat = MusicIntelligence(self.service, advanced_recommender=lambda request, state:
                                 recommend_with_llm(request, parser, self.service, state))
        response, _ = chat.handle("Recommend pop music", self.state)
        self.assertTrue(self.state.last_recommendation_ids)
        self.assertEqual(len(parser.calls), 1)

    def test_expansion_filters_compound_credits_before_topk(self):
        service = indexed_service()
        service.catalog.metadata_dict["original"]["artist_name"] = ["Classic feat. Blocked"]
        state = ConversationState()
        state.feedback.rate("liked", True)
        state.feedback.excluded_artists.add("blocked")
        result = service.rerank_feedback([], state, "rock", 1)
        self.assertEqual([t["track_id"] for t in result], ["related"])

    def test_ambiguous_featured_artist_feedback_asks_and_named_feedback_resolves(self):
        self.service.catalog.metadata_dict['3']['artist_name'] = ['Guest feat. Metallica']
        self.state.last_recommendation_ids = ['3']
        response, _ = self.chat.handle("I don't like this artist", self.state)
        self.assertIn('several artists', response)
        self.assertEqual(self.state.feedback, FeedbackState())
        self.chat.handle("I don't like the artist Metallica", self.state)
        self.assertIn('metallica', self.state.feedback.disliked_artists)
        self.assertEqual(self.state.feedback.votes, {'3': -1})

    def test_unknown_external_votes_do_not_switch_the_r1_ranking_path(self):
        service = indexed_service()
        state = ConversationState()
        state.feedback.rate("absent-id", True)
        before = deepcopy(state.feedback)
        result = service.rerank_feedback([service.get_track("original")], state, "rock")
        self.assertEqual([t["track_id"] for t in result], ["original"])
        self.assertIsNone(service._feedback_adapter._index)
        self.assertEqual(state.feedback, before)


if __name__ == "__main__":
    unittest.main()
