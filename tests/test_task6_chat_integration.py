"""Offline regression checks for the optional Part 4 chat integration."""
import os
import unittest
from unittest.mock import patch

from musiccrs.music_intelligence import MusicIntelligence
from musiccrs.music_service import ConversationState


class FakeService:
    def __init__(self):
        self.tracks = {
            "seed": {"track_id": "seed", "track_name": ["Punk Seed"], "artist_name": ["Rancid"], "tag_list": ["punk"]},
            "a": {"track_id": "a", "track_name": ["Track A"], "artist_name": ["Green Day"], "tag_list": ["punk"]},
            "b": {"track_id": "b", "track_name": ["Track B"], "artist_name": ["Green Day"], "tag_list": ["rock"]},
            "c": {"track_id": "c", "track_name": ["Track C"], "artist_name": ["The Clash"], "tag_list": ["punk"]},
        }

    def get_track(self, track_id):
        return self.tracks.get(track_id)

    @staticmethod
    def display_value(value):
        return ", ".join(value) if isinstance(value, list) else str(value or "")

    def display_track(self, track):
        return f"{self.display_value(track['track_name'])} — {self.display_value(track['artist_name'])}"

    def add_tracks(self, state, track_ids):
        for track_id in track_ids:
            if track_id not in state.playlist_ids:
                state.playlist_ids.append(track_id)
        return state.playlist_ids

    def recommend_from_playlist(self, state, limit=5, request=""):
        return [self.tracks[tid] for tid in ["a", "b", "c"][:limit]]


class AdvancedChatIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.service = FakeService()
        self.calls = []

        def provider(query, state):
            self.calls.append(query)
            return {"recommendations": [self.service.tracks[tid] for tid in ("a", "b", "c")]}

        self.intelligence = MusicIntelligence(self.service, advanced_recommender=provider)
        self.state = ConversationState(playlist_ids=["seed"])

    def test_new_recommendation_is_displayed_and_selectable(self):
        text, updated = self.intelligence.handle(
            "Recommend energetic punk songs, no Billy Talent", self.state
        )
        self.assertFalse(updated)
        self.assertIn("Track A — Green Day", text)
        self.assertEqual(self.state.pending_recommendation_ids, ["a", "b", "c"])
        self.assertEqual(len(self.calls), 1)
        confirmation, updated = self.intelligence.handle("add the first two", self.state)
        self.assertTrue(updated)
        self.assertIn("Added 2 song(s)", confirmation)
        self.assertEqual(self.state.playlist_ids, ["seed", "a", "b"])
        self.assertEqual(len(self.calls), 1, "A selection must not call the LLM")

    def test_no_clause_is_not_mistaken_for_rejecting_prior_suggestions(self):
        self.state.pending_recommendation_ids = ["a"]
        response, _ = self.intelligence.handle(
            "Recommend rock songs, no Linkin Park", self.state
        )
        self.assertIn("Here are recommendations", response)
        self.assertEqual(self.state.pending_recommendation_ids, ["a", "b", "c"])
        self.assertEqual(len(self.calls), 1)

    def test_empty_playlist_does_not_invoke_part4(self):
        state = ConversationState()
        response, changed = self.intelligence.handle("Recommend rock songs", state)
        self.assertFalse(changed)
        self.assertIn("Add at least one", response)
        self.assertEqual(len(self.calls), 0)

    def test_help_uses_existing_handler_without_llm(self):
        response, updated = self.intelligence.handle("What can you do?", self.state)
        self.assertIn("add a song", response)
        self.assertFalse(updated)
        self.assertFalse(self.calls)

    def test_bad_llm_response_keeps_playlist_unchanged_and_drops_old_choices(self):
        def failing(_query, _state):
            raise RuntimeError("Test-only network failure")
        intelligence = MusicIntelligence(self.service, advanced_recommender=failing)
        self.state.pending_recommendation_ids = ["c"]
        with self.assertLogs("musiccrs.music_intelligence", level="ERROR"):
            response, updated = intelligence.handle("Recommend punk music", self.state)
        self.assertIn("couldn't process", response)
        self.assertFalse(updated)
        self.assertEqual(self.state.playlist_ids, ["seed"])
        self.assertEqual(self.state.pending_recommendation_ids, [])

    def test_no_results_never_fall_back_to_unfiltered_recommendations(self):
        no_results = MusicIntelligence(
            self.service, advanced_recommender=lambda *_: {"recommendations": []}
        )
        response, updated = no_results.handle("Recommend music excluding Green Day", self.state)
        self.assertIn("exclusions", response)
        self.assertFalse(updated)
        self.assertFalse(self.state.pending_recommendation_ids)

    def test_old_path_still_works_when_flag_is_off(self):
        legacy = MusicIntelligence(self.service)
        response, updated = legacy.handle("Recommend rock songs", self.state)
        self.assertIn("Here are songs related to your playlist", response)
        self.assertFalse(updated)
        self.assertEqual(self.state.pending_recommendation_ids, ["a", "b", "c"])

    def test_actual_musiccrs_agent_wires_the_option_to_intelligence(self):
        from musiccrs.musiccrs import MusicCRS
        fake_parser = object()
        with (
            patch.dict(os.environ, {"MUSICCRS_ADVANCED_LLM": "true"}),
            patch("musiccrs.musiccrs.get_shared_music_service", return_value=self.service),
            patch("musiccrs.llm_preference_parser.LLMPreferenceParser", return_value=fake_parser),
            patch("musiccrs.llm_recommendations.recommend_with_llm") as recommendation,
        ):
            recommendation.return_value = {"recommendations": [self.service.tracks["a"]]}
            agent = MusicCRS(use_llm=False)
            agent._conversation_state.playlist_ids = ["seed"]
            response, updated = agent._music_intelligence.handle(
                "Recommend energetic rock songs without Billy Talent", agent._conversation_state
            )
            self.assertFalse(updated)
            self.assertIn("Track A", response)
            self.assertEqual(agent._conversation_state.pending_recommendation_ids, ["a"])
            recommendation.assert_called_once()
            args, kwargs = recommendation.call_args
            self.assertIs(args[1], fake_parser)
            self.assertIs(args[2], self.service)
            self.assertEqual(kwargs["limit"], 5)


if __name__ == "__main__":
    unittest.main()

