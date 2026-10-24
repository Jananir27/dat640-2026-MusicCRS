"""Part 4 feedback geometry, dialog routing, hard filters and evidence checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from musiccrs.feedback_language import interpret_feedback
from musiccrs.music_intelligence import MusicIntelligence
from musiccrs.music_service import ConversationState, MusicService
from musiccrs.musiccrs import MusicCRS, MusicCRSPlatform, MusicChatNamespace
from musiccrs.playlist_store import PlaylistStore
from musiccrs.recommendation_explanations import explain_recommendation
from musiccrs.relevance_feedback import FeedbackState, RankedTrack, RocchioFeedback
from tests.test_music_intelligence import FakeCatalog, FakeRetriever


class ToyEmbeddings:
    def get_track_embeddings(self, ids):
        vectors = {"1": [1, 0], "2": [0, 1], "3": [1, 0.05], "4": [0.1, 1],
                   "5": [-1, 0], "6": [0.9, 0.1], "7": [0.5, 0.5], "8": [1, 0.2]}
        return {key: vectors[key] for key in ids if key in vectors}

    def encode_query(self, _query):
        return np.array([1, 1])


class RocchioTests(unittest.TestCase):
    def setUp(self):
        self.model = RocchioFeedback()
        self.feedback = FeedbackState(votes={"like": 1, "dislike": -1})
        self.embeddings = {"like": [1, 0], "dislike": [0, 1], "a": [0, 1], "b": [1, 0]}
        self.candidates = [RankedTrack(key, {"track_id": key}, score)
                           for key, score in (("a", 0.9), ("b", 0.8))]

    def test_query_moves_toward_positive_and_away_from_negative(self):
        original = np.array([1, 1]) / np.sqrt(2)
        updated = self.model.updated_query(original, self.feedback, self.embeddings)
        expected = original + 0.75 * np.array([1, 0]) - 0.25 * np.array([0, 1])
        np.testing.assert_allclose(updated, expected / np.linalg.norm(expected))
        result = self.model.rerank(self.candidates, self.feedback, self.embeddings, original)
        self.assertEqual([t.track_id for t in result], ["b", "a"])
        self.assertGreater(result[0].scores["feedback_delta"], 0)
        self.assertLess(result[1].scores["feedback_delta"], 0)

    def test_no_feedback_preserves_order_and_scores(self):
        result = self.model.rerank(self.candidates, FeedbackState(), self.embeddings, [1, 1])
        self.assertEqual([t.track_id for t in result], ["a", "b"])
        self.assertEqual([t.score for t in result], [0.9, 0.8])

    def test_bad_vectors_are_ignored_and_outputs_are_finite(self):
        vectors = {"like": [float("nan"), 0], "dislike": [0, 0, 0], "a": [0, 0], "b": [1, 0]}
        result = self.model.rerank(self.candidates, self.feedback, vectors, [1, 1])
        self.assertEqual([t.track_id for t in result], ["a", "b"])
        self.assertTrue(all(np.isfinite(t.score) for t in result))

    def test_exclusions_cannot_be_overridden_by_feedback_scores(self):
        feedback = FeedbackState(votes={"a": 1}, excluded_track_ids={"a"}, excluded_artists={"BLOCKED"})
        candidates = self.candidates + [RankedTrack("collab", {"track_id": "collab", "artist_name": ["Allowed", "Blocked"]}, 100)]
        result = self.model.rerank(candidates, feedback, self.embeddings, [1, 1])
        self.assertEqual([t.track_id for t in result], ["b"])

    def test_latest_vote_wins_but_explicit_exclusion_survives(self):
        state = FeedbackState()
        state.rate("a", False)
        self.assertTrue(state.excludes({"track_id": "a"}))
        state.rate("a", True)
        self.assertFalse(state.excludes({"track_id": "a"}))
        state.excluded_track_ids.add("a")
        self.assertTrue(state.excludes({"track_id": "a"}))


class FeedbackConversationTests(unittest.TestCase):
    def setUp(self):
        self.service = MusicService(FakeCatalog(), FakeRetriever(), embedding_provider=ToyEmbeddings())
        self.chat = MusicIntelligence(self.service)
        self.state = ConversationState()
        self.chat.handle("Recommend pop music", self.state)

    def test_compound_feedback_is_not_misread_as_rejection_or_addition(self):
        before = self.state.last_recommendation_ids[:]
        response, changed = self.chat.handle("I like track 2, but I don't like track 4", self.state)
        self.assertFalse(changed)
        self.assertIn("liked", response)
        self.assertEqual(self.state.feedback.votes, {before[1]: 1, before[3]: -1})
        self.assertEqual(self.state.pending_recommendation_ids, before)
        self.assertEqual(self.state.playlist_ids, [])
        self.chat.handle("Recommend pop music", self.state)
        self.assertNotIn(before[3], self.state.pending_recommendation_ids)
        self.assertNotIn(before[1], self.state.pending_recommendation_ids)
        self.assertTrue(any(v.get("feedback_applied") for v in self.state.ranking_evidence.values()))

    def test_invalid_compound_feedback_is_atomic(self):
        response, _ = self.chat.handle("I like track 2 but I don't like track 9", self.state)
        self.assertIn(f"1 to {len(self.state.last_recommendation_ids)}", response)
        self.assertEqual(self.state.feedback.votes, {})

    def test_correction_within_one_utterance_uses_last_vote(self):
        target = self.state.last_recommendation_ids[1]
        self.chat.handle("I don't like track 2 but I like track 2", self.state)
        self.assertEqual(self.state.feedback.votes, {target: 1})

    def test_unavailable_embedding_evidence_does_not_claim_semantic_adaptation(self):
        result = RocchioFeedback().rerank(
            [RankedTrack("a", {"track_id": "a"}, 1)], FeedbackState(votes={"missing": 1}), {}, [1, 0],
        )
        self.assertFalse(result[0].evidence.get("feedback_applied", False))

    def test_ambiguous_pronoun_requests_clarification(self):
        response, _ = self.chat.handle("I like this song", self.state)
        self.assertIn("Which recommendation", response)
        self.assertEqual(self.state.feedback.votes, {})

    def test_numbered_feedback_without_recommendations_gives_a_useful_prompt(self):
        state = ConversationState()
        response, changed = self.chat.handle("I like track 2", state)
        self.assertFalse(changed)
        self.assertIn("request recommendations first", response)
        self.assertNotIn("1 to 0", response)
        self.assertFalse(state.feedback.votes)

    def test_track_by_artist_reference_does_not_dislike_entire_artist(self):
        target = self.state.last_recommendation_ids[0]
        artist = self.service.display_value(self.service.get_track(target)["artist_name"])
        self.chat.handle(f"I don't like the one by {artist}", self.state)
        self.assertEqual(self.state.feedback.votes, {target: -1})
        self.assertFalse(self.state.feedback.disliked_artists)

    def test_more_like_a_number_records_positive_feedback_and_returns_a_new_batch(self):
        before = self.state.recommendation_payload["batch_id"]
        target = self.state.last_recommendation_ids[1]
        response, changed = self.chat.handle("Recommend more songs like track 2", self.state)
        self.assertFalse(changed)
        self.assertEqual(self.state.feedback.votes[target], 1)
        self.assertNotEqual(self.state.recommendation_payload["batch_id"], before)
        self.assertIn("recommendations", response)

    def test_named_artist_dislike_excludes_collaborations(self):
        self.service.catalog.metadata_dict["8"]["artist_name"] = ["Artist Eight", "Metallica"]
        response, _ = self.chat.handle("I don't like the artist Metallica", self.state)
        self.assertIn("excluded artists", response)
        self.chat.handle("Recommend pop music", self.state)
        self.assertNotIn("8", self.state.pending_recommendation_ids)
        self.assertNotIn("2", self.state.pending_recommendation_ids)

    def test_inline_exclusion_is_persistent_and_not_passed_as_positive_query_text(self):
        with patch.object(self.service, "recommend_tracks", wraps=self.service.recommend_tracks) as retrieve:
            self.chat.handle("Recommend pop music but exclude Artist Three", self.state)
        self.assertNotIn("exclude", retrieve.call_args.args[0])
        self.assertNotIn("3", self.state.pending_recommendation_ids)
        self.chat.handle("Recommend pop music", self.state)
        self.assertNotIn("3", self.state.pending_recommendation_ids)

    def test_discovery_does_not_treat_command_filler_as_a_track_name(self):
        with patch.object(self.service, "recommend_tracks", return_value=[]) as retrieve:
            self.service.recommend_from_description(self.state, "Recommend some rock music")
        self.assertEqual(retrieve.call_args.args[0], "rock")
        self.assertEqual(retrieve.call_args.kwargs["request"], "rock")

    def test_rejecting_addition_does_not_invent_negative_feedback(self):
        self.chat.handle("No, don't add any", self.state)
        self.assertEqual(self.state.feedback.votes, {})
        self.assertEqual(self.state.pending_recommendation_ids, [])
        self.assertTrue(self.state.last_recommendation_ids)

    def test_old_batch_rating_targets_its_original_ids_and_rejects_forgery(self):
        payload = self.state.recommendation_payload
        track_id = payload["tracks"][0]["track_id"]
        self.chat.handle("Recommend rock music", self.state)
        data = {"batch_id": payload["batch_id"], "track_id": track_id, "event": "thumbs_up"}
        self.chat.handle("/feedback " + json.dumps(data), self.state)
        self.assertEqual(self.state.feedback.votes, {track_id: 1})
        data["track_id"] = "unknown"
        response, _ = self.chat.handle("/feedback " + json.dumps(data), self.state)
        self.assertIn("does not match", response)
        self.assertEqual(self.state.feedback.votes, {track_id: 1})

    def test_preferences_are_session_isolated_and_can_be_reset(self):
        self.chat.handle("I like track 1", self.state)
        other = ConversationState()
        self.assertFalse(other.feedback.votes)
        self.chat.handle("Reset my feedback", self.state)
        self.assertFalse(self.state.feedback.votes)

    def test_show_something_different_skips_previous_batch_without_negative_votes(self):
        before = self.state.last_recommendation_ids[:]
        self.chat.handle("Show me something different", self.state)
        self.assertFalse(set(before) & set(self.state.last_recommendation_ids))
        self.assertFalse(self.state.feedback.votes)

    def test_explanations_only_claim_verified_features(self):
        track = self.service.get_track("3")
        reason, facts = explain_recommendation(track, "chill jazz", [], self.state.feedback)
        self.assertNotIn("jazz", reason)
        self.assertNotIn("chill", reason)
        self.assertEqual(facts["requested_genres"], [])
        self.assertEqual(facts["requested_moods"], [])
        self.assertNotIn("liked", reason)
        reason, facts = explain_recommendation(track, "pop", [self.service.get_track("1")])
        self.assertIn("pop", reason)
        self.assertEqual(facts["requested_genres"], ["pop"])


class FeedbackSocketTests(unittest.TestCase):
    def test_switching_saved_playlists_keeps_feedback_but_discards_old_batches(self):
        service = MusicService(FakeCatalog(), FakeRetriever())
        with tempfile.TemporaryDirectory() as directory, patch("musiccrs.musiccrs.get_shared_music_service", return_value=service):
            store = PlaylistStore(Path(directory) / "playlists.sqlite3")
            agent = MusicCRS(use_llm=False)
            agent.attach_device("test-device", store)
            agent._conversation_state.feedback.rate("1", True)
            agent._conversation_state.recommendation_batches["old"] = ["2"]
            store.create("test-device", "Other")
            agent._restore_playlist()
            self.assertEqual(agent._conversation_state.feedback.votes, {"1": 1})
            self.assertEqual(agent._conversation_state.recommendation_batches, {})

    def test_another_socket_cannot_rate_a_batch_from_the_first_user(self):
        service = MusicService(FakeCatalog(), FakeRetriever())
        with patch("musiccrs.musiccrs.get_shared_music_service", return_value=service):
            platform = MusicCRSPlatform(MusicCRS)
            platform.socketio.on_namespace(MusicChatNamespace("/", platform))
            first = platform.socketio.test_client(platform.app)
            second = platform.socketio.test_client(platform.app)
            try:
                first.get_received()
                second.get_received()
                first.emit("message", {"message": "Recommend pop music"})
                message = first.get_received()[-1]["args"]["message"]
                batch = json.loads(next(a["value"] for act in message["dialogue_acts"] if act["intent"] == "RECOMMENDATIONS"
                                        for a in act["annotations"]))
                second.emit("feedback", {"batch_id": batch["batch_id"], "track_id": batch["tracks"][0]["track_id"], "event": "thumbs_down"})
                self.assertIn("does not match", second.get_received()[-1]["args"]["message"]["text"])
            finally:
                first.disconnect()
                second.disconnect()

    def test_real_socket_event_updates_preferences_without_playlist_mutation(self):
        service = MusicService(FakeCatalog(), FakeRetriever(), embedding_provider=ToyEmbeddings())
        with patch("musiccrs.musiccrs.get_shared_music_service", return_value=service):
            platform = MusicCRSPlatform(MusicCRS)
            platform.socketio.on_namespace(MusicChatNamespace("/", platform))
            client = platform.socketio.test_client(platform.app)
            try:
                client.get_received()
                client.emit("message", {"message": "Recommend pop music"})
                events = client.get_received()
                message = events[-1]["args"]["message"]
                annotation = next(a for act in message["dialogue_acts"] if act["intent"] == "RECOMMENDATIONS"
                                  for a in act["annotations"] if a["slot"] == "recommendations_json")
                batch = json.loads(annotation["value"])
                track_id = batch["tracks"][0]["track_id"]
                client.emit("feedback", {"batch_id": batch["batch_id"], "track_id": track_id, "event": "thumbs_down"})
                self.assertIn("disliked", client.get_received()[-1]["args"]["message"]["text"])
                client.emit("message", {"message": "Recommend pop music"})
                message = client.get_received()[-1]["args"]["message"]
                annotation = next(a for act in message["dialogue_acts"] if act["intent"] == "RECOMMENDATIONS"
                                  for a in act["annotations"] if a["slot"] == "recommendations_json")
                self.assertNotIn(track_id, [t["track_id"] for t in json.loads(annotation["value"])["tracks"]])
                self.assertEqual(client.emit("feedback", {}, callback=True)["ok"], False)
            finally:
                client.disconnect()


if __name__ == "__main__":
    unittest.main()
