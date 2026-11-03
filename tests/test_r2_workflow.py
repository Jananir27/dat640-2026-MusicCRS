"""R2 acceptance tests across parsing, R1 integration and conversation state.

Synthetic geometry verifies feedback direction, not real musical relevance.
No network calls or external LLM credentials are used.
"""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from musiccrs.feedback_workflow import FeedbackReferenceError, remember_r1_exclusions
from musiccrs.llm_recommendations import recommend_with_llm
from musiccrs.music_intelligence import MusicIntelligence
from musiccrs.music_service import ConversationState, MusicService
from musiccrs.relevance_feedback import RankedTrack
from tests.test_feedback_retrieval import LexicalFixture
from tests.test_music_intelligence import FakeCatalog, FakeRetriever
from tests.test_relevance_feedback import ToyEmbeddings


class FixedParser:
    def __init__(self, **overrides):
        self.overrides = overrides
        self.calls = []

    def parse(self, request):
        self.calls.append(request)
        return {"genre": ["jazz" if "jazz" in request.lower() else "rock"], "mood": [],
                "preferred_artists": [], "excluded_artists": [], "excluded_tracks": [],
                "use_playlist_context": True, "avoid_playlist_artists": False, **self.overrides}


def indexed_service():
    rows = [
        ("original", "Classic", ["rock", "classic rock"], [1, 0]),
        ("liked", "Preferred", ["rock", "alternative rock", "nu metal"], [0, 1]),
        ("related", "Preferred", ["rock", "alternative rock", "nu metal"], [.1, 1]),
        ("disliked", "Rejected", ["rock", "classic rock"], [1, 0]),
        ("blocked", "Blocked", ["rock", "nu metal"], [0, 1]),
        ("wrong_genre", "Preferred", ["jazz"], [0, 1]),
        ("playlist", "Seed", ["rock"], [1, 0]),
        ("duplicate", "Preferred", ["rock", "nu metal"], [0, 1]),
    ]
    catalog = FakeCatalog()
    catalog.metadata_dict = {t: {"track_id": t, "track_name": ["liked" if t == "duplicate" else t],
                                "artist_name": [artist], "tag_list": tags}
                             for t, artist, tags, _ in rows}
    matrix = np.array([row[3] for row in rows], dtype=np.float32)
    dense = SimpleNamespace(track_ids=[row[0] for row in rows], track_embeddings=matrix,
                            encode_query=lambda _: np.array([1., 0.]))
    lexical = LexicalFixture(["original", "related", "liked", "disliked", "blocked", "wrong_genre", "playlist", "duplicate"])
    retriever = SimpleNamespace(
        hybrid=SimpleNamespace(bm25=lexical, dense=dense, candidate_k=1, rrf_k=60, dense_weight=.25),
        text_to_item_retrieval=lambda query, topk: ["original", "liked", "disliked"][:topk],
    )
    return MusicService(catalog, retriever)


class FeedbackLanguageAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.service = MusicService(FakeCatalog(), FakeRetriever(), embedding_provider=ToyEmbeddings())
        self.chat = MusicIntelligence(self.service)
        self.state = ConversationState()
        self.chat.handle("Recommend pop music", self.state)
        self.ids = self.state.last_recommendation_ids[:]

    def test_polite_and_short_more_like_requests_resolve_the_number(self):
        for text in ("Please recommend more songs like track 2", "Could you suggest more like track 2"):
            with self.subTest(text=text):
                self.setUp()
                self.chat.handle(text, self.state)
                self.assertEqual(self.state.feedback.votes[self.ids[1]], 1)
                self.assertNotIn(self.ids[1], self.state.last_recommendation_ids)

    def test_more_like_a_focused_song_and_an_ambiguous_song(self):
        response, _ = self.chat.handle("Recommend more songs like this song", self.state)
        self.assertIn("Which recommendation", response)
        self.assertEqual(self.state.feedback.votes, {})
        self.state.last_question_track_id = self.ids[1]
        self.chat.handle("Recommend more songs like this song", self.state)
        self.assertEqual(self.state.feedback.votes[self.ids[1]], 1)

    def test_title_and_artist_more_like_request_records_a_like(self):
        target = self.service.get_track(self.ids[1])
        message = f"Please recommend more songs like {target['track_name'][0]} by {target['artist_name'][0]}"
        self.chat.handle(message, self.state)
        self.assertEqual(self.state.feedback.votes[self.ids[1]], 1)

    def test_literal_song_titles_are_not_split_into_feedback_clauses(self):
        for title in ("Love and Hate", "Peace and Love Again"):
            with self.subTest(title=title):
                self.service.get_track(self.ids[1])["track_name"] = [title]
                self.chat.handle(f"I like {title}", self.state)
                self.assertEqual(self.state.feedback.votes, {self.ids[1]: 1})

    def test_past_tense_compound_feedback_and_correction(self):
        self.chat.handle("I liked track 2 and I didn’t like track 4", self.state)
        self.assertEqual(self.state.feedback.votes, {self.ids[1]: 1, self.ids[3]: -1})
        self.chat.handle("I did not like track 2 but I like track 2", self.state)
        self.assertEqual(self.state.feedback.votes[self.ids[1]], 1)

    def test_feedback_and_new_genre_are_processed_in_either_order(self):
        for message in ("I like track 2 and recommend jazz music", "Recommend jazz music and I like track 2"):
            with self.subTest(message=message):
                self.setUp()
                with patch.object(self.service, "recommend_from_description", return_value=[]) as recommend:
                    self.chat.handle(message, self.state)
                self.assertEqual(self.state.feedback.votes, {self.ids[1]: 1})
                self.assertEqual(recommend.call_args.args[1], "recommend jazz music" if message.startswith("I") else "Recommend jazz music")

    def test_invalid_compound_feedback_with_new_request_is_atomic(self):
        for message in ("I like track 2 and I did not like track 99 and recommend jazz",
                        "Recommend jazz and I like track 99"):
            response, _ = self.chat.handle(message, self.state)
            self.assertIn("1 to", response)
            self.assertEqual(self.state.feedback.votes, {})
            self.assertEqual(self.state.last_recommendation_ids, self.ids)

    def test_repeated_ratings_are_not_accumulated(self):
        self.chat.handle("I like track 2", self.state)
        self.chat.handle("I like track 2", self.state)
        self.assertEqual(self.state.feedback.votes, {self.ids[1]: 1})

    def test_playlist_more_like_request_does_not_invent_track_votes(self):
        self.state.playlist_ids = ["1"]
        self.chat.handle("Recommend more songs like my playlist", self.state)
        self.assertEqual(self.state.feedback.votes, {})

    def test_empty_new_results_clear_old_numbered_choices_but_keep_button_batches(self):
        batch = self.state.recommendation_payload["batch_id"]
        self.chat.handle("Recommend jazz music", self.state)
        self.assertEqual(self.state.pending_recommendation_ids, [])
        self.assertEqual(self.state.last_recommendation_ids, [])
        self.assertIsNone(self.state.recommendation_payload)
        self.assertIn(batch, self.state.recommendation_batches)
        response, _ = self.chat.handle("I like track 2", self.state)
        self.assertIn("request recommendations first", response)

    def test_expired_button_batches_are_rejected_without_changing_feedback(self):
        batch = self.state.recommendation_payload
        for _ in range(20):
            self.service.remember_recommendations(self.state, [self.service.get_track("1")], "pop")
        import json
        response, _ = self.chat.handle("/feedback " + json.dumps({
            "batch_id": batch["batch_id"], "track_id": self.ids[0], "event": "thumbs_up"}), self.state)
        self.assertIn("does not match", response)
        self.assertEqual(self.state.feedback.votes, {})

    def test_inline_numbered_exclusion_refers_to_the_previous_displayed_batch(self):
        self.chat.handle("Recommend pop without track 2", self.state)
        self.assertIn(self.ids[1], self.state.feedback.excluded_track_ids)
        self.assertNotIn(self.ids[1], self.state.last_recommendation_ids)


class R1FeedbackIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.service = indexed_service()
        self.state = ConversationState(playlist_ids=["playlist"])
        self.state.feedback.rate("liked", True)
        self.state.feedback.rate("disliked", False)
        self.state.feedback.excluded_artists.add("blocked")

    def test_live_service_retrieves_beyond_r1_pool_and_keeps_constraints(self):
        matrix = self.service._retriever.hybrid.dense.track_embeddings.copy()
        tracks = self.service.rerank_feedback([self.service.get_track("original")], self.state, "rock", 1)
        self.assertEqual([t["track_id"] for t in tracks], ["related"])
        evidence = self.state.ranking_evidence["related"]
        self.assertEqual(evidence["feedback_path"], "adaptive_catalog")
        self.assertTrue(evidence["outside_r1_pool"])
        self.assertTrue(evidence["feedback_applied"])
        np.testing.assert_array_equal(matrix, self.service._retriever.hybrid.dense.track_embeddings)

    def test_advanced_route_keeps_feedback_with_and_without_exclusions(self):
        parser = FixedParser()
        chat = MusicIntelligence(self.service, advanced_recommender=lambda request, state:
                                 recommend_with_llm(request, parser, self.service, state))
        for request in ("Recommend rock music", "Recommend rock music without artist Blocked"):
            response, _ = chat.handle(request, self.state)
            self.assertIn("related", self.state.last_recommendation_ids)
            self.assertEqual(self.state.ranking_evidence["related"]["feedback_path"], "adaptive_catalog")
            self.assertFalse({"liked", "duplicate", "disliked", "blocked", "wrong_genre", "playlist"}
                             & set(self.state.last_recommendation_ids))
        self.assertEqual(len(parser.calls), 2)

    def test_new_genre_constrains_expansion_even_with_old_rock_likes(self):
        tracks = self.service.rerank_feedback([], self.state, "jazz", genres={"jazz"})
        self.assertEqual([t["track_id"] for t in tracks], ["wrong_genre"])

    def test_standard_playlist_route_honors_a_new_genre_with_existing_feedback(self):
        chat = MusicIntelligence(self.service)
        chat.handle("Recommend jazz music", self.state)
        self.assertEqual(self.state.last_recommendation_ids, ["wrong_genre"])

    def test_negative_only_feedback_uses_vectors_without_inventing_likes(self):
        self.state.feedback.votes = {"liked": -1}
        self.service._retriever.hybrid.dense.encode_query = lambda _: np.array([1., 1.])
        result = self.service.rerank_feedback([], self.state, "rock")
        self.assertTrue(result)
        self.assertNotIn("liked", [t["track_id"] for t in result])
        self.assertTrue(any(abs(e["scores"]["feedback_delta"]) > 1e-6
                            for e in self.state.ranking_evidence.values()))
        self.assertEqual(self.state.feedback.votes, {"liked": -1})

    def test_missing_liked_vector_is_reported_without_a_similarity_claim(self):
        dense = self.service._retriever.hybrid.dense
        dense.track_embeddings[dense.track_ids.index("liked")] = [float("nan"), 0]
        result = self.service.rerank_feedback([], self.state, "rock")
        self.assertTrue(result)
        for evidence in self.state.ranking_evidence.values():
            self.assertIn("liked", evidence["unavailable_feedback_ids"])
            self.assertNotIn("liked_track_id", evidence)
            self.assertTrue(all(np.isfinite(v) for v in evidence["scores"].values()))

    def test_reset_removes_preferences_even_when_the_shared_index_is_cached(self):
        self.service.rerank_feedback([], self.state, "rock")
        index = self.service._feedback_adapter._index
        MusicIntelligence(self.service).handle("Reset my feedback", self.state)
        result = self.service.rerank_feedback([self.service.get_track("original")], self.state, "rock")
        self.assertEqual([t["track_id"] for t in result], ["original"])
        self.assertEqual(self.state.feedback.votes, {})
        self.assertEqual(self.state.feedback.excluded_artists, set())
        self.assertIs(self.service._feedback_adapter._index, index)

    def test_unavailable_vector_provider_preserves_hard_filters_without_semantic_claims(self):
        provider = ToyEmbeddings()
        provider.get_track_embeddings = lambda _: {}
        service = MusicService(FakeCatalog(), FakeRetriever(), embedding_provider=provider)
        state = ConversationState()
        state.feedback.rate("1", True)
        state.feedback.excluded_artists.add("metallica")
        tracks = service.rerank_feedback(list(service.catalog.metadata_dict.values()), state, "pop")
        self.assertFalse({"1", "2"} & {t["track_id"] for t in tracks})
        self.assertTrue(all(not e.get("feedback_applied") for e in state.ranking_evidence.values()))

    def test_no_feedback_preserves_r1_order_and_does_not_build_an_index(self):
        state = ConversationState()
        tracks = [self.service.get_track(t) for t in ("disliked", "original", "related")]
        result = self.service.rerank_feedback(tracks, state, "rock")
        self.assertEqual(result, tracks)
        self.assertIsNone(self.service._feedback_adapter._index)
        self.assertTrue(all(not e.get("feedback_applied") for e in state.ranking_evidence.values()))

    def test_pool_only_genre_filtering_does_not_scan_unrelated_catalog_rows(self):
        from musiccrs.adaptive_retrieval import metadata_features
        with patch("musiccrs.feedback_workflow.metadata_features", wraps=metadata_features) as features:
            result = self.service.rerank_feedback([self.service.get_track("original")],
                                                  ConversationState(), "rock", genres={"rock"})
        self.assertEqual([t["track_id"] for t in result], ["original"])
        self.assertEqual(features.call_count, 1)

    def test_shared_index_does_not_leak_feedback_between_users(self):
        self.service.rerank_feedback([self.service.get_track("original")], self.state, "rock")
        other = ConversationState()
        tracks = self.service.rerank_feedback([self.service.get_track("original")], other, "rock")
        self.assertEqual([t["track_id"] for t in tracks], ["original"])
        self.assertEqual(other.feedback.votes, {})
        self.assertFalse(other.ranking_evidence["original"].get("feedback_applied"))

    def test_explicit_allowlist_and_exclusion_win_over_a_positive_vote(self):
        tracks = self.service.rerank_feedback([], self.state, "rock", allowed_ids={"original"})
        self.assertEqual([t["track_id"] for t in tracks], ["original"])
        self.state.feedback.excluded_track_ids.update(self.service.catalog.metadata_dict)
        self.assertEqual(self.service.rerank_feedback([], self.state, "rock"), [])

    def test_invalid_embedding_fails_gracefully_and_keeps_preferences(self):
        self.service._retriever.hybrid.dense.encode_query = lambda _: [1, 0, 0]
        chat = MusicIntelligence(self.service)
        before = deepcopy(self.state.feedback)
        with self.assertLogs("musiccrs.music_intelligence", level="ERROR"):
            response, _ = chat.handle("Recommend rock music", self.state)
        self.assertIn("feedback is saved", response)
        self.assertEqual(self.state.feedback, before)
        self.assertEqual(self.state.last_recommendation_ids, [])
        self.assertEqual(self.state.ranking_evidence, {})

    def test_ranked_track_contract_preserves_real_source_scores(self):
        service = MusicService(FakeCatalog(), FakeRetriever(), embedding_provider=ToyEmbeddings())
        candidate = RankedTrack("3", service.get_track("3"), .42, scores={"hybrid": .42})
        ranked = service._feedback_adapter.rank([candidate], ConversationState(), "pop")
        self.assertEqual(ranked[0].scores["hybrid"], .42)
        self.assertEqual(ranked[0].score, .42)


class TemporarySkipAndFailureTests(unittest.TestCase):
    def setUp(self):
        self.service = MusicService(FakeCatalog(), FakeRetriever())
        self.parser = FixedParser(genre=["pop"])
        self.chat = MusicIntelligence(self.service, advanced_recommender=lambda request, state:
                                      recommend_with_llm(request, self.parser, self.service, state))
        self.state = ConversationState(playlist_ids=["5"])

    def test_different_skips_one_request_then_allows_the_original_batch_again(self):
        self.chat.handle("Recommend pop music", self.state)
        original = self.state.last_recommendation_ids[:]
        self.chat.handle("Show me something different", self.state)
        self.assertFalse(set(original) & set(self.state.last_recommendation_ids))
        self.assertEqual(self.state.skip_recommendation_ids, set())
        self.assertEqual(self.state.feedback.votes, {})
        self.chat.handle("Recommend pop music", self.state)
        self.assertEqual(self.state.last_recommendation_ids, original)

    def test_different_clears_temporary_skips_after_provider_failure(self):
        self.chat.handle("Recommend pop music", self.state)
        self.chat._advanced_recommender = lambda *_: (_ for _ in ()).throw(RuntimeError("offline fixture"))
        with self.assertLogs("musiccrs.music_intelligence", level="ERROR"):
            self.chat.handle("Show me something different", self.state)
        self.assertEqual(self.state.skip_recommendation_ids, set())
        self.assertEqual(self.state.feedback.votes, {})


class PlaylistCreationRegressionTests(unittest.TestCase):
    def test_replacing_a_playlist_can_reuse_its_tracks_while_preserving_explicit_bans(self):
        service = MusicService(FakeCatalog(), FakeRetriever())
        chat = MusicIntelligence(service)
        state = ConversationState()
        _, updated = chat.handle("Create a playlist for a short commute", state)
        self.assertTrue(updated)
        self.assertTrue(state.playlist_ids)
        previous = set(state.playlist_ids)
        _, updated = chat.handle("Create a playlist for relaxing studying", state)
        self.assertTrue(updated)
        self.assertTrue(previous & set(state.playlist_ids))
        state.feedback.excluded_track_ids.add("2")
        _, updated = chat.handle("Create a playlist for a long road trip", state)
        self.assertTrue(updated)
        self.assertNotIn("2", state.playlist_ids)
        self.assertIn("2", state.feedback.excluded_track_ids)


class PersistentR1ConstraintTests(unittest.TestCase):
    def setUp(self):
        self.service = indexed_service()
        self.state = ConversationState()
        self.prefs = FixedParser().parse("rock")

    def test_llm_artist_exclusion_survives_subsequent_feedback_expansion(self):
        self.prefs["excluded_artists"] = ["Preferred"]
        remember_r1_exclusions(self.service, self.state, self.prefs)
        self.state.feedback.rate("liked", True)
        tracks = self.service.rerank_feedback([], self.state, "rock")
        self.assertTrue(tracks)
        self.assertTrue(all("Preferred" not in t["artist_name"] for t in tracks))

    def test_qualified_track_exclusion_does_not_remove_a_different_artist(self):
        for track_id, artist in (("liked", "Linkin Park"), ("related", "21 Savage")):
            self.service.catalog.metadata_dict[track_id].update(track_name=["Numb"], artist_name=[artist])
        self.prefs["excluded_tracks"] = ["Numb"]
        remember_r1_exclusions(self.service, self.state, self.prefs,
                               "Recommend rock, no Numb by Linkin Park")
        self.assertIn("liked", self.state.feedback.excluded_track_ids)
        self.assertNotIn("related", self.state.feedback.excluded_track_ids)

    def test_ambiguous_constraints_are_atomic(self):
        for track_id in ("original", "related"):
            self.service.catalog.metadata_dict[track_id]["track_name"] = ["Shared Title"]
        self.prefs.update(excluded_artists=["Classic"], excluded_tracks=["Shared Title"])
        with self.assertRaises(FeedbackReferenceError):
            remember_r1_exclusions(self.service, self.state, self.prefs)
        self.assertEqual(self.state.feedback.excluded_artists, set())
        self.assertEqual(self.state.feedback.excluded_track_ids, set())

    def test_a_title_containing_by_is_not_mistaken_for_an_artist_qualifier(self):
        self.service.catalog.metadata_dict["related"]["track_name"] = ["Stand By Me"]
        self.prefs["excluded_tracks"] = ["Stand By Me"]
        remember_r1_exclusions(self.service, self.state, self.prefs)
        self.assertEqual(self.state.feedback.excluded_track_ids, {"related"})

    def test_artist_identity_does_not_block_a_different_artist_with_overlapping_words(self):
        self.service.catalog.metadata_dict["original"]["artist_name"] = ["War"]
        self.service.catalog.metadata_dict["related"]["artist_name"] = ["The War On Drugs"]
        self.prefs["excluded_artists"] = ["War"]
        remember_r1_exclusions(self.service, self.state, self.prefs)
        self.assertTrue(self.state.feedback.excludes(self.service.get_track("original")))
        self.assertFalse(self.state.feedback.excludes(self.service.get_track("related")))


if __name__ == "__main__":
    unittest.main()
