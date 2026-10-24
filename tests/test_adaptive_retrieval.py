"""Behavioral tests for the isolated adaptive scorer and fair evaluation."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from musiccrs.adaptive_retrieval import AdaptiveConfig, AdaptiveMusicRetriever, metadata_features, request_preferences
from musiccrs.relevance_feedback import FeedbackState
from retrieval.evaluation.adaptive import metrics
from tests.test_feedback_retrieval import LexicalFixture


class AdaptiveRetrievalTests(unittest.TestCase):
    def setUp(self):
        rows = [
            ("original", "Generic", ["rock", "classic rock"], [1, 0]),
            ("related", "Preferred", ["rock", "alternative rock", "nu metal", "energetic"], [.1, 1]),
            ("liked", "Preferred", ["rock", "alternative rock", "nu metal"], [0, 1]),
            ("disliked", "Rejected", ["rock", "classic rock"], [1, 0]),
            ("blocked", "Blocked", ["rock", "nu metal"], [0, 1]),
            ("wrong_genre", "Preferred", ["electronic"], [0, 1]),
        ]
        self.catalog = SimpleNamespace(metadata_dict={t: {
            "track_id": t, "track_name": [t], "artist_name": [artist], "tag_list": tags
        } for t, artist, tags, _ in rows})
        self.dense = SimpleNamespace(track_ids=[r[0] for r in rows],
                                     track_embeddings=np.array([r[3] for r in rows], dtype=np.float32),
                                     encode_query=lambda text: [1., 0.])
        self.lexical = LexicalFixture(["original", "wrong_genre", "related"])
        self.retriever = AdaptiveMusicRetriever(self.catalog, self.lexical, self.dense,
                                                candidate_k=1, config=AdaptiveConfig(influence=0.8))
        self.state = FeedbackState(votes={"liked": 1, "disliked": -1}, excluded_artists={"Blocked"})

    def test_liked_style_and_new_catalog_candidate_can_overcome_lexical_prior(self):
        before = self.retriever.retrieve("Recommend rock music", self.state, mode="baseline", topk=1)
        after = self.retriever.retrieve("Recommend rock music", self.state, topk=1)
        self.assertEqual([t.track_id for t in before.tracks], ["original"])
        self.assertEqual([t.track_id for t in after.tracks], ["related"])
        self.assertIn("related", after.new_candidate_ids)
        self.assertTrue(after.tracks[0].evidence["liked_artist_match"])

    def test_genre_artist_track_and_duplicate_song_exclusions_apply_to_every_mode(self):
        self.catalog.metadata_dict["wrong_genre"]["track_name"] = ["liked"]
        retriever = AdaptiveMusicRetriever(self.catalog, self.lexical, self.dense)
        for mode in ("baseline", "rerank_only", "feedback_retrieval"):
            with self.subTest(mode=mode):
                result = retriever.retrieve("rock music", self.state, mode=mode, topk=10)
                self.assertEqual({t.track_id for t in result.tracks}, {"original", "related"})
                self.assertTrue(all("rock" in t.evidence["requested_genres"] for t in result.tracks))

    def test_artist_name_is_not_interpreted_as_a_genre_constraint(self):
        self.assertEqual(request_preferences("Recommend music similar to Daft Punk")[0], set())
        self.assertEqual(request_preferences("Recommend rock like Daft Punk")[0], {"rock"})

    def test_composite_and_subgenre_metadata_are_recognized(self):
        self.assertIn("hip hop", metadata_features({"tag_list": ["Hip-Hop/Rap"]})[0])
        self.assertIn("r&b", metadata_features({"tag_list": ["R&B/Soul"]})[0])
        self.assertIn("pop", metadata_features({"tag_list": ["Latin Pop"]})[0])
        self.assertIn("rock", metadata_features({"tag_list": ["Alternative Rock"]})[0])

    def test_no_feedback_keeps_all_three_modes_identical_and_does_not_mutate_state(self):
        empty = FeedbackState(excluded_artists={"Blocked"})
        before, matrix = deepcopy(empty), self.dense.track_embeddings.copy()
        modes = [self.retriever.retrieve("rock music", empty, mode=mode)
                 for mode in ("baseline", "rerank_only", "feedback_retrieval")]
        self.assertEqual([[t.track_id for t in r.tracks] for r in modes],
                         [[t.track_id for t in modes[0].tracks]] * 3)
        self.assertTrue(all(not t.evidence["feedback_applied"] for r in modes for t in r.tracks))
        self.assertEqual(empty, before)
        np.testing.assert_array_equal(matrix, self.dense.track_embeddings)

    def test_hard_allowlist_cannot_be_overridden_by_feedback_and_weights_are_validated(self):
        result = self.retriever.retrieve("rock music", self.state, allowed_ids={"original"})
        self.assertEqual([t.track_id for t in result.tracks], ["original"])
        for weight in (-1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                AdaptiveConfig(influence=weight)

    def test_negative_feedback_penalizes_similar_candidates_beyond_hiding_the_seed(self):
        positive = FeedbackState(votes={"liked": 1}, excluded_track_ids={"disliked"}, excluded_artists={"Blocked"})
        negative = deepcopy(positive)
        negative.rate("disliked", False)
        before = {t.track_id: t for t in self.retriever.retrieve("rock", positive, topk=10).tracks}
        after = {t.track_id: t for t in self.retriever.retrieve("rock", negative, topk=10).tracks}
        self.assertEqual(before.keys(), after.keys())
        original_penalty = before["original"].scores["context"] - after["original"].scores["context"]
        related_penalty = before["related"].scores["context"] - after["related"].scores["context"]
        self.assertGreater(original_penalty, related_penalty)
        self.assertGreater(related_penalty, 0)

    def test_explicit_artist_likes_can_affect_ranking_without_track_votes(self):
        self.dense.track_embeddings[1] = [1, 0]
        retriever = AdaptiveMusicRetriever(self.catalog, self.lexical, self.dense,
                                           dense_weight=1, config=AdaptiveConfig(influence=0.8))
        state = FeedbackState(liked_artists={"Preferred"}, excluded_track_ids={"liked", "disliked"},
                               excluded_artists={"Blocked"})
        before = retriever.retrieve("rock", state, mode="baseline", topk=2)
        after = retriever.retrieve("rock", state, mode="rerank_only", topk=2)
        self.assertEqual(before.tracks[0].track_id, "original")
        self.assertEqual(after.tracks[0].track_id, "related")
        self.assertTrue(after.tracks[0].evidence["liked_artist_match"])

    def test_common_judgment_pool_and_missing_judgments(self):
        queries = [{"id": "q", "liked_ids": ["seed"]}]
        runs = [{"query_id": "q", "method": method, "predicted_track_ids": tracks,
                 "latency_ms": [1], "final_excluded_count": 0}
                for method, tracks in (("before", ["partial"]), ("after", ["strong"]))]
        document = {"queries": queries, "runs": runs, "comparisons": [("before", "after")]}
        rows = [{"query_id": "q", "track_id": t, "grade": grade, "assessor": "fixture"}
                for t, grade in (("partial", "1"), ("strong", "2"))]
        result = metrics(document, rows)
        self.assertGreater(result["paired_feedback_delta"]["after minus before"]["ndcg@5"], 0)
        self.assertEqual(result["paired_feedback_delta"]["after minus before"]["precision@5"], 0)
        rows[1]["grade"] = ""
        with self.assertRaisesRegex(ValueError, "explicit grades"):
            metrics(document, rows)


if __name__ == "__main__":
    unittest.main()
