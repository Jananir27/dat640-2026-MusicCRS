"""Independent retrieval tests; no downloads or changes to the existing path."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from musiccrs.feedback_retrieval import FeedbackAwareRetriever, MODES
from musiccrs.relevance_feedback import FeedbackState


class LexicalFixture:
    def __init__(self, ids):
        self.track_ids = list(ids)
        self.calls = []

    def text_to_item_retrieval(self, query, topk):
        self.calls.append((query, topk))
        return self.track_ids[:topk]


class FeedbackRetrievalTests(unittest.TestCase):
    def setUp(self):
        ids = ["a", "b", "liked", "disliked", "blocked"]
        self.catalog = SimpleNamespace(metadata_dict={t: {
            "track_id": t, "track_name": [t], "artist_name": ["Artist " + t], "tag_list": ["rock"]
        } for t in ids})
        self.catalog.metadata_dict["blocked"]["artist_name"] = ["Guest", "Blocked"]
        self.dense = SimpleNamespace(
            track_ids=ids, track_embeddings=np.array([[1, 0], [.1, 1], [0, 1], [1, 0], [.2, 1]], dtype=np.float32),
            encode_query=lambda _: np.array([1., 0.]))
        self.lexical = LexicalFixture(["a"])
        self.retriever = FeedbackAwareRetriever(self.catalog, self.lexical, self.dense, candidate_k=1)
        self.feedback = FeedbackState(votes={"liked": 1, "disliked": -1}, excluded_artists={"Blocked"})

    def test_full_catalog_feedback_retrieves_a_track_outside_original_pool(self):
        before = self.retriever.retrieve("rock", self.feedback, mode="rerank_only", topk=1)
        after = self.retriever.retrieve("rock", self.feedback, topk=1)
        self.assertEqual(before.original_candidate_ids, ("a",))
        self.assertEqual([t.track_id for t in before.tracks], ["a"])
        self.assertEqual([t.track_id for t in after.tracks], ["b"])
        self.assertEqual(after.new_candidate_ids, ("b",))
        self.assertNotIn("b", before.candidate_ids)
        self.assertTrue(after.tracks[0].evidence["new_to_original_pool"])
        self.assertTrue(after.tracks[0].evidence["feedback_query_used_for_retrieval"])
        self.assertTrue(after.tracks[0].evidence["feedback_applied"])
        self.assertEqual(self.lexical.calls, [("rock", 1), ("rock", 1)])

    def test_all_three_modes_apply_the_same_exclusions_including_collaborators(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                result = self.retriever.retrieve("rock", self.feedback, topk=10, mode=mode)
                self.assertEqual({t.track_id for t in result.tracks}, {"a", "b"})
                self.assertFalse({"blocked", "liked", "disliked"} & set(result.candidate_ids))

    def test_feedback_query_is_not_applied_twice_during_reranking(self):
        result = self.retriever.retrieve("rock", self.feedback, topk=1)
        # q' = normalize([1,0] + .75*[0,1] - .25*[1,0])
        candidate = self.retriever.matrix[self.retriever.positions["b"]]
        expected = float(candidate @ (np.array([1., 1.]) / np.sqrt(2)))
        self.assertAlmostEqual(result.tracks[0].evidence["updated_cosine"], expected, places=6)

    def test_no_feedback_preserves_baseline_and_has_no_new_candidates(self):
        results = [self.retriever.retrieve("rock", FeedbackState(), mode=mode) for mode in MODES]
        for result in results:
            self.assertEqual([t.track_id for t in result.tracks], [t.track_id for t in results[0].tracks])
            self.assertEqual([t.score for t in result.tracks], [t.score for t in results[0].tracks])
            self.assertFalse(result.query_changed)
            self.assertFalse(result.new_candidate_ids)
            self.assertTrue(all(not t.evidence["feedback_applied"] for t in result.tracks))

    def test_exclusions_are_applied_before_topk_and_lexical_results_are_refilled(self):
        lexical = LexicalFixture(["liked", "disliked", "blocked", "a", "b"])
        retriever = FeedbackAwareRetriever(self.catalog, lexical, self.dense, candidate_k=2)
        result = retriever.retrieve("rock", self.feedback, topk=2)
        self.assertEqual({t.track_id for t in result.tracks}, {"a", "b"})
        self.assertEqual([k for _, k in lexical.calls], [2, 4, 5])
        self.assertEqual({t.evidence["lexical_rank"] for t in result.tracks}, {1, 2})

    def test_structured_allowlist_and_playlist_exclusions_cannot_be_overridden(self):
        result = self.retriever.retrieve("rock", self.feedback, allowed_ids={"a", "b"}, excluded_ids={"b"})
        self.assertEqual([t.track_id for t in result.tracks], ["a"])
        empty = self.retriever.retrieve("rock", self.feedback, allowed_ids=set())
        self.assertEqual(empty.tracks, [])

    def test_feedback_and_source_embeddings_are_not_mutated_or_retained_between_users(self):
        state = deepcopy(self.feedback)
        matrix = self.dense.track_embeddings.copy()
        self.retriever.retrieve("rock", self.feedback)
        clean = self.retriever.retrieve("rock", FeedbackState(), topk=1)
        self.assertEqual(self.feedback, state)
        np.testing.assert_array_equal(self.dense.track_embeddings, matrix)
        self.assertEqual([t.track_id for t in clean.tracks], ["a"])
        self.assertFalse(clean.query_changed)

    def test_missing_seed_vector_does_not_claim_feedback_or_expand_candidates(self):
        feedback = FeedbackState(votes={"unknown": 1})
        result = self.retriever.retrieve("rock", feedback)
        self.assertEqual(result.unavailable_feedback_ids, ("unknown",))
        self.assertFalse(result.usable_feedback_ids)
        self.assertFalse(result.query_changed)
        self.assertFalse(result.new_candidate_ids)
        self.assertTrue(all(not t.evidence["feedback_applied"] for t in result.tracks))

    def test_bad_embedding_rows_are_ignored_and_lexical_only_tracks_can_still_return(self):
        self.dense.track_embeddings[1] = [float("nan"), 1]
        self.dense.track_embeddings[2] = [0, 0]
        retriever = FeedbackAwareRetriever(self.catalog, LexicalFixture(["b", "a"]), self.dense, candidate_k=2)
        result = retriever.retrieve("rock", self.feedback, topk=2)
        self.assertIn("liked", result.unavailable_feedback_ids)
        self.assertIn("b", [t.track_id for t in result.tracks])
        for track in result.tracks:
            self.assertTrue(np.isfinite(track.score))
            if track.track_id == "b":
                self.assertFalse(track.evidence["feedback_applied"])

    def test_query_space_mismatch_and_malformed_indexes_fail_explicitly(self):
        self.dense.encode_query = lambda _: [1, 0, 0]
        with self.assertRaisesRegex(ValueError, "embedding space"):
            self.retriever.retrieve("rock", self.feedback)
        self.dense.track_ids = ["a"]
        with self.assertRaisesRegex(ValueError, "aligned"):
            FeedbackAwareRetriever(self.catalog, self.lexical, self.dense)
        self.dense.track_ids = ["a"] * 5
        with self.assertRaisesRegex(ValueError, "unique"):
            FeedbackAwareRetriever(self.catalog, self.lexical, self.dense)

    def test_negative_only_feedback_can_change_the_query(self):
        self.dense.encode_query = lambda _: [1, 1]
        result = self.retriever.retrieve("rock", FeedbackState(votes={"liked": -1}))
        self.assertTrue(result.query_changed)
        self.assertEqual(result.usable_feedback_ids, ("liked",))
        self.assertNotIn("liked", result.candidate_ids)

    def test_zero_weight_disables_that_candidate_branch(self):
        lexical_only = FeedbackAwareRetriever(self.catalog, self.lexical, self.dense,
                                               candidate_k=1, dense_weight=0)
        result = lexical_only.retrieve("rock", self.feedback, topk=1)
        self.assertEqual(result.candidate_ids, ("a",))
        self.assertFalse(result.query_changed)
        self.lexical.calls.clear()
        semantic_only = FeedbackAwareRetriever(self.catalog, self.lexical, self.dense,
                                                candidate_k=1, lexical_weight=0)
        result = semantic_only.retrieve("rock", self.feedback, topk=1)
        self.assertEqual(result.candidate_ids, ("b",))
        self.assertEqual(self.lexical.calls, [])

    def test_zero_topk_and_invalid_parameters(self):
        self.dense.encode_query = lambda _: self.fail("Zero topk must not encode a query")
        self.assertEqual(self.retriever.retrieve("rock", self.feedback, topk=0).tracks, [])
        for kwargs in ({"topk": -1}, {"topk": True}, {"mode": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.retriever.retrieve("rock", self.feedback, **kwargs)
        for kwargs in ({"candidate_k": 0}, {"rrf_k": 0}, {"dense_weight": float("nan")},
                       {"lexical_weight": 0, "dense_weight": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FeedbackAwareRetriever(self.catalog, self.lexical, self.dense, **kwargs)


if __name__ == "__main__":
    unittest.main()
