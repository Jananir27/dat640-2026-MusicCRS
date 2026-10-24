import csv
import json
from pathlib import Path
import tempfile
import unittest

from retrieval.evaluation.metrics import graded_retrieval_metrics, latency_summary
from retrieval.evaluation.part4 import run_benchmark, draft_judgments, score_benchmark
from tests.test_music_intelligence import FakeCatalog, FakeRetriever
from tests.test_relevance_feedback import ToyEmbeddings


class EmbeddedFixture(FakeRetriever, ToyEmbeddings):
    pass


class Part4EvaluationTests(unittest.TestCase):
    def test_graded_ndcg_uses_exponential_gains_and_full_ideal_ranking(self):
        ideal = graded_retrieval_metrics(["a", "b"], {"a": 2, "b": 1, "c": 0})
        swapped = graded_retrieval_metrics(["b", "a"], {"a": 2, "b": 1, "c": 0})
        self.assertEqual(ideal["ndcg@5"], 1.0)
        self.assertLess(swapped["ndcg@5"], 1.0)
        self.assertAlmostEqual(ideal["precision@5"], 0.4)
        self.assertEqual(graded_retrieval_metrics([], {"a": 2})["ndcg@5"], 0)
        self.assertEqual(graded_retrieval_metrics(["a"], {"a": 0})["ndcg@5"], 0)

    def test_invalid_or_unjudged_inputs_are_not_silently_scored(self):
        for predictions, judgments in [(["a", "a"], {"a": 2}), (["missing"], {"a": 2}), (["a"], {"a": 3})]:
            with self.subTest(predictions=predictions, judgments=judgments), self.assertRaises(ValueError):
                graded_retrieval_metrics(predictions, judgments)
        for values in ([], [-1], [float("nan")]):
            with self.assertRaises(ValueError):
                latency_summary(values)
        self.assertEqual(latency_summary([10, 20, 30])["p50_ms"], 20)

    def test_query_contribution_has_five_examples_in_each_category(self):
        queries = json.loads(Path("evaluation/part4/tobi_queries.json").read_text())
        self.assertEqual(len(queries), 25)
        self.assertEqual(len({q["id"] for q in queries}), 25)
        for category in ("genre", "artist_similarity", "playlist_context", "exclusion", "mixed_feedback"):
            self.assertEqual(sum(q["category"] == category for q in queries), 5)

    def test_ai_review_and_mixed_labels_are_not_reported_as_human_assessment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = {"queries": [{"id": "q"}], "runs": [
                {"query_id": "q", "category": "genre", "method": "bm25",
                 "predicted_track_ids": ["a", "b"], "latency_ms": [1], "final_excluded_count": 0}],
                "latency_scope": "fixture"}
            (root / "runs.json").write_text(json.dumps(document))
            cases = [
                (["codex-ai:metadata-reviewed:REVIEW-REQUIRED"] * 2, "AI-REVIEWED METADATA"),
                (["codex-ai:metadata-reviewed:REVIEW-REQUIRED", "assessor-1"], "MIXED JUDGMENTS"),
                (["metadata-proxy:REVIEW-REQUIRED"] * 2, "PROVISIONAL METADATA PROXY"),
                ([""] * 2, "PROVISIONAL METADATA PROXY"),
                (["assessor-1"] * 2, "assessor-labeled"),
            ]
            for assessors, expected in cases:
                with self.subTest(assessors=assessors):
                    with (root / "judgments.csv").open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=["query_id", "track_id", "grade", "assessor"])
                        writer.writeheader()
                        for track_id, assessor in zip(("a", "b"), assessors):
                            writer.writerow({"query_id": "q", "track_id": track_id, "grade": 2, "assessor": assessor})
                    result = score_benchmark(root / "runs.json", root / "judgments.csv")
                    self.assertTrue(result["judgment_status"].startswith(expected))
                    self.assertEqual(sum(result["judgment_provenance"].values()), 2)
                    self.assertEqual(result["methods"]["bm25"]["ndcg@5"], 1)

    def test_pooling_scoring_and_feedback_comparison_are_reproducible(self):
        queries = [{"id": "fixture", "category": "mixed_feedback", "query": "Recommend pop music",
                    "genres": ["pop"], "likes": [{"title": "Lover", "artist": "Artist Three"}],
                    "excluded_artists": ["Metallica"]}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense = EmbeddedFixture()
            document = run_benchmark(queries, FakeCatalog(), dense, dense, dense, root, pool_depth=5)
            self.assertEqual(len(document["runs"]), 4)
            self.assertTrue(all(run["final_excluded_count"] == 0 for run in document["runs"]))
            self.assertTrue(all("2" not in run["predicted_track_ids"] for run in document["runs"]))
            with self.assertRaises(ValueError):
                score_benchmark(root / "runs.json", root / "judgments.csv")
            draft_judgments(root / "runs.json", root / "judgments.csv", root / "draft.csv")
            result = score_benchmark(root / "runs.json", root / "draft.csv")
            self.assertIn("PROVISIONAL", result["judgment_status"])
            self.assertEqual(result["feedback_query_count"], 1)
            self.assertEqual(set(result["methods"]), {"bm25", "semantic", "hybrid", "hybrid_feedback"})
            with (root / "draft.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(all(row["reason"] for row in rows))


if __name__ == "__main__":
    unittest.main()
