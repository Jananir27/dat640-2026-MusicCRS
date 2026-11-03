"""Verify the benchmark uses the live workflow and never fills missing labels."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from musiccrs.music_intelligence import MusicIntelligence
from musiccrs.music_service import ConversationState, MusicService
from retrieval.evaluation.integrated import METHODS, load_queries, hard_feedback, run_integrated, verify_integrated
from retrieval.evaluation.part4 import prepare_query, score_benchmark
from tests.test_music_intelligence import FakeCatalog, FakeRetriever
from tests.test_relevance_feedback import ToyEmbeddings


class IntegratedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.catalog = FakeCatalog()
        self.services = {name: MusicService(self.catalog, FakeRetriever(), embedding_provider=ToyEmbeddings())
                         for name in METHODS}
        self.queries = [
            {"id": "plain", "category": "genre", "query": "Recommend pop music"},
            {"id": "feedback", "category": "mixed_feedback", "query": "Recommend pop music",
             "likes": [{"title": "Love", "artist": "Artist One"}],
             "dislikes": [{"title": "Pop Song", "artist": "Artist Four"}],
             "excluded_artists": ["Metallica"]},
        ]

    def test_rankings_match_direct_conversation_handling_and_share_hard_constraints(self):
        with tempfile.TemporaryDirectory() as temp:
            document, coverage = run_integrated(self.queries, self.catalog, self.services,
                                                 Path(temp) / "result", repeats=2)
            self.assertEqual(coverage["run_count"], 8)
            self.assertEqual(coverage["timing_samples"], 16)
            self.assertEqual(coverage["excluded_tracks_returned"], 0)
            replay = verify_integrated(Path(temp) / "result", self.catalog, self.services)
            self.assertTrue(replay["matched"])
            self.assertEqual(replay["maximum_score_difference"], 0)
            self.assertTrue(all('/venv/' not in p and '/.venv/' not in p and '/site-packages/' not in p
                                for p in replay['current_source_sha256']))
            for run in document["runs"]:
                query = next(q for q in self.queries if q["id"] == run["query_id"])
                prepared = prepare_query(query, self.catalog)
                feedback = hard_feedback(prepared, self.catalog)
                if run["method"] != "hybrid_feedback":
                    feedback.votes.clear()
                state = ConversationState(feedback=feedback)
                MusicIntelligence(self.services[run["method"]]).handle(run["application_request"], state)
                self.assertEqual(run["predicted_track_ids"], state.last_recommendation_ids)
                if query["id"] == "feedback":
                    self.assertFalse({"1", "2", "4"} & set(run["predicted_track_ids"]))
            with (Path(temp) / "result/human_review_blank.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), coverage["unjudged_pairs"])
            self.assertTrue(all(not row[c] for row in rows for c in ("grade", "reason", "assessor")))
            self.assertNotIn("method", rows[0])
            with self.assertRaisesRegex(ValueError, "Complete all"):
                score_benchmark(Path(temp) / "result/runs.json", Path(temp) / "result/judgments.csv")

    def test_existing_ai_labels_are_reused_with_their_provenance_and_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_integrated(self.queries, self.catalog, self.services, root / "first", repeats=1)
            with (root / "first/judgments.csv").open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            for row in rows:
                row.update(grade="2", reason="synthetic fixture grade", assessor="AI-assisted:metadata-only:PROVISIONAL")
            with (root / "source.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            _, coverage = run_integrated(self.queries, self.catalog, self.services, root / "second", repeats=1,
                                          judgment_sources=[root / "source.csv"])
            self.assertEqual(coverage["unjudged_pairs"], 0)
            result = score_benchmark(root / "second/runs.json", root / "second/judgments.csv")
            self.assertTrue(result["judgment_status"].startswith("AI-REVIEWED"))
            with self.assertRaises(FileExistsError):
                run_integrated(self.queries, self.catalog, self.services, root / "second", repeats=1)

    def test_duplicate_queries_and_changed_metadata_are_rejected_or_left_unjudged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "queries.json"
            path.write_text(json.dumps(self.queries + self.queries))
            with self.assertRaisesRegex(ValueError, "unique"):
                load_queries([path])
            run_integrated(self.queries, self.catalog, self.services, root / "first", repeats=1)
            with (root / "first/judgments.csv").open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            for row in rows:
                row.update(grade="2", assessor="fixture", title="different recording")
            with (root / "source.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            _, coverage = run_integrated(self.queries, self.catalog, self.services, root / "second", repeats=1,
                                          judgment_sources=[root / "source.csv"])
            self.assertEqual(coverage["reused_judgments"], 0)
            self.assertEqual(coverage["metadata_mismatches_not_reused"], len(rows))


if __name__ == "__main__":
    unittest.main()
