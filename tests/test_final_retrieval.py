"""Offline verification of the final pipeline and artifact replay guard."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from musiccrs.music_service import ConversationState, MusicService
from retrieval.export_final import export_artifact
from retrieval.final import FinalRetriever, load_artifact
from retrieval.tune_selective_expansion import selective_rerank
from tests.retrieval_fixtures import FixedDenseRetriever, fixture_artifact
from tests.test_music_intelligence import FakeCatalog


class FinalRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.catalog = FakeCatalog()
        seed = self.catalog.metadata_dict["1"]
        self.catalog.metadata_dict = {
            str(i): {**seed, "track_id": str(i), "track_name": [f"Track {i}"]}
            for i in range(1, 121)
        }
        self.lexical = FixedDenseRetriever(self.catalog.metadata_dict)
        self.dense = FixedDenseRetriever(reversed(self.catalog.metadata_dict))
        self.artifact = fixture_artifact()
        self.retriever = FinalRetriever(
            self.catalog, self.artifact, bm25_retriever=self.lexical, dense_retriever=self.dense,
        )

    def test_final_ranking_protects_prefix_and_expands_tail(self):
        before = deepcopy(self.artifact)
        result = self.retriever.text_to_item_retrieval("relaxing music", 40)
        self.assertEqual(result[:12], [str(i) for i in range(1, 13)])
        self.assertEqual(len(result), 20)
        self.assertEqual(len(set(result)), 20)
        self.assertTrue(any(int(track) > 20 for track in result[12:]))
        self.assertEqual([k for _, k in self.lexical.calls], [50, 50, 100])
        self.assertEqual([k for _, k in self.dense.calls], [50, 50, 100])
        self.assertEqual(self.artifact, before, "Requests must not mutate frozen novelty statistics")
        self.assertEqual(self.retriever.text_to_item_retrieval("relaxing music", 5), result[:5])

    def test_service_passes_request_context_and_filters_after_reranking(self):
        service = MusicService(self.catalog, self.retriever)
        state = ConversationState(playlist_ids=["1", "2"], recent_retrieval_requests=["I like jazz"])
        with patch.object(self.retriever, "retrieve_context", wraps=self.retriever.retrieve_context) as call:
            tracks = service.recommend_from_playlist(state, request="Suggest calm songs")
        self.assertEqual(len(tracks), 5)
        self.assertFalse({"1", "2"} & {track["track_id"] for track in tracks})
        item = call.call_args.args[0]
        self.assertEqual(item["conversation_goal"]["specificity"], "LL")
        self.assertIn("I like jazz", [turn["content"] for turn in item["conversations"]])
        self.assertEqual(item["conversations"][-1]["content"], "Suggest calm songs")
        self.assertEqual(state.recent_retrieval_requests, ["I like jazz", "Suggest calm songs"])
        with patch.object(self.retriever, "retrieve_context", wraps=self.retriever.retrieve_context) as call:
            service.recommend_tracks("gym", state=state, new_playlist=True, limit=20)
        self.assertEqual(len(call.call_args.args[0]["conversations"]), 1)
        self.assertEqual(state.recent_retrieval_requests, ["gym"])
        service.clear_playlist(state)
        self.assertEqual(state.recent_retrieval_requests, [])

    def test_playlist_recommendation_builds_genre_mood_query_and_keeps_final_retriever(self):
        service = MusicService(self.catalog, self.retriever)
        state = ConversationState(playlist_ids=["1"])
        with patch.object(self.retriever, "retrieve_context", return_value=[str(i) for i in range(1, 21)]) as call:
            result = service.recommend_from_playlist(
                state, limit=5, request="Recommend something mellow based on my playlist",
            )
        self.assertEqual(len(result), 5)
        goal = call.call_args.args[0]["conversation_goal"]["listener_goal"]
        self.assertIn("preferred genres: pop", goal)
        self.assertIn("preferred moods: chill", goal)
        self.assertNotIn("love", goal)
        self.assertNotIn("favorites", goal)
        self.assertNotIn("Linkin Park", goal)
        self.assertNotIn("music", [turn["role"] for turn in call.call_args.args[0]["conversations"]])

    def test_new_playlist_uses_genre_and_mood_preferences(self):
        service = MusicService(self.catalog, self.retriever)
        with patch.object(self.retriever, "retrieve_context", return_value=[str(i) for i in range(1, 21)]) as call:
            tracks = service.recommend_for_playlist_description(
                "hip-hop and R&B mellow", limit=5, request="Make a hip-hop and R&B mellow playlist",
            )
        self.assertEqual(len(tracks), 5)
        goal = call.call_args.args[0]["conversation_goal"]["listener_goal"]
        self.assertIn("hip-hop and R&B mellow", goal)

    def test_default_service_creates_final_retriever(self):
        service = MusicService(self.catalog)
        with patch("musiccrs.music_service.FinalRetriever", return_value=self.retriever) as factory:
            self.assertIs(service._get_retriever(), self.retriever)
            self.assertIs(service._get_retriever(), self.retriever)
        factory.assert_called_once_with(catalog=self.catalog)

    def test_missing_artifact_fails_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "retrieval.export_final"):
                load_artifact(Path(directory) / "missing.json")
            with patch.dict("os.environ", {"MUSICCRS_RETRIEVAL_ARTIFACT": str(Path(directory) / "missing.json")}):
                with patch("retrieval.hybrid.HybridRetriever") as hybrid:
                    with self.assertRaises(RuntimeError):
                        MusicService(self.catalog)._get_retriever()
                    hybrid.assert_not_called()

    def test_bad_configuration_is_rejected_before_model_load(self):
        for change in ({"selective": {}}, {"hybrid": {**self.artifact["hybrid"], "dense_weight": float("nan")}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                FinalRetriever(self.catalog, {**self.artifact, **change})

    def test_export_replays_saved_outputs_and_rejects_mismatched_files(self):
        base = [str(i) for i in range(1, 21)]
        reservoir = [str(i) for i in range(1, 101)]
        final = selective_rerank(
            base, reservoir, self.artifact["selective_statistics"]["frequency"], 100,
            **self.artifact["selective"],
        )

        def rows(tracks):
            return [{"session_id": "fixture", "turn_number": i, "predicted_track_ids": tracks}
                    for i in range(100)]

        files = {
            "results_prediction_fusion.json": {"best_weights": self.artifact["fusion_weights"]},
            "results_exact_specificity_tuning.json": {"best_weights": self.artifact["exact_weights"]},
            "results_diversity_tuning.json": {"best_configuration": self.artifact["diversity"]},
            "results_selective_expansion_fine.json": {"best_configuration": self.artifact["selective"], "best_metrics": {}},
            "predictions_exact_specificity_best.json": rows(base),
            "predictions_diversity_best.json": rows(base),
            "predictions_adaptive_goal_top100_full.json": rows(reservoir),
            "predictions_selective_expansion_fine_best.json": rows(final),
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for name, data in files.items():
                (directory / name).write_text(json.dumps(data))
            output = directory / "runtime.json"
            exported = export_artifact(directory, output)
            self.assertEqual(exported["verified_rerank_turns"], 100)
            self.assertEqual(load_artifact(output)["selective_statistics"], self.artifact["selective_statistics"])
            self.assertEqual(len(exported["source_sha256"]), 8)
            saved = output.read_bytes()
            (directory / "predictions_selective_expansion_fine_best.json").write_text(json.dumps(rows(base)))
            with self.assertRaisesRegex(ValueError, "does not reproduce"):
                export_artifact(directory, output)
            self.assertEqual(output.read_bytes(), saved, "Failed verification must not overwrite the artifact")


if __name__ == "__main__":
    unittest.main()
