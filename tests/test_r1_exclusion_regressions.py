"""R1 review regressions across standard, configured LLM and legacy services."""
from copy import deepcopy
from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch

from musiccrs.llm_recommendations import recommend_with_llm
from musiccrs.music_intelligence import MusicIntelligence
from musiccrs.music_service import ConversationState, MusicService
from musiccrs.request_constraints import ConstraintError
from tests.test_r2_workflow import FixedParser
from tests.test_task6_chat_integration import FakeService as PlaylistOnlyService


def tracks():
    records = [
        ("metallica", "Battery", ["Metallica"]),
        ("safe", "Peace Sells", ["Megadeth"]),
        ("lp", "Numb", ["Linkin Park"]),
        ("lp-version", "Numb - 2003 Remastered", ["Linkin Park"]),
        ("savage", "Numb", ["21 Savage"]),
        ("war", "Why Can't We Be Friends", ["War"]),
        ("drugs", "Red Eyes", ["The War on Drugs"]),
        ("featured", "Collaboration", ["Guest feat. War"]),
        ("credited", "Another Collaboration", ["Guest", "War"]),
        ("warpaint", "Love Is to Die", ["Warpaint"]),
    ]
    return {tid: {"track_id": tid, "track_name": [title], "artist_name": artists,
                  "tag_list": ["rock", "metal"]} for tid, title, artists in records}


class CandidateService:
    """Original recommendation interface, without R2 or a full catalog object."""
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    def get_track(self, tid):
        return self.rows.get(tid)

    def recommend_tracks(self, query, **kwargs):
        self.queries.append(query)
        return list(self.rows.values())[:kwargs["limit"]]


def production_service(rows):
    retriever = SimpleNamespace(text_to_item_retrieval=lambda query, topk: list(rows)[:topk])
    return MusicService(SimpleNamespace(metadata_dict=rows, id_to_metadata=lambda tid: rows[tid]), retriever)


class R1ExclusionRegressionTests(unittest.TestCase):
    def test_sentence_transformers_library_does_not_allow_mixing_sbert_with_cached_qwen_vectors(self):
        from retrieval.dense import DenseRetriever
        with (patch("retrieval.dense.load_dataset") as load,
              patch("retrieval.dense.SentenceTransformer") as model):
            with self.assertRaisesRegex(ValueError, "regenerate matching track embeddings"):
                DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
        load.assert_not_called()
        model.assert_not_called()

    def test_both_exclusion_wordings_work_with_and_without_playlist_in_both_routes(self):
        for advanced in (False, True):
            for playlist in ([], ["lp"]):
                for wording in ("without Metallica", ", no Metallica"):
                    with self.subTest(advanced=advanced, playlist=playlist, wording=wording):
                        service = production_service(tracks())
                        parser = FixedParser(genre=["metal"], excluded_artists=["Metallica"])
                        provider = (lambda request, state: recommend_with_llm(request, parser, service, state)) if advanced else None
                        chat = MusicIntelligence(service, advanced_recommender=provider)
                        state = ConversationState(playlist_ids=playlist[:])
                        chat.handle("Recommend metal " + wording, state)
                        self.assertTrue(state.last_recommendation_ids)
                        self.assertNotIn("metallica", state.last_recommendation_ids)
                        self.assertIn("metallica", state.feedback.excluded_artists)
                        self.assertEqual(len(parser.calls), int(advanced))

    def test_configured_advanced_provider_is_used_by_empty_playlist_minimal_service(self):
        service = PlaylistOnlyService()
        calls = []
        def provider(request, state):
            calls.append(request)
            return {"recommendations": [service.tracks["a"]]}
        state = ConversationState()
        response, _ = MusicIntelligence(service, advanced_recommender=provider).handle("Recommend rock songs", state)
        self.assertEqual(calls, ["Recommend rock songs"])
        self.assertIn("Track A", response)

    def test_actual_agent_calls_configured_parser_with_empty_playlist(self):
        from musiccrs.musiccrs import MusicCRS
        service = production_service(tracks())
        parser = FixedParser(genre=["metal"], excluded_artists=["Metallica"])
        with (patch.dict(os.environ, {"MUSICCRS_ADVANCED_LLM": "true"}),
              patch("musiccrs.musiccrs.get_shared_music_service", return_value=service),
              patch("musiccrs.llm_preference_parser.LLMPreferenceParser", return_value=parser)):
            agent = MusicCRS(use_llm=False)
            state = agent._conversation_state
            agent._music_intelligence.handle("Recommend metal, no Metallica", state)
        self.assertEqual(len(parser.calls), 1)
        self.assertTrue(state.last_recommendation_ids)
        self.assertNotIn("metallica", state.last_recommendation_ids)

    def test_qualified_track_identity_is_independent_of_llm_string_format(self):
        for factory in (production_service, CandidateService):
            for parsed in ("Numb", "Numb by Linkin Park", "Numb by 21 Savage", None):
                with self.subTest(factory=factory.__name__, parsed=parsed):
                    service = factory(tracks())
                    state = ConversationState()
                    parser = FixedParser(genre=[], excluded_tracks=[parsed] if parsed else [])
                    result = recommend_with_llm("Recommend music, exclude Numb by Linkin Park", parser,
                                                service, state, limit=10, candidate_limit=10)
                    ids = {row["track_id"] for row in result["recommendations"]}
                    self.assertIn("savage", ids)
                    self.assertFalse({"lp", "lp-version"} & ids)

    def test_excluded_identity_is_not_used_as_a_positive_retrieval_seed_when_llm_has_no_preferences(self):
        service = CandidateService(tracks())
        recommend_with_llm("Recommend music, exclude Numb by Linkin Park", FixedParser(genre=[]),
                           service, ConversationState(), limit=10, candidate_limit=10)
        self.assertEqual(service.queries, ["Recommend music"])

    def test_qualified_reference_never_bans_the_only_other_artists_same_title_in_partial_pool(self):
        rows = {t: row for t, row in tracks().items() if t not in {"lp", "lp-version"}}
        result = recommend_with_llm("Recommend music without Numb by Linkin Park",
                                    FixedParser(genre=[], excluded_tracks=["Numb"]),
                                    CandidateService(rows), ConversationState(), limit=10, candidate_limit=10)
        self.assertIn("savage", {row["track_id"] for row in result["recommendations"]})

    def test_ambiguous_title_asks_for_artist_instead_of_excluding_all_same_named_songs(self):
        for factory in (production_service, CandidateService):
            with self.subTest(factory=factory.__name__):
                state = ConversationState()
                saved = deepcopy(state.feedback)
                with self.assertRaisesRegex(ConstraintError, "Which artist"):
                    recommend_with_llm("Recommend music, exclude War and Numb",
                                       FixedParser(genre=[], excluded_artists=["War"], excluded_tracks=["Numb"]),
                                       factory(tracks()), state, limit=10, candidate_limit=10)
                self.assertEqual(state.feedback, saved)

    def test_exact_artist_identity_preserves_unrelated_bands_and_blocks_all_credits(self):
        for factory in (production_service, CandidateService):
            with self.subTest(factory=factory.__name__):
                result = recommend_with_llm("Recommend rock without War", FixedParser(excluded_artists=["War"]),
                                           factory(tracks()), ConversationState(), limit=10, candidate_limit=10)
                ids = {r["track_id"] for r in result["recommendations"]}
                self.assertTrue({"drugs", "warpaint"} <= ids)
                self.assertFalse({"war", "featured", "credited"} & ids)

    def test_avoid_playlist_artists_uses_individual_featured_credits(self):
        for factory in (production_service, CandidateService):
            with self.subTest(factory=factory.__name__):
                state = ConversationState(playlist_ids=["featured"])
                result = recommend_with_llm("Recommend different artists from my playlist",
                                           FixedParser(genre=[], avoid_playlist_artists=True),
                                           factory(tracks()), state, limit=10, candidate_limit=10)
                ids = {r["track_id"] for r in result["recommendations"]}
                self.assertIn("drugs", ids)
                self.assertNotIn("war", ids)
                self.assertNotIn("credited", ids)


if __name__ == "__main__":
    unittest.main()
