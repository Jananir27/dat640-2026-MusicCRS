"""Run integrated R1–R8 backend checks, with a readable conversation transcript.

    python -m tests.full_music_smoke           # real catalog, Qwen, final G1 pipeline
    python -m tests.full_music_smoke --offline # fixture, BM25, fake dense, real rerankers

Exercises the real agent and its playlist events using an in-process connector.
Does not start a web server, render the UI, or call an LLM. See retrieval/README.md#musiccrs-verification.
"""

import argparse
from collections import defaultdict
import json
import math
import tempfile
from time import perf_counter
import unittest
from unittest.mock import Mock, patch

from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.participant import DialogueParticipant

from musiccrs.musiccrs import MusicCRS
from musiccrs.music_service import MusicService
from retrieval.bm25 import BM25Retriever
from retrieval.final import FinalRetriever, load_artifact
from tests.retrieval_fixtures import FixedDenseRetriever, fixture_artifact
from tests.test_music_intelligence import FakeCatalog


class FullMusicSmoke(unittest.TestCase):
    offline = False
    verbose_chat = True
    timings = []

    @classmethod
    def setUpClass(cls):
        started = perf_counter()
        if cls.offline:
            cache = tempfile.TemporaryDirectory(prefix="musiccrs-smoke-")
            cls.addClassCleanup(cache.cleanup)
            catalog = FakeCatalog()
            lexical = BM25Retriever(
                catalog=catalog, cache_dir=cache.name,
                corpus_types=["track_name", "artist_name", "album_name", "tag_list"],
            )
            retriever = FinalRetriever(
                catalog, fixture_artifact(), bm25_retriever=lexical,
                dense_retriever=FixedDenseRetriever(catalog.metadata_dict),
            )
            cls.service = MusicService(catalog=catalog, retriever=retriever)
            print("OFFLINE: synthetic settings and dense rankings; real G1 ranking functions.")
        else:
            load_artifact()  # Fail before downloading the catalog/model if unconfigured.
            cls.service = MusicService()
            cls.service._get_retriever()
            print("LIVE: final G1 pipeline with saved settings and Qwen embeddings.")
        if not isinstance(cls.service._get_retriever(), FinalRetriever):
            raise AssertionError("Full smoke test must exercise FinalRetriever")
        rows = list(cls.service.catalog.metadata_dict.values())
        if not rows:
            raise AssertionError("Catalog is empty")
        cls.seed = next(row for row in rows if all(
            row.get(key) for key in (
                "track_name", "artist_name", "album_name", "release_date", "tag_list"
            )
        ))
        cls.title = cls.first_value(cls.seed["track_name"])
        cls.artist = cls.first_value(cls.seed["artist_name"])
        cls.entity = f"{cls.title} by {cls.artist}"
        grouped = defaultdict(list)
        for row in rows:
            names = row["track_name"]
            for title in names if isinstance(names, list) else [names]:
                grouped[title.casefold()].append(row)
        # Use the requirement's example when available; otherwise another exact duplicate.
        cls.ambiguous_title = next(
            title for title in ["love", *grouped]
            if len({row["track_id"] for row in grouped[title]}) >= 2
        )
        print(f"\nLoaded {len(rows):,} tracks; catalog/index setup: "
              f"{perf_counter() - started:.2f}s (separate from turn timings).", flush=True)
        print(f"Seed: {cls.entity}", flush=True)

    @staticmethod
    def first_value(value):
        return value[0] if isinstance(value, list) else value

    def new_agent(self):
        with patch("musiccrs.musiccrs.get_shared_music_service", return_value=self.service):
            agent = MusicCRS(use_llm=False)
        agent.connect_dialogue_connector(Mock())
        return agent

    def setUp(self):
        self.agent = self.new_agent()
        self.state = self.agent._conversation_state
        if self.verbose_chat:
            print(f"\n--- {self._testMethodName} ---", flush=True)

    def send(self, message, updated=False):
        before = list(self.state.playlist_ids)
        connector = self.agent._dialogue_connector
        connector.reset_mock()
        started = perf_counter()
        self.agent.receive_utterance(Utterance(message, participant=DialogueParticipant.USER))
        elapsed = perf_counter() - started
        self.timings.append(elapsed)
        connector.register_agent_utterance.assert_called_once()
        reply = connector.register_agent_utterance.call_args.args[0]
        if self.verbose_chat:
            print(f"YOU: {message}\nBOT: {reply.text}\n[{elapsed:.3f}s]", flush=True)
        self.assertTrue(reply.text.strip(), "Agent returned an empty response")
        self.assertNotIn("I'm sorry, I don't understand", reply.text)
        events = [act for act in reply.dialogue_acts if str(act.intent) == "PLAYLIST_UPDATED"]
        self.assertEqual(bool(events), updated, "Wrong playlist update signal")
        if updated:
            self.assertEqual(len(events), 1)
            payloads = [a.value for a in events[0].annotations if a.slot == "playlist_json"]
            self.assertEqual(len(payloads), 1)
            records = json.loads(payloads[0])
            self.assertEqual(records, self.service.get_playlist_records(self.state))
            for record in records:
                self.assertTrue({"track_id", "track_name", "artist_name"} <= record.keys())
        else:
            self.assertEqual(self.state.playlist_ids, before, "Silent playlist mutation")
        self.assertEqual(len(self.state.playlist_ids), len(set(self.state.playlist_ids)))
        self.assertTrue(all(self.service.get_track(t) for t in self.state.playlist_ids))
        return reply.text

    def seed_playlist(self):
        self.send(f"/playlist add {self.seed['track_id']}", updated=True)

    def recommend(self):
        if not self.state.playlist_ids:
            self.seed_playlist()
        existing = set(self.state.playlist_ids)
        self.send("Recommend something based on my playlist")
        pending = list(self.state.pending_recommendation_ids)
        self.assertTrue(3 <= len(pending) <= 5, "R6 must return 3–5 suggestions")
        self.assertEqual(len(pending), len(set(pending)))
        self.assertFalse(existing.intersection(pending))
        self.assertTrue(all(self.service.get_track(t) for t in pending))
        return pending

    def test_r1_panel_commands_and_payload(self):
        self.seed_playlist()
        self.send("/playlist show", updated=True)
        self.send(f"/playlist remove {self.seed['track_id']}", updated=True)
        self.assertEqual(self.state.playlist_ids, [])
        self.seed_playlist()
        self.send("/playlist clear", updated=True)
        self.assertEqual(self.state.playlist_ids, [])

    def test_r2_add_show_remove_clear(self):
        self.send(f"Add {self.entity}", updated=True)
        self.assertEqual(self.state.playlist_ids, [self.seed["track_id"]])
        self.assertIn(self.title, self.send("Show my playlist"))
        self.send(f"Remove {self.entity}", updated=True)
        self.assertEqual(self.state.playlist_ids, [])
        self.send(f"Add {self.entity}", updated=True)
        self.send("Clear my playlist", updated=True)
        self.assertEqual(self.state.playlist_ids, [])

    def test_r2_unknown_and_duplicate_tracks(self):
        self.seed_playlist()
        self.send(f"Add {self.entity}", updated=True)
        self.assertEqual(self.state.playlist_ids, [self.seed["track_id"]])
        self.send("Add xyzabc123notasong")
        self.send("/playlist add __unknown_track__")
        self.send("/playlist remove __unknown_track__")
        self.assertEqual(self.state.playlist_ids, [self.seed["track_id"]])

    def test_r3_help_formulations(self):
        for prompt in ("Help me", "What can you do?", "How can I use this?"):
            with self.subTest(prompt=prompt):
                answer = self.send(prompt).casefold()
                for capability in ("add", "remove", "clear", "recommend", "create"):
                    self.assertIn(capability, answer)

    def test_r4_disambiguation_and_popularity(self):
        for selection in ("2", "the second one"):
            with self.subTest(selection=selection):
                self.send("Clear my playlist", updated=True)
                self.send(f"Add {self.ambiguous_title}")
                pending = list(self.state.pending_disambiguation_ids)
                self.assertGreaterEqual(len(pending), 2)
                popularities = [float(self.service.get_track(t).get("popularity") or 0) for t in pending]
                self.assertEqual(popularities, sorted(popularities, reverse=True))
                self.send(selection, updated=True)
                self.assertEqual(self.state.playlist_ids, [pending[1]])
                self.assertFalse(self.state.pending_disambiguation_ids)

    def test_r5_four_metadata_questions(self):
        for prompt, field in (
            (f"Who performs {self.entity}?", "artist_name"),
            (f"What album is {self.entity} on?", "album_name"),
            (f"When was {self.entity} released?", "release_date"),
            (f"What genre is {self.entity}?", "tag_list"),
        ):
            with self.subTest(field=field):
                answer = self.send(prompt)
                if self.state.pending_disambiguation_ids:
                    self.assertIn(self.seed["track_id"], self.state.pending_disambiguation_ids)
                    position = self.state.pending_disambiguation_ids.index(self.seed["track_id"]) + 1
                    answer = self.send(str(position))
                self.assertIn(self.first_value(self.seed[field]), answer)

    def test_r5_artist_catalog_question(self):
        answer = self.send(f"What songs by {self.artist}?")
        self.assertIn(self.artist, answer)
        self.assertIn("Here are tracks", answer)

    def test_r6_empty_playlist(self):
        self.assertIn("Add at least one", self.send("Recommend something based on my playlist"))
        self.assertEqual(self.state.pending_recommendation_ids, [])

    def test_r6_related_recommendations(self):
        pending = self.recommend()
        seed_tags = set(self.service.display_value(self.seed["tag_list"]).casefold().split(", "))
        seed_artists = set(self.service.display_value(self.seed["artist_name"]).casefold().split(", "))
        related = []
        for track_id in pending:
            track = self.service.get_track(track_id)
            tags = set(self.service.display_value(track["tag_list"]).casefold().split(", "))
            artists = set(self.service.display_value(track["artist_name"]).casefold().split(", "))
            related.append(bool(seed_tags & tags or seed_artists & artists))
        self.assertTrue(any(related), "No recommendation shares tags or an artist with the seed")

    def test_r7_add_all_formulations(self):
        for phrase in ("Add all of them", "Add them all", "Add everything"):
            with self.subTest(phrase=phrase):
                self.send("Clear my playlist", updated=True)
                pending = self.recommend()
                before = list(self.state.playlist_ids)
                self.send(phrase, updated=True)
                self.assertEqual(self.state.playlist_ids, before + pending)
                self.assertFalse(self.state.pending_recommendation_ids)

    def test_r7_first_two(self):
        pending = self.recommend()
        before = list(self.state.playlist_ids)
        self.send("Add the first two", updated=True)
        self.assertEqual(self.state.playlist_ids, before + pending[:2])

    def test_r7_multiple_numbers(self):
        pending = self.recommend()
        before = list(self.state.playlist_ids)
        self.send("Add 1 and 3", updated=True)
        self.assertEqual(self.state.playlist_ids, before + [pending[0], pending[2]])

    def test_r7_artist_exclusion(self):
        pending = self.recommend()
        # Choose an artist that identifies one item in the displayed list.
        for track_id in pending:
            artist = self.first_value(self.service.get_track(track_id)["artist_name"])
            matched = [t for t in pending if artist.casefold() in
                       self.service.display_value(self.service.get_track(t)["artist_name"]).casefold()]
            if len(matched) == 1:
                break
        else:
            self.fail("No unambiguous artist in this recommendation list")
        before = list(self.state.playlist_ids)
        self.send(f"Add them except the one by {artist}", updated=True)
        self.assertEqual(self.state.playlist_ids, before + [t for t in pending if t != track_id])

    def test_r7_reject_suggestions(self):
        for phrase in ("No, don't add any", "None", "Skip them"):
            with self.subTest(phrase=phrase):
                self.recommend()
                self.send(phrase)
                self.assertFalse(self.state.pending_recommendation_ids)

    def test_r8_create_from_descriptions(self):
        for description in ("sad love songs", "energetic music for gym session", "relaxing for studying", "a short commute", "a long road trip"):
            with self.subTest(description=description):
                answer = self.send(f"Create a playlist for {description}", updated=True)
                self.assertIn("created a playlist", answer)
                minimum = 4 if "short" in description else 8
                self.assertTrue(minimum <= len(self.state.playlist_ids) <= 20)
                self.assertFalse(self.state.pending_recommendation_ids)
                self.assertFalse(self.state.pending_disambiguation_ids)

    def test_r8_missing_description_preserves_playlist(self):
        self.seed_playlist()
        self.assertIn("mood, genre, or activity", self.send("Create a playlist"))

    def test_session_isolation(self):
        other = self.new_agent()
        self.recommend()
        self.assertEqual(other._conversation_state.playlist_ids, [])
        self.assertEqual(other._conversation_state.pending_recommendation_ids, [])

    def test_panel_clear_discards_pending_choices(self):
        self.recommend()
        self.send("/playlist clear", updated=True)
        self.assertEqual(self.state.pending_recommendation_ids, [])
        self.send(f"Add {self.ambiguous_title}")
        self.send("/playlist clear", updated=True)
        self.assertEqual(self.state.pending_disambiguation_ids, [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use fixture tracks, fake dense rankings, and real G1 reranking")
    parser.add_argument("--quiet", action="store_true", help="Hide conversation transcripts")
    args = parser.parse_args()
    FullMusicSmoke.offline = args.offline
    FullMusicSmoke.verbose_chat = not args.quiet
    FullMusicSmoke.timings = []
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FullMusicSmoke)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    timings = sorted(FullMusicSmoke.timings)
    if timings:
        print(f"\n{len(timings)} turns; median {timings[len(timings)//2]:.3f}s; "
              f"p95 {timings[max(0, math.ceil(len(timings)*0.95)-1)]:.3f}s; "
              f"max {timings[-1]:.3f}s (local backend, LLM disabled).")
    print("Browser rendering/socket transport and recommendation quality need the manual checks in retrieval/README.md#musiccrs-verification.")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
