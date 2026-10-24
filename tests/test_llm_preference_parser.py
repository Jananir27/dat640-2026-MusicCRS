"""Offline unit tests; no UiS token, GPU, or TalkPlay download required."""

import json
import unittest

from musiccrs.llm_preference_parser import (
    LLMPreferenceParser,
    PreferenceExtractionError,
)


class FakeOllamaClient:
    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {"response": self.response_text}


def sample(**overrides):
    payload = dict(
        genre=["rock"],
        mood=["energetic"],
        preferred_artists=["Green Day"],
        excluded_artists=["Linkin Park"],
        excluded_tracks=[],
        use_playlist_context=True,
        avoid_playlist_artists=False,
    )
    payload.update(overrides)
    return payload


class LLMPreferenceParserTests(unittest.TestCase):
    def test_extraction_and_exactly_one_call(self):
        fake = FakeOllamaClient(json.dumps(sample()))
        parser = LLMPreferenceParser(client=fake)
        result = parser.parse("Rock like Green Day, exclude Linkin Park, like my playlist")
        self.assertEqual(result["excluded_artists"], ["Linkin Park"])
        self.assertEqual(result["preferred_artists"], ["Green Day"])
        self.assertTrue(result["use_playlist_context"])
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["format"], "json")
        self.assertFalse(fake.calls[0]["stream"])

    def test_duplicates_removed_without_lowercasing(self):
        fake = FakeOllamaClient(json.dumps(sample(genre=["rock", " ROCK ", "punk"])))
        result = LLMPreferenceParser(client=fake).parse("play rock and punk")
        self.assertEqual(result["genre"], ["rock", "punk"])

    def test_different_artists_boolean(self):
        fake = FakeOllamaClient(json.dumps(sample(avoid_playlist_artists=True)))
        result = LLMPreferenceParser(client=fake).parse("from different artists")
        self.assertTrue(result["avoid_playlist_artists"])

    def test_bad_json_rejected(self):
        parser = LLMPreferenceParser(client=FakeOllamaClient("not json"))
        with self.assertRaises(PreferenceExtractionError):
            parser.parse("recommend rock")

    def test_missing_required_field_rejected(self):
        payload = sample()
        del payload["excluded_tracks"]
        parser = LLMPreferenceParser(client=FakeOllamaClient(json.dumps(payload)))
        with self.assertRaises(PreferenceExtractionError):
            parser.parse("recommend rock")

    def test_invalid_boolean_rejected(self):
        parser = LLMPreferenceParser(
            client=FakeOllamaClient(json.dumps(sample(use_playlist_context="yes")))
        )
        with self.assertRaises(PreferenceExtractionError):
            parser.parse("recommend rock")

    def test_empty_user_request_rejected_without_call(self):
        fake = FakeOllamaClient(json.dumps(sample()))
        with self.assertRaises(ValueError):
            LLMPreferenceParser(client=fake).parse("  ")
        self.assertEqual(fake.calls, [])


if __name__ == "__main__":
    unittest.main()

