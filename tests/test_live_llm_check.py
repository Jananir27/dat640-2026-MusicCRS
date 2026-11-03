"""Live-check guards: no scripted fallback, no credential leakage or false pass."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from musiccrs.llm_preference_parser import PreferenceExtractionError
from tests.r2_live_smoke import RecordedPreferenceParser, main


class LiveLLMCheckTests(unittest.TestCase):
    def test_missing_credentials_stop_before_models_and_do_not_create_a_success_report(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "result.json"
            with (patch.dict(os.environ, {"OLLAMA_API_KEY": ""}),
                  patch("sys.argv", ["check", "--live-llm", "--output", str(output)]),
                  patch("tests.r2_live_smoke.MusicService", side_effect=AssertionError("Models must not load")),
                  patch("tests.r2_live_smoke.ScriptedPreferenceClient", side_effect=AssertionError("No fake live responses"))):
                with self.assertRaisesRegex(PreferenceExtractionError, "OLLAMA_API_KEY is not set"):
                    main()
            self.assertFalse(output.exists())

    def test_live_mode_performs_a_provider_preflight_without_a_scripted_client(self):
        class StopAfterPreflight(RuntimeError):
            pass
        parsed = {"excluded_artists": ["Linkin Park"], "preferred_artists": ["Green Day"],
                  "use_playlist_context": True}
        delegate = SimpleNamespace(parse=Mock(return_value=parsed))
        with tempfile.TemporaryDirectory() as temp:
            with (patch("sys.argv", ["check", "--live-llm", "--output", str(Path(temp) / "result.json")]),
                  patch("tests.r2_live_smoke.LLMPreferenceParser", return_value=delegate),
                  patch("tests.r2_live_smoke.ScriptedPreferenceClient", side_effect=AssertionError("No fake live responses")),
                  patch("tests.r2_live_smoke.MusicService", side_effect=StopAfterPreflight)):
                with self.assertRaises(StopAfterPreflight):
                    main()
        delegate.parse.assert_called_once()
        self.assertIn("Green Day", delegate.parse.call_args.args[0])

    def test_recording_keeps_preferences_but_never_client_authorization(self):
        delegate = SimpleNamespace(parse=Mock(return_value={"genre": ["rock"]}),
                                   client=SimpleNamespace(headers={"Authorization": "private-test-credential"}))
        recorder = RecordedPreferenceParser(delegate)
        self.assertEqual(recorder.parse("Recommend rock"), {"genre": ["rock"]})
        self.assertEqual(len(recorder.calls), 1)
        self.assertGreaterEqual(recorder.calls[0]["duration_ms"], 0)
        self.assertNotIn("private-test-credential", json.dumps(recorder.calls))
        self.assertNotIn("Authorization", json.dumps(recorder.calls))


if __name__ == "__main__":
    unittest.main()
