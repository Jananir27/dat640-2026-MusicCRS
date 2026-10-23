"""Regression checks for DialogueKit's platform startup contract."""

import os
import unittest
from unittest.mock import patch

from musiccrs.musiccrs import MusicCRS, MusicCRSPlatform


class MusicCRSPlatformTests(unittest.TestCase):
    def test_platform_allows_werkzeug_for_the_development_server(self):
        platform = MusicCRSPlatform(MusicCRS)
        with (
            patch.object(platform.socketio, "on_namespace") as on_namespace,
            patch.object(platform.socketio, "run") as run,
        ):
            platform.start(host="0.0.0.0", port="5000")

        on_namespace.assert_called_once()
        run.assert_called_once_with(
            platform.app,
            host="0.0.0.0",
            port="5000",
            allow_unsafe_werkzeug=True,
        )

    @patch("musiccrs.musiccrs.MusicIntelligence")
    @patch("musiccrs.musiccrs.get_shared_music_service")
    def test_dialoguekit_positional_agent_id_does_not_enable_llm(
        self, get_service, _music_intelligence
    ):
        service = object()
        get_service.return_value = service
        with patch.dict(os.environ, {"MUSICCRS_USE_LLM": "false"}):
            agent = MusicCRS("MusicCRS")

        self.assertEqual(agent.id, "MusicCRS")
        self.assertIsNone(agent._llm)
        self.assertIs(agent._music_service, service)


if __name__ == "__main__":
    unittest.main()
