"""The frontend receives canonical per-session ratings, including resets."""
import json
import unittest
from unittest.mock import patch

from musiccrs.music_service import MusicService
from musiccrs.musiccrs import MusicCRS, MusicCRSPlatform, MusicChatNamespace
from tests.test_music_intelligence import FakeCatalog, FakeRetriever


class FeedbackStatusTests(unittest.TestCase):
    def test_socket_snapshots_follow_chat_votes_reset_and_rejected_batches(self):
        service = MusicService(FakeCatalog(), FakeRetriever())
        with patch("musiccrs.musiccrs.get_shared_music_service", return_value=service):
            platform = MusicCRSPlatform(MusicCRS)
            platform.socketio.on_namespace(MusicChatNamespace("/", platform))
            client = platform.socketio.test_client(platform.app)
            client.get_received()
            def message(text):
                client.emit("message", {"message": text})
                msg = [e for e in client.get_received() if e["name"] == "message"][-1]["args"]["message"]
                data = {act["intent"]: {a["slot"]: a["value"] for a in act["annotations"]}
                        for act in msg["dialogue_acts"]}
                return data, json.loads(data["FEEDBACK_STATE"]["feedback_state_json"])
            try:
                data, status = message("Recommend pop music")
                batch = json.loads(data["RECOMMENDATIONS"]["recommendations_json"])
                track = batch["tracks"][0]["track_id"]
                self.assertEqual(status["votes"], {})
                self.assertIn(batch["batch_id"], status["valid_batch_ids"])
                _, status = message("I like track 1")
                self.assertEqual(status["votes"], {track: 1})
                _, status = message("I don't like track 1")
                self.assertEqual(status["votes"], {track: -1})
                _, status = message("/feedback " + json.dumps({"track_id": track, "batch_id": "forged", "event": "thumbs_up"}))
                self.assertEqual(status["votes"], {track: -1})
                _, status = message("Reset my feedback")
                self.assertEqual(status["votes"], {})
                for payload in ({"batch_id": [], "track_id": track, "event": "thumbs_up"},
                                {"batch_id": "x", "track_id": track, "event": "other"}):
                    self.assertFalse(client.emit("feedback", payload, callback=True)["ok"])
            finally:
                client.disconnect()


if __name__ == "__main__":
    unittest.main()
