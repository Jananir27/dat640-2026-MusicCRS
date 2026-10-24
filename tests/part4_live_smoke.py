"""Real catalog/G1/Qwen integration through Socket.IO, isolated from saved user data.

    python -m tests.part4_live_smoke --output evaluation/part4/results/live_feedback_demo.json

Requires the normal cached catalog, embeddings, model and final retrieval artifact.
"""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from musiccrs.music_service import MusicService
from musiccrs.musiccrs import MusicCRS, MusicCRSPlatform, MusicChatNamespace
from musiccrs.relevance_feedback import name_key, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="evaluation/part4/results/live_feedback_demo.json")
    args = parser.parse_args()
    service = MusicService()
    service._get_retriever()
    turns = []
    with patch("musiccrs.musiccrs.get_shared_music_service", return_value=service), patch.dict(os.environ, {"MUSICCRS_USE_LLM": "false"}):
        platform = MusicCRSPlatform(MusicCRS)
        platform.socketio.on_namespace(MusicChatNamespace("/", platform))
        client = platform.socketio.test_client(platform.app)  # no device identity or saved-playlist mutations
        client.get_received()

        def send(text):
            started = perf_counter()
            client.emit("message", {"message": text})
            message = [e for e in client.get_received() if e["name"] == "message"][-1]["args"]["message"]
            elapsed = (perf_counter() - started) * 1000
            payload = next((json.loads(a["value"]) for act in message.get("dialogue_acts", [])
                            if act["intent"] == "RECOMMENDATIONS" for a in act["annotations"]
                            if a["slot"] == "recommendations_json"), None)
            turns.append({"query": text, "response": message["text"], "backend_response_ms": elapsed,
                          "recommendations": payload})
            print(text, "->", message["text"], flush=True)
            return payload

        try:
            before = send("Recommend some rock music")
            assert before and len(before["tracks"]) == 5
            liked, disliked = before["tracks"][1]["track_id"], before["tracks"][3]["track_id"]
            send("I like track 2, but I don't like track 4")
            after = send("Recommend some rock music")
            assert after and disliked not in [t["track_id"] for t in after["tracks"]]
            assert any(abs(t["scores"].get("feedback_delta", 0)) > 1e-6 for t in after["tracks"])
            blocked_artist = values(service.get_track(after["tracks"][0]["track_id"])["artist_name"])[0]
            send(f"I don't like the artist {blocked_artist}")
            filtered = send("Recommend some rock music")
            assert filtered and all(name_key(blocked_artist) not in {name_key(a) for a in
                                   values(service.get_track(t["track_id"]).get("artist_name"))}
                                   for t in filtered["tracks"])
            client.emit("feedback", {"batch_id": before["batch_id"], "track_id": liked, "event": "thumbs_down"})
            assert "disliked" in client.get_received()[-1]["args"]["message"]["text"]
            send("Reset my feedback")
            assert "cleared" in turns[-1]["response"]
            send("Add Numb by Linkin Park")
            playlist = send("Recommend something based on my playlist")
            assert playlist and 1 <= len(playlist["tracks"]) <= 5
            report = {
                "fixture": False, "catalog_size": len(service.catalog.metadata_dict),
                "retriever": "FinalRetriever (G1)", "embedding_model": "Qwen3-Embedding-0.6B", "turns": turns,
                "checks": {"negative_track_removed": True, "nonzero_vector_feedback": True,
                           "blocked_artist_removed": True, "older_batch_rating_accepted": True,
                           "reset_accepted": True, "playlist_recommendations_preserved": True},
                "latency_scope": "In-process Socket.IO through the real agent, retrieval, explanations and serialization; browser/network excluded",
            }
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print("PASS: real-catalog G1 feedback integration", flush=True)
        finally:
            client.disconnect()


if __name__ == "__main__":
    main()
