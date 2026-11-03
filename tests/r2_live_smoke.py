"""Real catalog/R1/R2 and Socket.IO checks with scripted or live LLM output.

Default: scripted preferences. --live-llm: real configured Ollama service.
No device identity, saved-playlist edits or relevance labels are created.
"""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from musiccrs.adaptive_retrieval import metadata_features
from musiccrs.llm_preference_parser import LLMPreferenceParser, PreferenceExtractionError
from musiccrs.music_service import MusicService
from musiccrs.musiccrs import MusicCRS, MusicCRSPlatform, MusicChatNamespace
from musiccrs.relevance_feedback import name_key, values
from retrieval.evaluation.part4 import resolve_track


class ScriptedPreferenceClient:
    """Fixed test responses for these smoke-test requests; not an LLM emulator."""
    def __init__(self):
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        request = json.loads(kwargs["prompt"].split("USER_REQUEST (JSON string):\n", 1)[1]).casefold()
        return {"response": json.dumps({
            "genre": ["jazz" if "jazz" in request else "rock"], "mood": [],
            "preferred_artists": [],
            "excluded_artists": ["Metallica"] if "metallica" in request else [],
            "excluded_tracks": [], "use_playlist_context": True, "avoid_playlist_artists": False,
        })}


class RecordedPreferenceParser:
    """Record parsed preferences and durations; never record client headers."""
    def __init__(self, delegate):
        self.delegate = delegate
        self.calls = []

    def parse(self, request):
        started = perf_counter()
        preferences = self.delegate.parse(request)
        self.calls.append({"request": request, "preferences": preferences,
                           "duration_ms": (perf_counter() - started) * 1000})
        return preferences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--live-llm", action="store_true",
                        help="Call the configured external Ollama service; requires OLLAMA_API_KEY")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Use a new output path: {output}")
    # Fail early on missing credentials or service/extraction errors, before
    # loading the catalog/model. Live mode never constructs a scripted client.
    delegate = LLMPreferenceParser() if args.live_llm else LLMPreferenceParser(client=ScriptedPreferenceClient())
    preferences = RecordedPreferenceParser(delegate)
    if args.live_llm:
        preflight = preferences.parse(
            "Recommend energetic rock songs similar to my playlist, but exclude Linkin Park "
            "and suggest something like Green Day.")
        assert "linkin park" in {name_key(a) for a in preflight["excluded_artists"]}, "Live parser missed the artist exclusion"
        assert "green day" in {name_key(a) for a in preflight["preferred_artists"]}, "Live parser missed the preferred artist"
        assert preflight["use_playlist_context"], "Live parser missed playlist context"
    service = MusicService()
    service._get_retriever()
    seed = resolve_track({"title": "Numb", "artist": "Linkin Park"}, service.catalog)
    turns = []
    with (patch("musiccrs.musiccrs.get_shared_music_service", return_value=service),
          patch("musiccrs.llm_preference_parser.LLMPreferenceParser", return_value=preferences),
          patch.dict(os.environ, {"MUSICCRS_USE_LLM": "false", "MUSICCRS_ADVANCED_LLM": "true"})):
        platform = MusicCRSPlatform(MusicCRS)
        platform.socketio.on_namespace(MusicChatNamespace("/", platform))
        client = platform.socketio.test_client(platform.app)
        client.get_received()

        def send(text, *, event="message", data=None):
            started = perf_counter()
            client.emit(event, data if data is not None else {"message": text})
            message = [e for e in client.get_received() if e["name"] == "message"][-1]["args"]["message"]
            payload = next((json.loads(a["value"]) for act in message.get("dialogue_acts", [])
                            if act["intent"] == "RECOMMENDATIONS" for a in act["annotations"]
                            if a["slot"] == "recommendations_json"), None)
            feedback_status = next((json.loads(a["value"]) for act in message.get("dialogue_acts", [])
                                    if act["intent"] == "FEEDBACK_STATE" for a in act["annotations"]
                                    if a["slot"] == "feedback_state_json"), None)
            turns.append({"query": text, "event": event, "response": message["text"],
                          "backend_response_ms": (perf_counter() - started) * 1000,
                          "recommendations": payload, "feedback_status": feedback_status})
            print(json.dumps({"query": text, "tracks": [] if not payload else
                              [t["track_name"] for t in payload["tracks"]]}, ensure_ascii=False), flush=True)
            return payload

        def ids(batch):
            return [t["track_id"] for t in batch["tracks"]]

        def ranking(batch):
            return [t["evidence"]["ranking"] for t in batch["tracks"]]

        try:
            seed_free = send("Recommend rock music")
            assert seed_free and 1 <= len(ids(seed_free)) <= 5
            send(f"/playlist add {seed}")
            before = send("Recommend rock music")
            assert before and len(ids(before)) == 5
            liked, disliked = ids(before)[1], ids(before)[3]
            send("I liked track 2 and I didn’t like track 4")
            assert turns[-1]["feedback_status"]["votes"] == {liked: 1, disliked: -1}
            after = send("Recommend rock music, no Metallica")
            assert after and not {liked, disliked, seed} & set(ids(after))
            assert all(e["feedback_path"] == "adaptive_catalog" and e["feedback_applied"] for e in ranking(after))
            assert any(abs(t["scores"]["feedback_delta"]) > 1e-6 for t in after["tracks"])
            assert all("metallica" not in {name_key(a) for a in values(service.get_track(t)["artist_name"])}
                       for t in ids(after))
            assert ids(before) != ids(after)

            polite_target = ids(after)[1]
            polite = send("Please recommend more songs like track 2")
            assert polite and polite_target not in ids(polite)
            persistent = send("Recommend rock music")
            assert persistent and all("metallica" not in {name_key(a) for a in values(service.get_track(t)["artist_name"])}
                                      for t in ids(persistent))
            different = send("Show me something different")
            assert different and not set(ids(persistent)) & set(ids(different))
            restored = send("Recommend rock music")
            assert restored and ids(restored) == ids(persistent)

            mixed_target = ids(restored)[0]
            jazz = send("I like track 1 and recommend jazz music")
            assert jazz and mixed_target not in ids(jazz)
            assert all("jazz" in metadata_features(service.get_track(t))[0] for t in ids(jazz))
            saved_votes = turns[-1]["feedback_status"]["votes"]
            send("I like track 2 and I did not like track 99")
            assert "1 to" in turns[-1]["response"]
            assert turns[-1]["feedback_status"]["votes"] == saved_votes
            unchanged = send("Recommend jazz music")
            assert unchanged and ids(unchanged) == ids(jazz)

            send("Rate a track from the original batch", event="feedback", data={
                "batch_id": before["batch_id"], "track_id": ids(before)[0], "event": "thumbs_down"})
            assert "disliked" in turns[-1]["response"]
            send("Reset my feedback")
            assert "cleared" in turns[-1]["response"]
            assert turns[-1]["feedback_status"]["votes"] == {}
            reset = send("Recommend rock music")
            assert reset and all(not e.get("feedback_applied") for e in ranking(reset))
            assert all(e["feedback_path"] == "r1_pool_rocchio" for e in ranking(reset))

            report = {
                "catalog_fixture": False, "catalog_size": len(service.catalog.metadata_dict),
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                "llm_preferences": ("live external Ollama responses through the actual JSON validator"
                                    if args.live_llm else "scripted test responses through the actual JSON validator; no external LLM call"),
                "llm_model": delegate.model,
                "external_llm_calls": len(preferences.calls) if args.live_llm else 0,
                "preference_parse_calls": len(preferences.calls),
                "preference_requests": preferences.calls,
                "checks": {name: True for name in (
                    "positive_negative_feedback", "adaptive_catalog_path", "ranking_changes", "nonzero_vector_feedback",
                    "rated_seeds_withheld", "persistent_r1_exclusions", "polite_more_like", "temporary_skip_cleared",
                    "mixed_feedback_and_new_genre", "invalid_compound_atomic", "older_batch_rating", "reset_to_r1_ranking",
                    "seed_free_advanced", "server_vote_snapshots")},
                "turns": turns,
                "latency_scope": "In-process Socket.IO, retrieval, feedback, explanations and serialization; browser/network excluded",
                "quality_claim": "Integration correctness only; no human relevance assessment or new nDCG/Precision measurement.",
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x") as stream:
                json.dump(report, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            print("PASS: R1/R2 advanced-route Socket.IO integration", flush=True)
        finally:
            client.disconnect()


if __name__ == "__main__":
    try:
        main()
    except PreferenceExtractionError as exc:
        raise SystemExit(f"Could not run the LLM integration check: {exc}")
