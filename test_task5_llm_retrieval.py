"""Live test, run from MusicCRS repository root on Gorina4."""
import json
from musiccrs.llm_preference_parser import LLMPreferenceParser
from musiccrs.llm_recommendations import recommend_with_llm, _artist_matches
from musiccrs.music_service import ConversationState, MusicService

service = MusicService()
state = ConversationState()
for title, artist in [("This Suffering", "Billy Talent"), ("Indestructible", "Rancid")]:
    matched = service.search_tracks(title=title, artist=artist, limit=1)
    if not matched:
        raise RuntimeError(f"Seed not found in catalog: {title} / {artist}")
    service.add_tracks(state, [matched[0]["track_id"]])

request = (
    "Recommend energetic rock music similar to my playlist, "
    "but exclude Billy Talent and suggest something like Green Day."
)
result = recommend_with_llm(request, LLMPreferenceParser(), service, state)
print("\n--- EXTRACTED PREFERENCES ---")
print(json.dumps(result["preferences"], indent=2, ensure_ascii=False))
print("\n--- FILTERED RECOMMENDATIONS ---")
for rank, track in enumerate(result["recommendations"], 1):
    print(f"{rank}. {service.display_track(track)}")
print("\nCandidate count:", result["candidate_count"])
print("Timing (seconds):", result["timing_seconds"])

assert "Billy Talent" in result["preferences"]["excluded_artists"], "LLM missed explicit exclusion"
assert all(not _artist_matches(track.get("artist_name"), {"billy talent"})
           for track in result["recommendations"]), "Excluded artist leaked into results"
assert all(track["track_id"] not in state.playlist_ids for track in result["recommendations"])
print("\nTask 5 safety checks passed (artist exclusion + no playlist repeats).")
if len(result["recommendations"]) < 5:
    print("NOTE: Fewer than 5 results; increase candidate coverage in the next iteration.")
if result["timing_seconds"]["total"] > 5:
    print("NOTE: End-to-end latency exceeds the course's 5-second target.")

# ============================================
# TASK 5 STEP 2: WARM RETRIEVAL PERFORMANCE
# ============================================

print("\n--- SECOND REQUEST (WARM SYSTEM) ---")

# Fresh conversation state, same loaded MusicService
warm_state = ConversationState()
warm_state.playlist_ids = list(state.playlist_ids)

warm_result = recommend_with_llm(
    request,
    LLMPreferenceParser(),
    service,
    warm_state
)

print("\n--- WARM RECOMMENDATIONS ---")

for rank, track in enumerate(
    warm_result["recommendations"], 1
):
    print(f"{rank}. {service.display_track(track)}")

print("\n--- PERFORMANCE COMPARISON ---")

print("First request:", result["timing_seconds"])
print("Warm request:", warm_result["timing_seconds"])

assert all(
    not _artist_matches(
        track.get("artist_name"),
        {"billy talent"}
    )
    for track in warm_result["recommendations"]
)

assert all(
    track["track_id"] not in warm_state.playlist_ids
    for track in warm_result["recommendations"]
)

print("\nWarm retrieval test passed!")

