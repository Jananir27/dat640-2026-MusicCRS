
from musiccrs.music_service import MusicService, ConversationState
from musiccrs.music_features import profile_tracks

service = MusicService()
state = ConversationState()

# Find two songs to use as the user's playlist.
seed_songs = [
    ("This Suffering", "Billy Talent"),
    ("Indestructible", "Rancid"),
]

seed_ids = []

for title, artist in seed_songs:
    matches = service.search_tracks(
        title=title,
        artist=artist,
        limit=1
    )
    if not matches:
        raise RuntimeError(f"Song not found: {title}")

    seed_ids.append(matches[0]["track_id"])

service.add_tracks(state, seed_ids)

print("\n--- USER PLAYLIST ---")
playlist = [
    service.get_track(track_id)
    for track_id in state.playlist_ids
]

for song in playlist:
    print(service.display_track(song))

genres, moods = profile_tracks(playlist)

print("\nPlaylist genres:", dict(genres))
print("Playlist moods:", dict(moods))

# Generate recommendations using playlist context.
query = "Recommend energetic rock songs similar to my playlist"

results = service.recommend_from_playlist(
    state,
    limit=5,
    request=query
)

print("\n--- PLAYLIST RECOMMENDATIONS ---")

for i, song in enumerate(results, 1):
    print(f"{i}. {service.display_track(song)}")
    print("   Tags:", song.get("tag_list", [])[:10])

# Recommended tracks must not already be in the playlist.
assert all(
    song["track_id"] not in state.playlist_ids
    for song in results
)

print(f"\nReturned {len(results)} recommendations")
print("Playlist exclusion check passed!")

