"""Exercise R5–R8 with the real TalkPlay track catalog and BM25 index.

Run from the repository root with ``./.venv/bin/python -m tests.live_music_smoke``.
The first run needs Hugging Face access unless the dataset is already cached.
BM25's tag-enriched index is built under ``./cache`` and reused afterwards.
"""

from musiccrs.music_intelligence import MusicIntelligence
from musiccrs.music_service import ConversationState, MusicService


def main() -> None:
    service = MusicService()
    catalog_size = len(service.catalog.metadata_dict)
    if not catalog_size:
        raise RuntimeError("The TalkPlay track catalog loaded with no records.")
    print(f"Loaded {catalog_size:,} real catalog tracks.")

    # Pick a usable real track as the starting playlist seed.
    seed = next(
        (
            track
            for track in service.catalog.metadata_dict.values()
            if service.display_value(track.get("track_name"))
            and service.display_value(track.get("artist_name"))
            and service.display_value(track.get("tag_list"))
        ),
        None,
    )
    if seed is None:
        raise RuntimeError("No catalog track has both a name, artist, and tags.")

    state = ConversationState()
    service.add_tracks(state, [seed["track_id"]])
    intelligence = MusicIntelligence(service)
    title = service.display_value(seed.get("track_name")).split(", ")[0]
    print(f"Seed track: {service.display_track(seed)}")

    # R5: answer from actual track metadata, resolving ambiguity if needed.
    answer, changed = intelligence.handle(f"Who performs {title}?", state)
    if answer and "Which track did you mean?" in answer:
        answer, changed = intelligence.handle("option 1", state)
    if not answer or changed:
        raise AssertionError("R5 did not answer the live catalog question.")
    print(f"R5: {answer}")

    # R6: this lazily loads/builds the real BM25 index on first use.
    suggestions, changed = intelligence.handle(
        "Recommend something based on my playlist",
        state,
    )
    if changed:
        raise AssertionError("R6 should present suggestions without changing the playlist.")
    print(f"R6: {suggestions}")

    pending = list(state.pending_recommendation_ids)
    if pending:
        if len(pending) >= 2:
            selection = "Add the first two"
            expected_added = pending[:2]
        else:
            selection = "Add all of them"
            expected_added = pending
        response, changed = intelligence.handle(selection, state)
        if not changed:
            raise AssertionError("R7 did not apply the selection from live suggestions.")
        if not set(expected_added).issubset(state.playlist_ids):
            raise AssertionError("R7 did not add the selected live catalog tracks.")
        print(f"R7: {response}")
    else:
        print("R7: skipped because BM25 returned no eligible suggestions.")

    # R8: generation should replace the current playlist only after retrieval.
    response, changed = intelligence.handle(
        "Create a playlist for a short commute",
        state,
    )
    if not changed or not state.playlist_ids:
        raise AssertionError("R8 did not create a playlist from live retrieval results.")
    if len(state.playlist_ids) != len(set(state.playlist_ids)):
        raise AssertionError("R8 returned duplicate track IDs.")
    if any(track_id not in service.catalog.metadata_dict for track_id in state.playlist_ids):
        raise AssertionError("R8 returned a track ID outside the live catalog.")
    print(f"R8: {response}")
    print("Live-data smoke checks passed.")


if __name__ == "__main__":
    main()
