# MusicCRS verification

Run from the project root in the activated Python environment:

```bash
python -m unittest tests.test_final_retrieval tests.test_music_intelligence tests.test_bm25_cache
python -m tests.full_music_smoke --offline --quiet
```

Saved-playlist persistence and Socket.IO checks (temporary SQLite databases;
fixture catalog, no model downloads):

```bash
python -m unittest tests.test_saved_playlists tests.test_musiccrs_platform
```

These check device isolation, reopening the database, reconnecting sockets,
auto-saving additions/removals/clear, keeping generated playlists as separate
entries, renaming/deleting, and discarding stale recommendation selections
when switching playlists. Frontend controls and browser identity tests:

```bash
cd frontend
CI=true npm test -- --watchAll=false --runInBand --runTestsByPath src/tests/PlaylistPanel.test.tsx src/tests/deviceIdentity.test.ts
```

The unit tests check the final pipeline's protected prefix, candidate expansion,
separate 50/100 retrieval depths, saved-artifact replay, missing-configuration
failure, context handling, playlist exclusions, and existing music operations.

Offline smoke mode exercises the agent, playlist events, real BM25, and real
G1 fusion/reranking functions using eight fixture tracks, synthetic settings,
and deterministic fake dense rankings. It does not test Qwen, semantic quality,
or the submitted development score. The larger unit fixture exercises the
reranked tail, which the eight-track smoke fixture cannot do.

For the real catalog and embedding pipeline, first follow
[the final retrieval setup](../retrieval/FINAL_RETRIEVAL.md), then run:

```bash
python -m tests.full_music_smoke --quiet
```

Omit `--quiet` to see the full conversation transcript. The legacy command
`python -m tests.live_music_smoke` invokes the same suite. The test requires
`FinalRetriever`; it does not fall back to BM25. It disables only the chat LLM,
not the Qwen embedding model. No web server or simulator upload is started.

For manual UI verification, run the backend and frontend, then:

1. Add a known track and confirm the playlist panel updates.
2. Ask who performs it, its album, release date, and genre.
3. Request related songs and inspect whether the explanations and music fit.
4. Select recommendations by position, exclusion, and accept/reject-all.
5. Create playlists for a mood and an activity; confirm the shared panel is
   replaced and the tracks fit the request.
6. Clear the playlist and confirm old recommendations cannot be accepted.

Backend tests validate behavior and payloads. Browser rendering, socket
transport, and recommendation quality still require these manual checks.
