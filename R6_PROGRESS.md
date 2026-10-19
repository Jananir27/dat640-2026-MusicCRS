# R6 Progress: Playlist-Based Recommendations

R6 recommends tracks from a listener's current playlist. We chose to derive genre and mood preferences from the catalog's existing `tag_list` instead of adding new metadata fields or changing the retrieval artifact. The tags contain useful genre and mood clues alongside unrelated values such as years and ratings, so `music_features.py` classifies only recognized genres and moods and ignores the rest.

For playlist recommendations, the service profiles the tracks already in the playlist, combines those preferences with any genre or mood stated in the request, and sends a query through the shared **final G1 retriever**. The query retains seed titles, artists, and specific catalog styles, including distinctions such as nu metal and alternative metal. It excludes tracks already in the playlist, then ranks candidates by genre/style cosine similarity to individual seed tracks. This avoids treating the union of a mixed playlist's tags as one musical style. Same-artist evidence and explicit preferences contribute to selection; retrieval rank and artist variety have smaller weights. Weak style matches are filtered rather than used to fill slots. Each suggestion is explained using shared specific styles, artist identity, genres, or moods. The final G1 retrieval implementation is preserved.

Playlist creation (R8) continues using the existing broad genre/mood features and diverse selection. The stricter style selection applies to R6. R6 returns up to five suggestions and reports when fewer than three eligible matches are available.

## Manual browser checks

After the 2026-10-19 R6 relevance fix, **Numb — Linkin Park** returned By Myself,
Faint, With You, Drowning — Crazy Town, and Into You — Dead By Sunrise. Each
result explained specific shared rock/metal styles. Suggestions did not alter
the panel, first-two selection added exactly those songs, and rejection left
the playlist unchanged. A **Numb — 21 Savage** check returned five relevant
same-artist tracks/collaborations; artist variety remains limited for that
candidate pool. The 18-case offline integrated smoke suite passed.

Earlier checks before the relevance fix:

- With **Numb — Linkin Park** in the playlist, “Recommend something based on my playlist” returned five tracks across different artists, including alternative, emo, and hip hop matches.
- With **Numb — 21 Savage** in the playlist, the same request returned five tracks by different artists, all tagged hip hop.
- “Create a playlist for chill jazz” produced a 12-track playlist spanning multiple artists, with jazz/chill-related results.

These checks show that the preference signal follows the seed playlist and that both recommendation and explicit mood/genre playlist generation work in the browser. They are manual behavior checks, not a full quality evaluation of recommendation relevance.
