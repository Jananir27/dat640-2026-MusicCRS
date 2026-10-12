# R6 Progress: Playlist-Based Recommendations

R6 recommends tracks from a listener's current playlist. We chose to derive genre and mood preferences from the catalog's existing `tag_list` instead of adding new metadata fields or changing the retrieval artifact. The tags contain useful genre and mood clues alongside unrelated values such as years and ratings, so `music_features.py` classifies only recognized genres and moods and ignores the rest.

For playlist recommendations, the service profiles the tracks already in the playlist, combines those preferences with any genre or mood stated in the request, and sends a preference-focused query through the shared **final G1 retriever**. It excludes tracks already in the playlist, then uses a diversity-aware selection step to balance preference fit with variety across genres and artists. Each suggestion is briefly explained using shared genre or mood evidence when available. This preserves the final retrieval implementation while using tags to personalize and diversify recommendations.

The same feature pipeline supports playlist creation (R8): explicit genre or mood phrases in the request become preferences, candidates come from the final retriever, and diverse selection builds a longer playlist. R6 returns up to five suggestions; R8 targets a longer, automatically sized set.

## Manual browser checks

- With **Numb — Linkin Park** in the playlist, “Recommend something based on my playlist” returned five tracks across different artists, including alternative, emo, and hip hop matches.
- With **Numb — 21 Savage** in the playlist, the same request returned five tracks by different artists, all tagged hip hop.
- “Create a playlist for chill jazz” produced a 12-track playlist spanning multiple artists, with jazz/chill-related results.

These checks show that the preference signal follows the seed playlist and that both recommendation and explicit mood/genre playlist generation work in the browser. They are manual behavior checks, not a full quality evaluation of recommendation relevance.
