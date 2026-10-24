# Part 4 — Tobi: adaptive relevance feedback and explanations

## Implemented scope

This implements R2 (adaptive feedback), R3 (grounded explanations), and Tobi's
25-query contribution and evaluation tools for R4. It adapts the existing G1
retriever; it does not replace Janani's planned R1 retrieval work. There are no
external LLM calls in this path. Existing R5–R8 operations remain available.

The current application uses Qwen3-Embedding-0.6B, already present in G1, rather
than silently substituting a different Sentence-BERT model. The feedback module
is model independent: Janani may supply another encoder if **all query and track
vectors in that request belong to the same embedding space**.

## Chat and UI

Start with `Recommend some rock music`, or add a playlist seed and ask for
recommendations. Every recommendation response includes numbered tracks,
catalog-grounded explanations, and per-song like/dislike buttons.

Supported examples:

```text
Recommend some rock music
I like track 2, but I don't like track 4
Recommend some rock music
I like first, second and fourth
I don't like the artist Metallica
Recommend rock but exclude Linkin Park
Recommend rock without Numb by Linkin Park
Recommend more songs like track 2
Show me something different
Reset my feedback
```

Positions refer to the latest displayed recommendation list, not the playlist.
Titles with multiple catalog versions require a unique displayed match or an
artist. "I like this song"/"I don't like this artist" requires a single displayed
track or an explicitly discussed track in that list. Ambiguous references ask for
clarification. Compound feedback is validated atomically before anything changes.
The latest vote wins, including a correction within one utterance.

Feedback is processed **before** R7 selection. `No, don't add any` is only a
rejection of playlist addition and does not fabricate dislike votes. `Show me
something different` skips that displayed batch for one retrieval without
inventing negative votes. Likes/dislikes alone acknowledge preferences;
`Recommend more songs like ...` requests another set immediately.

Feedback lives in per-agent `ConversationState`, survives turns and playlist
changes within that connection, and is isolated from other users. It is **not
persisted across refresh, disconnect, process restart, or another device**. Saved
playlists remain persistent using their existing database. Reset explicitly
clears feedback and hard exclusions, without changing playlist contents.

The backend emits a `RECOMMENDATIONS` dialogue act whose `recommendations_json`
contains `{batch_id, tracks}`. The UI emits:

```json
{"batch_id":"<issued UUID>","track_id":"<catalog ID>","event":"thumbs_up"}
```

The server checks that this track belonged to that batch in **this session**.
It retains up to 20 immutable batches, so an older message cannot accidentally
rate the current track at the same number. Switching saved playlists clears
the old batches; expired or forged ratings are rejected. DialogueKit's default
feedback method is a no-op; `MusicChatNamespace.on_feedback` now routes ratings
into this application's validated handler. Generic response-level thumbs were
replaced with track-specific controls, avoiding an ambiguous whole-list vote.

## Rocchio-inspired algorithm

`musiccrs/relevance_feedback.py` implements:

```text
q_feedback = normalize(alpha*q_original
                       + beta*mean(normalized liked track vectors)
                       - gamma*mean(normalized disliked track vectors))
```

Defaults: alpha=1, beta=0.75, gamma=0.25. Empty positive/negative groups contribute
zero. The original query is re-encoded using the same encoder as the tracks only
when vector feedback exists. Dense embeddings retain signed coordinates; the
nonnegative clipping used for sparse term vectors is not appropriate here.

For each candidate:

```text
feedback_delta = cosine(candidate, q_feedback) - cosine(candidate, q_original)
final_score = base_score + 0.75*feedback_delta + 0.10*liked_artist_match
```

If there is no original query embedding, feedback centroids can still supply the
direction; the before cosine is zero. Missing, zero, nonfinite, or mismatched
vectors are ignored rather than fabricated. Without usable embeddings, hard
exclusions and artist preferences still work, but semantic adaptation is absent.
An evidence record identifies whether vector feedback actually applied.

The original-query anchoring and positive/negative centroids follow the
[Rocchio relevance-feedback formulation in Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/the-rocchio71-algorithm-1.html).
Weights are configurable, not claimed as experimentally optimal.

The G1 adapter's base score is explicitly `1 - rank / pool_size`, a **rank prior**.
It is not described as a raw BM25 score or a calibrated probability. The current
adapter reranks the retrieved candidate pool; it cannot recover an unseen track
outside that pool. Improving candidate recall belongs in the R1 retriever.

For seed-free discovery, the bridge strips command filler, uses
`FinalRetriever.retrieve_query()` to retain saved G1 stages without challenge
dialogue-header words, and requires catalog genre evidence when a genre was
explicitly requested. Existing playlist/R8 conversation retrieval is preserved.
Rated positive tracks are withheld from new suggestions while their vectors remain
available as feedback seeds, so "more like this" suggests additional tracks.

Hard exclusions are checked both when fetching catalog candidates and after
feedback reranking. Explicitly excluded tracks, disliked tracks, and tracks
crediting a blocked artist are removed, including collaborations. Exact normalized
artist identity is used rather than substring matching. A high feedback score
cannot override an exclusion. A positive vote reverses an earlier negative vote,
but cannot silently cancel a separate explicit exclusion.

## Interface for Janani

`RankedTrack` provides `track_id`, verified `metadata`, a finite comparable `score`,
optional `embedding`, optional raw `scores`, and `evidence`. Metadata includes the
catalog's original title/artist/album/tags fields. The module never invents names.

```python
from musiccrs.relevance_feedback import FeedbackState, RankedTrack, RocchioFeedback

feedback = FeedbackState()
feedback.rate("liked-id", True)
feedback.rate("disliked-id", False)
feedback.excluded_artists.add("linkin park")  # normalize with name_key()

candidates = [RankedTrack(
    track_id=row["track_id"], metadata=row, score=normalized_relevance,
    embedding=track_vector,
    scores={"bm25": lexical_score, "semantic": semantic_score},
)]
results = RocchioFeedback().rerank(
    candidates, feedback,
    embeddings={"liked-id": liked_vector, "disliked-id": disliked_vector},
    query=query_vector,
    excluded_ids=explicit_track_exclusions,
)
```

`DenseRetriever` and `FinalRetriever` now expose `encode_query(text)` and
`get_track_embeddings(ids)`. `MusicService` also accepts an `embedding_provider`
for isolated development. The shared service holds models/indexes; all mutable
votes, exclusions, references, and ranking evidence stay in each user's state.
Janani should apply structured exclusions before retrieval and pass them through
this final filter as well. A small conversational exclusion bridge currently
supports `exclude`, `excluding`, `without`, and `avoid`; it is not a replacement
for R1's eventual complete query parser.

## Grounded explanations

`musiccrs/recommendation_explanations.py` builds response text and evidence from:

- Requested genres/moods actually present in the candidate's catalog tags.
- Shared playlist artist, specific styles, and broad genres.
- An explicitly liked artist.
- A recorded cosine match of at least 0.5 to a positively rated catalog track.

Embedding similarity is called **metadata embedding similarity**, not an audio
or mood fact. No shared evidence yields the neutral "selected by the catalog
retrieval ranking" explanation. Each browser record includes structured evidence
and scores for inspection. Missing metadata never becomes a guessed attribute.

## Evaluation contribution and commands

`evaluation/part4/tobi_queries.json` contains 25 queries: five each for genre,
artist similarity, playlist context, exclusions, and mixed preferences/feedback.
Track references use title + artist and are resolved against SQLite before model
loading; absent references fail explicitly. Janani can append 25 more unique IDs.

The benchmark compares BM25, semantic Qwen, weighted-RRF hybrid, and that hybrid
with feedback. These baselines reuse existing retrieval implementations; they do
not pretend to evaluate Janani's not-yet-integrated R1 module. All four apply the
same structured hard constraints. This isolates soft relevance feedback from
simple exclusion removal and does not evaluate R1's language extraction quality.

```bash
# Real catalog/models; defaults download/cache missing dependencies as in the app.
python -m retrieval.evaluation.part4 run \
  --output-dir evaluation/part4/results --pool-depth 10 --repeats 3

# Optional transparent draft labels; review rather than treating these as human truth.
python -m retrieval.evaluation.part4 draft-judgments \
  --runs evaluation/part4/results/runs.json \
  --pool evaluation/part4/results/judgments.csv \
  --output evaluation/part4/results/provisional_judgments.csv

python -m retrieval.evaluation.part4 score \
  --runs evaluation/part4/results/runs.json \
  --judgments evaluation/part4/results/provisional_judgments.csv \
  --output evaluation/part4/results/provisional_metrics.json
```

If using the repository's existing laptop cache, set `HF_HOME` to the absolute
`cache/huggingface` path. `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` prevents network
lookups when all model/dataset files are already cached.

Proposed rubric until the team confirms its shared guidelines: 0 = unrelated or
violates a hard constraint; 1 = partial musical match; 2 = strong musical match.
For draft labels, a requested artist, specific style, or matching genre with all
requested mood conditions gives 2; genre/mood-only partial matches give 1;
otherwise 0. Multi-genre requests are interpreted as alternatives, not a demand
that every song exhibit every genre. Catalog tags can be noisy: these labels are
marked `metadata-proxy:REVIEW-REQUIRED`, contain reasons, and require assessor
review for a defensible final study. They never use retrieval/feedback scores.

Candidate pooling unions each method's top 10 by query/track ID. The CSV hides
system/rank information from assessors. Fill every grade and assessor field;
unjudged top-five results cause scoring to fail. nDCG@5 uses `(2^grade - 1)` gain
and logarithmic discount; Precision@5 counts grades 1–2 and divides by five even
for short lists. Existing challenge binary metrics are unchanged.

The report includes macro scores, warm-request mean/p50/p95 latency, exclusion
violations, and paired before/after deltas on feedback queries. Startup/model
loading/warmup is excluded and reported separately. Repeats do not train or tune
weights on judgments. Single-repeat measurements are exploratory. Do not claim
that feedback improves quality unless independently reviewed results support it.

## Verification

All 502 pooled relevance labels have received an explicit AI metadata review,
with 133 grade changes, preserved originals, row-by-row reasons, and recalculated
metrics. See [`evaluation/part4/LABEL_REVIEW.md`](evaluation/part4/LABEL_REVIEW.md).
The reviewed labels still require independent human assessment; they must not be
presented as human ground truth. Reproduce the frozen review with:

```bash
python -m retrieval.evaluation.review_labels
```

The completed local run, results, and limitations are recorded in
[`evaluation/part4/RESULTS.md`](evaluation/part4/RESULTS.md).

```bash
MUSICCRS_NLU_LOCAL_ONLY=1 HF_HUB_OFFLINE=1 python -m unittest \
  tests.test_relevance_feedback tests.test_part4_evaluation \
  tests.test_music_intelligence tests.test_music_features \
  tests.test_final_retrieval tests.test_saved_playlists tests.test_musiccrs_platform
python -m tests.full_music_smoke --offline --quiet
python -m tests.part4_live_smoke  # real catalog, cached Qwen and saved G1 artifact
cd frontend
CI=true npm test -- --watchAll=false --runInBand --runTestsByPath \
  src/tests/RecommendationFeedback.test.tsx src/tests/PlaylistPanel.test.tsx src/tests/deviceIdentity.test.ts
./node_modules/.bin/tsc --noEmit
```

Geometry tests use small, explicitly synthetic vectors to verify direction and
ranking changes. Socket.IO tests verify the actual event/annotation contract.
Frontend tests check exact identities, vote corrections, and single-event delivery.
These are correctness tests; real-catalog retrieval quality is evaluated separately.

Manual demo: request rock, like track 2 and dislike track 4, request rock again,
inspect changes and evidence, block an artist and repeat, rate a song in an older
message, test an ambiguous pronoun, then reset feedback. Playlist addition remains
an independent action throughout.
