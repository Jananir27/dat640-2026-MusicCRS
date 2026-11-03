# R2 integrated feedback workflow

The conversational feedback module now uses R1's existing BM25/dense indexes
through `musiccrs/feedback_workflow.py`. Standard recommendations and the optional
LLM preference route both call `MusicService.rerank_feedback()` before display.
`musiccrs/request_constraints.py` supplies the same catalog constraint resolution
to the standard and LLM routes. No additional embedding model is loaded by R2, and no user state is stored in
the shared index. This guide supersedes the original candidate-only integration
description in `PART4_FEEDBACK.md`.

## Tasks 1–6

| Task | Implementation |
| --- | --- |
| Interpret feedback | `feedback_language.py` recognises positive/negative votes, polite “more like” requests, and feedback combined with one new recommendation request in either order. |
| Resolve references | Existing number/title/artist resolution runs before generic recommendation routing. Ambiguous references ask for clarification. Whole compound feedback is validated before preferences change. |
| Maintain preferences | `ConversationState.feedback` stores latest votes, artist preferences and hard exclusions. Old UI buttons retain immutable track/batch identities. Temporary “different” skips clear after one request, including errors. |
| Apply Rocchio feedback | The existing `RocchioFeedback` updates the query using positive/negative centroids. The existing `AdaptiveMusicRetriever` can retrieve additional full-catalog candidates and score metadata/style and vector feedback. |
| Return valid tracks | The final adapter enforces R1 allowlists, requested genres, playlist IDs, rated seeds, artist/track exclusions and temporary skips. R1 LLM exclusions are resolved and stored before expansion. |
| Verify end to end | `tests/test_r2_workflow.py` covers the integration and regression cases. `tests/part4_live_smoke.py` and `tests/r2_live_smoke.py` exercise actual Socket.IO handlers with the real catalog and cached Qwen model. |

## Ranking and R1 handoff

The service accepts catalog records or the existing `RankedTrack` contract:
track ID, authoritative metadata, a comparable score, optional embedding, raw
source scores and evidence. The adapter always uses catalog metadata for display.
It retains genuine source scores when reranking supplied `RankedTrack` candidates.
ID-only G1 results use an explicitly named rank prior, not invented BM25 scores.

Without track votes or liked artists, R1's surviving candidate order is preserved.
Hard constraints can still remove results. Artist dislikes alone are hard filters.
This path inspects only R1's supplied candidates, avoiding a catalog-wide feature
scan. Full-catalog feedback uses the index's cached features and artist identities.

When track votes or liked artists exist and R1 provides complete indexes, the
adapter lazily creates one `AdaptiveMusicRetriever` over those shared indexes.
It uses the existing alpha=1, beta=0.75, gamma=0.25 Rocchio update and the existing
responsive adaptive configuration. It recomputes weighted-RRF candidates and
feedback/style scores; it does not reuse G1's post-reranking score as a raw lexical
score. The original candidate pool can therefore be expanded, including when
filtering left that pool empty. Query encoding retains `FinalRetriever`'s lock.

The adaptive scorer's configuration is unchanged: influence=0.8, negative
weight=0.25, mood weight=0.15 and artist diversity penalty=0.04. These are the
previously developed settings, not newly tuned against the demonstration.

For injected providers without complete indexes, the existing pool-based Rocchio
adapter remains available. Invalid or missing individual vectors are ignored by
the existing vector validation; hard filters still apply. Metadata feedback can
remain useful when a track has no vector, but no liked-vector similarity is
fabricated. A malformed query/index or provider failure returns a retry response
and preserves preferences instead of returning unfiltered suggestions.

UI payloads expose final scores and `evidence.ranking`, including `feedback_path`,
whether a returned track was outside R1's input pool, usable/missing seed IDs and
the recorded ranking signals. Embeddings are never included. On the adaptive
path, `feedback_delta` means the change in query cosine; the final score also
contains fusion, context, mood and diversity terms. It is not the entire score
change or a relevance judgment.

## Conversations and edge cases

```text
Recommend rock music
I liked track 2 and I didn't like track 4
Recommend rock music
Please recommend more songs like track 2
I like track 1 and recommend jazz music
Show me something different
Reset my feedback
```

Newly supported cases include polite prefixes, short “more like track 2” requests,
named/focused-song references, past-tense feedback, and a new genre request in the
same message as a vote. “More songs like my playlist” remains a playlist request;
it does not manufacture positive votes for its contents.

The acceptance tests also cover:

- Invalid positions and mixed valid/invalid feedback: no partial preference update.
- Literal catalog titles such as “Love and Hate” are not split into separate votes.
- Vote corrections and repeats: latest vote wins without increasing its weight.
- New genres: earlier rock feedback cannot override an explicit jazz constraint.
- Empty/failed requests: old numbered choices and payloads are cleared. Original
  UI batches remain separately identifiable until their normal expiration.
- Creating a replacement playlist can reuse tracks from the old playlist.
  Ordinary recommendations still withhold playlist seeds; explicit bans and
  feedback remain enforced in both modes.
- “Different”: the previous batch is withheld for one retrieval; no dislike votes
  are fabricated. Skips clear on successful, empty and failed requests.
- Exclusions: allowlists, explicit bans and collaborations remain enforced after
  full-catalog feedback retrieval. Likes cannot cancel a separate explicit ban.
- Same-title songs: LLM constraints use title/artist identity, retain an explicit
  artist qualifier from the request, and ask when multiple artists remain possible.
  Exact duplicate title/artist IDs are excluded together. Titles containing “by”
  are checked as literal titles before treating “by” as an artist separator.
- Artist identity: banning War does not ban The War on Drugs.
  A bare exact artist name takes priority over a same-named song (for example,
  Daft Punk). Use “track X” or “X by Y” to identify a song instead.
- The same resolver handles “no”, “not by”, “but not”, exclusion lists, quoted
  musical names, explicit featured-artist credits and remastered duplicates.
  “No Doubt” and quoted “No Surprises” remain positive references when requested.
- Invalid exclusions accompanying a vote are validated before that vote is saved.
  A positive seed elsewhere in a message cannot disambiguate a bare excluded title.
- Artist names such as Daft Punk do not create an unintended punk genre filter.
- “Avoid the style of [song]” records a negative track preference. Ambiguous
  artist pronouns on collaborations ask which credited artist the user means.
  Literal song titles containing “by” also work for positive feedback.
- Negative-only feedback, missing vectors, mismatched embedding dimensions and
  finite output scores.
- Shared indexes across users, reset after index creation, immutable old button
  batches, rejected expired/foreign ratings and saved-playlist switches.

Feedback persists across turns and playlist switches in the current connection.
It is not persisted across refresh, disconnect or restart. The reset command
clears votes, artist preferences and explicit exclusions. Supported feedback
templates are conservative; arbitrary conversation language is not guaranteed.
Natural-language R1 extraction beyond these templates remains R1's responsibility.

## Verification and reproduction

The implementation passed 202 Python tests (including the R4 finalization checks), eight selected frontend tests,
TypeScript checking and the frontend production build. The
standard real-catalog demonstration verifies vector adaptation, removed rated
tracks, blocked artists, older-batch ratings, reset and preserved playlist results.
The current advanced demonstration verifies 14 acceptance conditions over 15 turns.
Its LLM responses are explicitly scripted and passed through the real JSON
validator; it does not test external LLM availability or extraction accuracy.
The legacy integration smoke test also passes all 18 scenarios over 74 turns,
including repeated playlist creation with different listening descriptions.

```bash
MUSICCRS_NLU_LOCAL_ONLY=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv/bin/python -m unittest discover -s tests -p 'test_*.py'

# Set HF_HOME to the absolute repository cache/huggingface directory when needed.
# Use fresh output paths; the advanced demonstration refuses to overwrite files.
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MUSICCRS_ADVANCED_LLM=false \
  .venv/bin/python -m tests.part4_live_smoke --output /tmp/r2-standard-new.json
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv/bin/python -m tests.r2_live_smoke --output /tmp/r2-advanced-new.json
```

The optional real LLM route uses `MUSICCRS_ADVANCED_LLM=true` and a configured
Ollama service/token. It now supports requests with an empty playlist as well.
When disabled, the standard route supports seed-free discovery and feedback.

To check the external service, run the following in an environment that already
has `OLLAMA_API_KEY` configured. The Python command reads shell variables;
unlike Compose, it does not automatically load a `.env` file.

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv/bin/python -m tests.r2_live_smoke --live-llm \
  --output /tmp/r2-external-llm-new.json
```

Live mode first checks actual preference extraction, then runs the Socket.IO
workflow with real provider responses. It records parsed preferences and elapsed
times without recording authorization headers. It never falls back to scripted
responses. The completed live check on 2026-11-03 passed all 14 acceptance
conditions over 15 turns using the real catalog and external Ollama responses.
The report is `evaluation/part4/live_llm_check_v1/live_external_demo.json`.
UiS currently exposes `llama3.3:70b-instruct-q4_K_M`; the shorter default
`llama3.3:70b` returned “Model not found”. The successful local configuration
uses that available model and `OLLAMA_TIMEOUT_SECONDS=60`, since six seconds
expired. Compose forwards both settings and the advanced-route flag. This live
check establishes integration correctness; it does not establish a relevance
metric improvement.

The browser consumes per-session `FEEDBACK_STATE` snapshots from the backend.
Buttons reflect accepted votes and chat corrections, clear on reset, and disable
for expired/foreign batches or disconnects. A locally clicked button does not
claim success before the server confirms the vote. Standalone components retain
their original local-state fallback when no server-state context is supplied.

The current demonstration reports and edge-case/UI integration checks are
recorded in `evaluation/part4/r2_integration_v2/`. Earlier demonstration
snapshots remain local and are omitted from the submission changes.
They establish integration behaviour, not musical relevance or a new metric gain.
The previous R4 results remain historical artifacts. The new benchmark in
`retrieval/evaluation/integrated.py` evaluates this live standard workflow over
the combined 50 queries. See [INTEGRATED_EVALUATION.md](INTEGRATED_EVALUATION.md).
Final nDCG@5 and Precision@5 still require completed, appropriately reviewed
judgments. Older standalone pool-reranking metrics do not measure this adapter.
