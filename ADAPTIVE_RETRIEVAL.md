# Adaptive retrieval improvement — 2026-10-24

The isolated implementation is
[`musiccrs/adaptive_retrieval.py`](musiccrs/adaptive_retrieval.py). It improves
measured relevance on the current **AI-reviewed metadata benchmark**. It does
not change the running chat or any previous implementation, labels, or results.

## Measured result

All comparisons below use the **same expanded judgment pool**. The cleanest
comparison holds query processing, genre constraints, weights, scoring features,
and diversity policy constant, changing whether feedback can retrieve additional
candidates.

| New experimental mode | nDCG@5, all 25 queries | Precision@5, all 25 | nDCG@5, five feedback queries | Precision@5, feedback queries |
| --- | ---: | ---: | ---: | ---: |
| Baseline, no soft feedback | 0.8647 | 0.9840 | 0.4755 | 1.0000 |
| Feedback reranking within the original pool | 0.8847 | 0.9840 | 0.5752 | 1.0000 |
| Feedback-aware retrieval and reranking | **0.9377** | **0.9840** | **0.8405** | **1.0000** |

Feedback-aware retrieval adds **0.2653 absolute nDCG@5** over reranking alone on
the feedback queries, or **0.0531** averaged over all 25 queries. Precision is
unchanged: the new experimental baseline already returns partially relevant
items, and Precision@5 counts both partial and strong relevance as successes.

Three feedback cases improve; two are unchanged. None regresses under these
judgments. All 20 queries without feedback return identical rankings across the
three new modes. No checked excluded track or credited artist appears in final
recommendations.

| Feedback request | Reranking only nDCG@5 | Expanded retrieval nDCG@5 | Top-five tracks outside original pool |
| --- | ---: | ---: | ---: |
| Rock: like Numb — Linkin Park, dislike Numb — 21 Savage | 0.5339 | 0.9125 | 4 |
| Hip hop: reverse those preferences | 0.3333 | 0.9125 | 4 |
| Pop: liked Skyfall — Adele | 0.4748 | 0.4748 | 1 |
| Alternative rock: like In the End, exclude Metallica | 0.5339 | 0.9026 | 4 |
| Energetic rock: like Numb, exclude 21 Savage | 1.0000 | 1.0000 | 3 |

For the first case, Faint, Somewhere I Belong, Figure.09, and From The Inside
enter the top five from outside the original pool. For the hip-hop case, four
verified 21 Savage tracks enter. The improvement largely reflects stronger
seed-style and same-artist relationships; requests requiring different artists
have not been evaluated.

The Skyfall case remains weak: broad pop similarities do not establish a strong
cinematic soul/pop-ballad match. This limitation is preserved in the results.

## What changed

The earlier isolated experiment found new candidates but kept a strong lexical
rank prior. The new adapter:

1. Retrieves original-query BM25 and semantic candidates.
2. Searches the full available embedding index with the Rocchio-adjusted query.
3. Also searches with the centroid of liked tracks, retaining all original
   candidates so expansion cannot discard baseline choices.
4. Scores explicit seed embedding similarity, specific catalog style overlap,
   verified credited-artist relationships, and negative similarity.
5. Applies identical explicit genre eligibility, hard exclusions, mood preferences,
   and a modest artist-repetition penalty in all comparison modes.

The score before diversity is:

```text
score = (1 - influence) * normalized_fusion_prior
        + influence * feedback_context
        + 0.15 * explicit_requested_mood_match

positive feedback_context = 0.50 * mean_liked_embedding_cosine
                            + 0.30 * maximum_liked_style_cosine
                            + 0.20 * liked_seed_artist_match
                            - 0.25 * (0.75 * mean_disliked_embedding_cosine
                                      + 0.25 * maximum_disliked_style_cosine)
```

With negative feedback alone, adjusted-query cosine supplies the positive
anchor. Explicit artist-only likes also contribute without inventing track votes.
No soft feedback means `influence=0`. Artist repetition subtracts 0.04 per already
selected occurrence, equally across modes. These are ranking scores, not probabilities.

The selected/default influence is **0.8**. BM25/semantic fusion weights remain
1.0/0.25 and the RRF constant remains 60. Rocchio query weights remain
alpha=1, beta=0.75, gamma=0.25. The seed centroid supplies an additional branch
of up to 50 candidates; candidate budgets are therefore larger in the expanded
mode, as intended by this recall experiment. It is not an equal-budget ablation.

Only catalog facts and request/feedback inputs enter production ranking. The
module contains no evaluation grades, query IDs, or artist-specific ranking rules.
Legacy tag alias gaps for composite genres and recognized subgenres are handled
inside this isolated module. Conflicting source tags can still cause failures.

## Development and assessment protocol

- Ten development queries use seed artists different from those in the 25-query
  evaluation: Nirvana, Foo Fighters, Kendrick Lamar, Drake, Norah Jones,
  Amy Winehouse, The Chemical Brothers, Massive Attack, John Coltrane, and Deftones.
- Three predeclared influence profiles (0.35, 0.60, 0.80) were compared using
  **development-only metadata-proxy labels**. The 0.80 profile was selected and
  its configuration hash recorded before running the 25-query evaluation.
- The evaluation uses three warm repetitions per new method/query: 75 new
  rankings and 225 timing observations. New-method mean request time is about
  384 ms on the local CPU, excluding model loading, explanations, and transport.
  This is not a speed improvement claim; older timings have a different scope.
- Every one of the previous **502 labels is preserved**. The 232 newly pooled
  query/track pairs received explicit AI review under the same rubric. The common
  pool contains **734 pairs**. Missing judgments were not scored as zero.
- All titles, artists, albums and tags match SQLite exactly. The review passed
  157 shared-candidate consistency comparisons across related queries.
- Two conflicting tag records were checked externally: Boards of Canada's
  Dayvan Cowboy is on an [Electronic release](https://music.apple.com/us/album/the-campfire-headphase/81696254),
  while Justin Bieber's Stuck In the Moment is classified as
  [Pop](https://www.shazam.com/song/1440661549/stuck-in-the-moment).
  Their contradictory K-pop/extreme-metal tags did not earn relevance credit.
  The catalog itself was not changed.
- A separate standard-library calculation agrees with the evaluator for all
  175 original-plus-new rankings, seven method averages, and all paired deltas.
- The 75 new rankings were replayed against the frozen configuration to check
  exact top-ten reproducibility and score agreement.

The existing evaluation had already been inspected earlier in this task, so it
is **not an unseen holdout**. Development and evaluation labels both depend on
metadata and may favor metadata-based ranking. No audio review or independent
human assessment occurred. With only five feedback cases, these results do not
establish general musical-quality improvement or statistical significance.

When the four older systems are rescored against the expanded pool, the old
hybrid-plus-feedback system has nDCG@5 0.6669 and Precision@5 0.8320 overall.
Do not compare 0.6669 directly with the earlier 0.7388 report: additional strongly
relevant candidates changed the shared ideal ranking without changing old labels.
Gains over that older system include common query/genre-processing changes as
well as feedback; the matched three-mode comparison above isolates the incremental
benefit of feedback-aware candidate generation more clearly.

## Use the isolated implementation

```python
from musiccrs.adaptive_retrieval import AdaptiveMusicRetriever

# Reuse already loaded models. Defaults match the development-selected profile.
retriever = AdaptiveMusicRetriever(catalog, lexical, dense)
result = retriever.retrieve(
    "rock music",
    state.feedback,
    excluded_ids=set(state.playlist_ids) | state.skip_recommendation_ids,
    topk=5,
    # genres={"rock"}, moods=set(), allowed_ids=eligible_ids,  # structured R1 constraints
)
```

The returned tracks use the existing `RankedTrack` contract and carry score and
ranking evidence. The caller continues to own feedback across turns. Integration
with the live chat remains opt-in; no existing application wiring was changed.

## Reproduce

```bash
export HF_HOME="$PWD/cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Choose new output directories: old experiments are never overwritten.
.venv/bin/python -m retrieval.evaluation.adaptive develop \
  --output-dir /private/tmp/musiccrs-adaptive-development
.venv/bin/python -m retrieval.evaluation.adaptive evaluate \
  --config /private/tmp/musiccrs-adaptive-development/selected_config.json \
  --output-dir /private/tmp/musiccrs-adaptive-evaluation --repeats 3

# To reproduce the recorded scores directly from the frozen rankings and labels:
.venv/bin/python -m retrieval.evaluation.adaptive score \
  --runs evaluation/part4/adaptive_evaluation_v1/runs.json \
  --judgments evaluation/part4/adaptive_evaluation_v1/reviewed_judgments.csv \
  --output /private/tmp/musiccrs-adaptive-metrics.json

.venv/bin/python -m unittest tests.test_adaptive_retrieval \
  tests.test_feedback_retrieval tests.test_relevance_feedback \
  tests.test_part4_evaluation tests.test_final_retrieval
```

Development artifacts are in
[`evaluation/part4/adaptive_development_v1`](evaluation/part4/adaptive_development_v1/selected_config.json).
Evaluation [metrics](evaluation/part4/adaptive_evaluation_v1/metrics.json),
[reviewed labels](evaluation/part4/adaptive_evaluation_v1/reviewed_judgments.csv),
[explicit review decisions](evaluation/part4/adaptive_evaluation_v1/review_decisions.json),
[review audit](evaluation/part4/adaptive_evaluation_v1/review_audit.json), and
[replay verification](evaluation/part4/adaptive_evaluation_v1/replay_verification.json)
are kept separately from the previous results.

For final course evidence, obtain independent human judgments, add Janani's 25
queries, and evaluate the final integrated R1/R2 system. Preserve both successful
and unchanged cases when reporting these prototype results.
