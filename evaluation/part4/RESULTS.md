# Part 4 exploratory evaluation — 2026-10-23

**Label review update:** all 502 pooled labels have now received explicit AI
metadata review; 133 grades changed. See [the review report](LABEL_REVIEW.md) and
[`results/reviewed_metrics.json`](results/reviewed_metrics.json) for the current
assessment. The tables below preserve the original automatic-proxy results for
comparison. Neither label set is independent human ground truth.

This is Tobi's 25-query contribution, not the combined 50-query final study.
The catalog and Qwen embeddings/model are real cached TalkPlay data; the catalog
contains 47,071 tracks. Four systems were evaluated with three warm repetitions
per query. Model loading/warmup is outside the timed interval.

## Artifacts

- `tobi_queries.json`: five queries in each of the five agreed categories.
- `results/runs.json`: resolved references, 100 system/query rankings and 300
  timing samples, parameters, startup time, and the timing boundary.
- `results/judgments.csv`: 502 unique pooled query/track pairs for blind review,
  including the query context, title, artist, album, and catalog tags.
- `results/provisional_judgments.csv`: 0–2 draft labels with metadata-based reasons;
  every row is explicitly marked `metadata-proxy:REVIEW-REQUIRED`.
- `results/provisional_metrics.json`: overall and per-category metrics, latency,
  exclusion checks, and paired feedback deltas.
- `results/live_feedback_demo.json`: separate real G1/Qwen application check
  through in-process Socket.IO, including replies, evidence, and backend times.

## Provisional metadata-proxy results

These are not human or audio-based relevance judgments. The draft policy uses
catalog genre, mood, artist, and style matches and is independent of ranking
scores. Because labels rely on metadata terms, they can favor lexical retrieval.
Catalog tags themselves can be noisy. Review the labels using the team's agreed
guidelines before using these numbers as final evidence of musical relevance.

| System | nDCG@5 | Precision@5 | Mean warm request (ms) | Forbidden returned tracks |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.8412 | 0.8480 | 2.6 | 0 |
| Qwen semantic | 0.4344 | 0.4400 | 209.5 | 0 |
| Weighted-RRF hybrid | 0.8190 | 0.8320 | 206.1 | 0 |
| Hybrid + Rocchio feedback | 0.8187 | 0.8320 | 243.7 | 0 |

On the five feedback queries, paired nDCG@5 changed by **−0.0019** and
Precision@5 by **0.0000**. These draft results do **not** establish a feedback
quality improvement. The implementation is validated as changing scores and
honoring feedback; independent relevance assessment remains necessary.

All systems receive identical structured hard exclusions and known seed IDs.
Consequently, the comparison measures soft feedback effects rather than giving
the adaptive method a benefit from exclusions unavailable to the baselines.
It does not assess Janani's future natural-language exclusion parser.

Warm request timing includes retrieval, filtering, feedback where applicable,
and top-five explanations. Browser/network time is excluded. Times are local CPU
observations, not a production SLA. The semantic and hybrid implementations are
existing repo baselines; Janani's final R1 implementation should be benchmarked
again after integration. The online G1 bridge and R6 style selection are not the
same ranking configuration as the standalone hybrid baseline in this table.

## Application integration check

`python -m tests.part4_live_smoke` used the real G1 artifact and Qwen embeddings:

1. A seed-free rock request returned catalog rock candidates including ZZ Top,
   Fleetwood Mac, The Rolling Stones, Alice Cooper, and KISS.
2. Liking track 2 and disliking track 4 produced nonzero vector-feedback scores;
   subsequent suggestions withheld both already-rated songs and returned others.
3. Disliking the artist of a displayed result excluded that artist in the next set.
4. A feedback button tied to the older batch still targeted its original track.
5. Resetting feedback succeeded without altering playlist contents.
6. Adding Numb — Linkin Park and requesting playlist recommendations preserved
   the established results: By Myself, Faint, With You, Drowning, and Into You.

This uses the same Socket.IO event handlers as the UI, with no browser device ID,
so the user's saved playlists are untouched. It verifies integration rather than
claiming all returned songs were listened to or independently judged relevant.

## Correctness checks

- 97 Python unit/regression tests passed, including actual Socket.IO payload tests.
- 18 offline integrated cases passed (74 turns, real G1 ranking functions with
  explicitly synthetic dense rankings).
- Five selected frontend tests passed, including exact track/batch rating IDs,
  vote correction, saved playlist controls, and browser identity.
- TypeScript checking and the React production build passed. The build reports
  existing dependency/CSS deprecations and an existing unused connection-error state.

No tests infer relevance from their own feedback scores. The geometry tests use
small synthetic vectors only to check the algorithm mathematically.

## Next evaluation steps

Have both team members review the pooled labels and record assessor identities;
merge Janani's 25 queries and baseline adapter; repeat the run and report paired
before/after results using the final shared judgment policy. Preserve these
provisional artifacts rather than presenting them as final human ground truth.
