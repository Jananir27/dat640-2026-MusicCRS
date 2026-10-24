# Part 4 relevance-label review — 2026-10-23

All **502 pooled query/track pairs** across **25 queries** and **208 distinct
catalog tracks** have explicit AI review decisions. **133 grades changed**:
107 were reduced and 26 increased. The original pool, provisional labels,
retrieval rankings, and timing measurements are preserved.

These are **AI-reviewed metadata judgments, not human relevance judgments**.
No audio was played. The reviewer had already seen the draft labels and aggregate
results, so this is neither independent nor fully blind. After an initial reviewed
metrics calculation, the final evidence check reduced three trap-similarity
labels: the actual 21 Savage seed has no trap tag. This amendment corrects an
unsupported seed inference; no grades were tuned to improve a system's score.

## Reviewed artifacts

- [Reviewed labels](results/reviewed_judgments.csv): one row per pooled pair,
  including the original grade, revised grade, rationale, verbatim catalog tags,
  caveats, review priority, and explicit AI assessor identity.
- [Every decision, grouped by query](LABEL_REVIEW_DETAILS.md): a readable table of
  all 502 decisions and their reasons.
- [Explicit decision manifest](label_review_decisions.json): the 0–2 rubric,
  interpretation rules, query-specific grade choices, and track caveats. Track
  keys map to catalog UUIDs in the audit file.
- [Audit evidence](results/label_review_audit.json): input/output hashes, exact
  pool coverage, catalog checks, seed metadata, related-query consistency checks,
  per-query counts, and all grade transitions.
- [Recalculated metrics](results/reviewed_metrics.json): the original rankings
  scored with the reviewed labels, explicitly identified as AI metadata review.

## Judgment policy

**0** means no defensible match in the supplied metadata, insufficient music
evidence, or a hard-constraint violation. With incomplete tags, this is not proof
that a song would sound irrelevant. **1** means a partial/adjacent match, only one
part of a compound request, or broad genre overlap without convincing contextual
similarity. **2** requires strong metadata support for the actual request,
including mood and seed similarity when those are requested.

The literal query and its context take priority over the helper genre arrays.
“Alternative rock” is treated as a specific style. “Rock and hip hop” in a mixed
playlist allows compatibility with either seed. A disliked song is forbidden,
but “less like” it does not ban its entire genre. Explicit artist exclusions
apply to collaborators. Title/artist identity is checked as well as catalog ID
for excluded songs and previously supplied seeds.

Specificity matters: a generic rock match can earn 2 for a broad rock request
and only 1 for “rock like Numb.” The grade evaluates the full request, not just
whether any catalog tag overlaps a helper array. Dominant conflicting styles and
weak incidental tags are recorded as caveats rather than treated as proof.

## Material corrections

| Query/candidate | Draft → reviewed | Reason |
| --- | --- | --- |
| Alternative rock / Darkest Things — The Submarines | 2 → 0 | Indie/Pop/Alternative and mellow are documented; rock is not. |
| Rock excluding Numb / rock + roll — EDEN | 0 → 2 | Alternative Rock is explicitly documented; the title is not the evidence. |
| Pop excluding Adele / Get Lucky — Agapornis | 0 → 2 | Latin Pop qualifies as pop; no language restriction was requested. |
| Upbeat pop / Chloe — GROUPLOVE | 1 → 2 | Indie pop and upbeat are both explicitly documented. |
| Daft Punk similarity / Showbiz — Muse | 2 → 0 | An isolated electronic tag does not override dominant alternative/progressive-rock evidence. |
| Daft Punk similarity / Episode 2, Todd Edwards — Daft Punk | 2 → 0 | Only Others is documented; relevant music is not established. This is not a claim that the recording is definitely spoken audio. |
| Jazz / Get A Hold — A Tribe Called Quest | 0 → 1 | Jazz-hop is a legitimate partial jazz relationship. |
| Feedback toward Numb — 21 Savage / Get Right Witcha — Migos | 2 → 1 | The candidate has a trap tag, but the actual seed does not. A shared specific seed style is not verified. |

Composite tags such as Hip-Hop/Rap and R&B/Soul are recognized as meaningful
catalog evidence even where the application's exact-alias helper misses them.
Shared film names, a song title containing “rock”/“pop”/“metal,” or a credited
artist whose name contains “Adele” do not establish musical relevance.

The strongest changes concern contextual evaluation: 62 of the 100 feedback
query labels changed. Broad genre matches had frequently been given full credit
without accounting for the liked seed. This affects how convincingly the
benchmark measures feedback rather than only genre matching.

## Validation

- Exactly 502 distinct pooled pairs have decisions; no extra or missing pairs.
- All titles, artists, albums, and tags exactly match the local SQLite catalog.
- The pool exactly matches the union of the four systems' saved candidate lists.
- No pooled candidate violates the checked artist/song/seed constraints.
- Related queries pass 98 consistency comparisons on shared candidates; adding
  a seed-similarity preference cannot raise a grade above its broad counterpart.
- An independent calculation using Python's standard math library agrees with
  the evaluator for all 100 rankings, method averages, and paired feedback deltas.
- Five evaluation unit tests pass, including AI/proxy/named-assessor/mixed label
  provenance. AI labels are never reported as independent human judgments.

The revised distribution is **166 grade-0**, **138 grade-1**, and **198 grade-2**
judgments. All rows retain `human_review_required=yes`; 360 receive high review
priority because of contextual similarity or specific catalog caveats.

## Results with the reviewed labels

| System | nDCG@5 | Precision@5 | Excluded tracks returned |
| --- | ---: | ---: | ---: |
| BM25 | 0.7603 | 0.8480 | 0 |
| Qwen semantic | 0.3553 | 0.4880 | 0 |
| Weighted-RRF hybrid | 0.7392 | 0.8320 | 0 |
| Hybrid + Rocchio feedback | 0.7388 | 0.8320 | 0 |

On the five feedback queries, the paired difference is **−0.00223 nDCG@5** and
**0.00000 Precision@5**. This does not establish a feedback improvement. No
statistical significance or general musical-quality conclusion is claimed.
Differences from the original report reflect revised labels, not new rankings
or retriever changes. Latencies are the original measurements, not a rerun.

## Reproduce and finish assessment

```bash
.venv/bin/python -m retrieval.evaluation.review_labels
.venv/bin/python -m unittest tests.test_part4_evaluation
```

The exporter verifies frozen input hashes, applies the explicit decisions, checks
catalog/constraint/consistency evidence, and recalculates the metrics. It does
not assign grades using a new automatic tag-matching rule. Changed source data
requires a new review.

For the final study, independent human assessors should judge the original blind
pool using the team's agreed rubric and listening where metadata is insufficient,
record their identities and disagreements, and preserve these AI labels separately.
In particular, review sparse/contradictory tags, inferred moods, covers/remixes,
the episode-like record, and all seed/artist similarity judgments. The combined
50-query evaluation and Janani's final R1 integration remain separate work.
