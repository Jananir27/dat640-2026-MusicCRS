# R4 comparative evaluation

The frozen integrated run contains 50 queries, 200 rankings and 554 pooled query/track pairs.

## Relevance judgments

Imported 150 submitted human reviews. 40 apply to the current pool; the rest refer to earlier candidate pools.

Human-reviewed coverage of the current pool: 7.2%. The human grades change 4 of those current-pool assessments.

Current assessment counts: {"new_ai_metadata": 322, "reused_ai_metadata": 192, "submitted_human": 40}.

| Review file | Pairs | Queries | Exact agreement with previous AI grades |
| --- | ---: | ---: | ---: |
| Tobi_75_Combined_Submitted_Reviews.csv | 75 | 25 | 86.7% (75 comparable) |
| Janani_75_Blind_Human_Review_For_Tobi.xlsx | 75 | 25 | 81.3% (75 comparable) |

Submitted reviewer identities are retained. Independence, listening and blinding were not independently verified. Sample agreement is not full-pool validation or inter-human agreement.

Human reviews override AI grades only for identical query, track and context evidence. They are never generalized to unreviewed songs. Remaining AI grades are provisional; the human sample is not treated as full-pool ground truth.

Recorded reversible metadata repairs: 3. Original files and human grades are preserved.

Only 11 of 75 returned Tobi-pool reviews match the originally exported seeded sample. Each query still has three reviews, but the returned sample must not be described as that seeded random selection. Agreement is descriptive for these submitted pairs.

## Comparative results

**MIXED JUDGMENTS — includes AI-reviewed metadata; requires independent human assessment**. This is not a fully human-judged evaluation.

| Method | nDCG@5 | Precision@5 | Mean warm ms | p50 ms | p95 ms | Exclusions returned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.8650 | 0.9440 | 207.8 | 21.6 | 909.8 | 0 |
| semantic | 0.6236 | 0.7920 | 514.5 | 240.2 | 1689.0 | 0 |
| hybrid | 0.8636 | 0.9400 | 853.9 | 464.1 | 2397.4 | 0 |
| hybrid_feedback | 0.9259 | 0.9520 | 846.6 | 773.4 | 1871.2 | 0 |

On the 10 feedback queries, mean paired hybrid-with-feedback minus hybrid differences are +0.3112 nDCG@5 and +0.0600 Precision@5. These are observed differences under mixed judgments, without a significance claim.

Same query, playlist and hard exclusions; hybrid without versus with supplied soft feedback. This is a controlled feedback ablation, not a longitudinal user study.

Feedback outcomes by query: {"ndcg@5": {"improved": 7, "tied": 3}, "precision@5": {"improved": 2, "tied": 8}}. The independent standard-library calculation agrees with method averages and paired feedback deltas (maximum method-average difference 1.1e-16).

| Category | BM25 nDCG@5 | Semantic nDCG@5 | Hybrid nDCG@5 | Hybrid + feedback nDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| activity | 0.8486 | 0.3870 | 0.8430 | 0.8430 |
| artist_similarity | 0.9115 | 0.6859 | 0.9167 | 0.9167 |
| exclusion | 0.9903 | 0.8866 | 0.9869 | 0.9869 |
| genre | 0.9577 | 0.5453 | 0.9640 | 0.9640 |
| mixed_feedback | 0.6266 | 0.6214 | 0.6266 | 0.9378 |
| mood | 0.7709 | 0.2174 | 0.7513 | 0.7513 |
| playlist_context | 0.9702 | 0.8394 | 0.9702 | 0.9702 |

Queries without feedback whose hybrid ranking changed: 0. Overall BM25 and hybrid scores are very close; these results do not establish that hybrid without feedback beats BM25. The small difference between hybrid mean latencies does not establish a speed improvement.

Returned excluded tracks: 0. Timing scope: Warm conversation handling, constraint extraction, retrieval, feedback, explanations and payload construction; excludes schema preparation, model/index initialization, transport and browser rendering.

nDCG@5 uses gains 2^grade − 1 and log2(rank + 1) discounts against the same complete query pool. Precision@5 counts grades 1 and 2 and divides by five, including short lists. Results are macro-averaged over queries. This run contains 600 timing samples.

## Protocol and limits

Grades follow the existing rubric: 0 = unsupported/unrelated or a hard-constraint violation; 1 = partial match; 2 = strong support for the full request. Metadata-only judgments are not listening tests. Genre, mood and seed requirements apply together; retrieval scores and system identity are not relevance evidence.

A zero due to missing metadata is not proof that the song sounds irrelevant. Folksonomic tags can be sparse, contradictory or incidental. Metadata-based assessment may favor lexical retrieval, and these results do not establish general musical quality.

Query coverage: {"activity": 5, "artist_similarity": 5, "exclusion": 10, "genre": 10, "mixed_feedback": 10, "mood": 5, "playlist_context": 5}. The combined set covers all proposal categories, although the partners' category distributions differ.

The actual semantic encoder is Qwen/Qwen3-Embedding-0.6B, rather than the proposal's Sentence-BERT. This difference must be disclosed. The benchmark evaluates the standard application route; the separate live Ollama smoke check is integration evidence, not an additional retrieval-quality evaluation.

The approved proposal called for manual judgments of pooled candidates. A human-reviewed sample with remaining AI judgments is a methodological deviation. Report it explicitly and obtain course acceptance or finish full manual assessment before presenting this as the proposed fully manual study.

No system was retuned using these human reviews. Historical review effects are recorded separately in report.json and are not pooled across different implementations. Source hashes, review imports, all grade overrides and normalized encoding repairs are recorded there.
