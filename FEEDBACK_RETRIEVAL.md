# Isolated feedback-aware retrieval

The new implementation lives entirely in
[`musiccrs/feedback_retrieval.py`](musiccrs/feedback_retrieval.py).
It is opt-in: no existing source file, UI handler, benchmark, label, or application
startup path has been changed. The existing chat still uses its original path.

## What changes in this experiment

The existing Rocchio module adjusts the query and reranks an already retrieved
pool. The new adapter uses that adjusted query to search **every usable catalog
embedding**, before selecting semantic candidates:

```text
Original text ──────────────────────────────→ BM25 candidates
Original query vector + likes − dislikes ───→ Full embedding search
                                                    ↓
                       Filter constraints → fuse candidate branches → rerank
```

Full embedding search scores every usable row, applies eligibility constraints,
and selects the allowed top candidates. BM25 expands its fetch depth when
exclusions consume leading results. Both branches therefore enforce constraints
before truncation; the final output is checked again.

The adapter reuses `FeedbackState`, `RankedTrack`, and `RocchioFeedback` from the
existing module. It reads the existing dense matrix without changing it; raw
injected matrices are normalized in a separate array. Invalid/missing rows are
ignored. Query dimensions and catalog/embedding identities are validated.
All vectors must use the same embedding model and space.

## Interface

```python
from musiccrs.feedback_retrieval import FeedbackAwareRetriever

# Reuse already loaded catalog, BM25, and DenseRetriever instances.
retriever = FeedbackAwareRetriever(catalog, lexical, dense)

result = retriever.retrieve(
    "rock music",                    # original positive request, not a negative prompt
    state.feedback,                  # caller owns this state across conversation turns
    topk=5,
    excluded_ids=set(state.playlist_ids) | state.skip_recommendation_ids,
    # allowed_ids=genre_and_playlist_eligible_ids,  # optional structured R1 constraints
)

for candidate in result.tracks:       # existing RankedTrack interface
    print(candidate.track_id, candidate.score, candidate.metadata)
    print(candidate.evidence["new_to_original_pool"])

print(result.new_candidate_ids)       # absent from BOTH original retrieval branches
print(result.unavailable_feedback_ids)
```

The module does not parse conversation language or infer genre/playlist hard
constraints. Pass the cleaned original request and structured constraints from
the caller. `allowed_ids=None` allows the whole catalog; an empty set permits
nothing. Persistent artist/track exclusions, dislikes, and explicitly passed
excluded IDs always apply. Artist exclusions include credited collaborators.
Liked tracks are withheld by default while their vectors remain feedback seeds.

`FeedbackRetrievalResult` contains ranked tracks, original/current candidate IDs,
new candidate IDs, whether retrieval used a changed query, and available/missing
feedback seed IDs. Each track includes branch ranks, cosine evidence, raw RRF,
normalized fusion prior, feedback delta, and final score where reranking applies.
Metadata comes directly from the catalog. No shared mutable user state is stored.

## Three controlled modes

| `mode` | Candidate retrieval | Final ranking |
| --- | --- | --- |
| `baseline` | Original BM25 + original semantic query | Fusion |
| `rerank_only` | Same original candidate pool | Fusion + existing Rocchio reranker |
| `feedback_retrieval` (default) | Original BM25 + feedback-adjusted semantic query | Fusion + same Rocchio reranker |

All modes use identical exclusions and weights. By default, each branch supplies
up to 50 eligible candidates; the combined pool can contain up to 100 distinct
tracks. `topk` above 50 increases the branch depth accordingly.

The defaults retain BM25 weight 1.0, semantic weight 0.25, RRF constant 60,
Rocchio alpha 1.0, beta 0.75, gamma 0.25, and reranking weight 0.75. Weights are
configurable at construction but were not tuned on the evaluation labels.
Fusion scores are divided by the maximum possible first-place RRF score, placing
both candidate sources on a common scale rather than appending new tracks last.
These scores are ranking evidence, not probabilities.

Reranking receives the **original** query vector; giving it the already adjusted
vector would apply the query update twice. Without usable vector feedback,
candidate retrieval stays anchored to the original query. Artist preferences can
still affect the existing reranker. A zero branch weight disables that branch.

This new comparison uses a fusion prior and an explicit union of branch candidates.
The previous 25-query benchmark used `1 - rank / pool_size` over its hybrid output.
Consequently, this is a controlled ablation within the new adapter, not a replay
of the old benchmark. Direct quality comparisons require a new run and pooling.

## Run separately

```bash
# Optional offline cache configuration for this repository's laptop cache:
export HF_HOME="$PWD/cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

.venv/bin/python -m musiccrs.feedback_retrieval \
  --query "rock music" \
  --like ffc14c3d-21b1-4938-b58e-28c21c4d39a3 \
  --dislike 625d9682-8214-4391-8689-fd14b87d01e8 \
  --exclude-artist "21 Savage" \
  --output /private/tmp/musiccrs-feedback-comparison.json
```

The CLI runs all three modes, warms the model before timing, uses four CPU threads
by default, and refuses to overwrite an existing output file. Output includes
metadata, ranking evidence, explanations, candidate sets, and observed times.
The original query is encoded once per request. Retrieval itself searches the
available embedding matrix and performs no additional query-model inference.

## Verification and observed limitation

```bash
.venv/bin/python -m unittest tests.test_feedback_retrieval \
  tests.test_relevance_feedback tests.test_part4_evaluation tests.test_final_retrieval
```

The new tests cover a previously unreachable candidate reaching first place,
positive and negative seed effects, exclusions/collaborations, overfetching,
structured allowlists, no double query update, invalid/missing vectors, zero
branch weights, and session/source-data isolation.

[`evaluation/part4/feedback_retrieval_demo.json`](evaluation/part4/feedback_retrieval_demo.json)
records a real cached-catalog check with the exact command's query and feedback.
The catalog contains 47,071 tracks; 46,579 have usable embeddings. The new search
introduced **32 candidates absent from the original pool**. The liked/disliked
seed IDs and excluded artist were absent from returned tracks in every mode.

In this sample, the top five from feedback-aware retrieval matched those from
reranking alone. The existing weights still favor the lexical branch strongly;
candidate expansion does not guarantee a top-five or relevance improvement.
No new nDCG/precision result is claimed. New candidates need pooled relevance
judgments before a quality comparison; missing labels must not be treated as 0.
