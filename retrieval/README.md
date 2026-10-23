# Retrieval

Text-to-item retrieval over the track metadata catalog used in the RecSys 2026
MusicCRS challenge, plus evaluation utilities for scoring predictions.
Adapted from [`music-crs-baselines`](https://github.com/nlp4musa/music-crs-baselines)
and [`music-crs-evaluator`](https://github.com/nlp4musa/music-crs-evaluator).

The chat backend uses `FinalRetriever` in `final.py`. See
[Final G1 retrieval in the chat backend](#final-g1-retrieval-in-the-chat-backend)
for setup, saved-artifact requirements, and server verification commands.
The BM25 examples below document the original baseline.

## Task

Given a conversation between a user and a music recommender system, your task is to
**rank the track catalog and return the top tracks the user is most likely looking
for at the current turn**. This is the retrieval component of the MusicCRS pipeline:
given the conversation so far, produce a ranked list of candidate track IDs from the
~47k-track catalog.

You are given a working BM25 baseline (below) as your starting point. Your task is
to build a retrieval method that **outperforms it** on the final score defined
under [Final score](#final-score).

### Input and output

  - **Input:** a conversation turn, consisting of the dialogue history up to and
  including the user's current message (see `run_bm25_baseline.py` for how the
  baseline builds its query from this history).
  - **Output:** a ranked list of up to 20 track IDs from the catalog
  (`talkpl-ai/TalkPlayData-Challenge-Track-Metadata`), most relevant first.

Implement your method as a subclass of `RetrievalModule` (see `base.py`):

```python
def text_to_item_retrieval(self, query: str, topk: int) -> list[str]: ...
def batch_text_to_item_retrieval(self, queries: list[str], topk: int) -> list[list[str]]: ...
```

This keeps your method a drop-in replacement for `BM25Retriever` everywhere it's
used (prediction generation, evaluation).

### Data available to you

  - Track metadata (name, artist, album, tags, popularity, release date)
  - Track and user embeddings, precomputed with multiple encoders (audio, image,
  collaborative filtering, and text) — see the top-level
  [README's Dataset section](../README.md#dataset) for the full list
  - The conversation dataset itself, including user profiles and conversation goals

You are free to use any combination of these. Some directions to consider:
dense/semantic retrieval using the precomputed embeddings, hybrid ranking
(combining BM25 with dense similarity), query expansion or rewriting,
personalization using user profiles, or reranking with a cross-encoder.

## Structure

```
retrieval/
├── base.py               # RetrievalModule abstract interface
├── data_loader.py        # MusicCatalogLoader — loads track metadata from Hugging Face
├── bm25.py                # BM25Retriever — BM25 baseline
├── run_bm25_baseline.py   # generates BM25 predictions over a dialogue dataset split
└── evaluation/
    ├── metrics.py             # nDCG
    ├── diversity.py           # item (catalog) diversity
    ├── make_ground_truth.py   # extract ground truth from the challenge dialogue dataset
    └── evaluate.py            # evaluate() + CLI runner — nDCG, diversity, and final_score
```

See the top-level [README's Dataset section](../README.md#dataset) for what
data is available in the RecSys 2026 MusicCRS challenge collection.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Retrieval

`BM25Retriever` builds (and caches to `./cache/bm25/<corpus_name>`) a BM25 index
over selected metadata fields for `talkpl-ai/TalkPlayData-Challenge-Track-Metadata`,
then ranks track IDs for a query.

```python
from retrieval import BM25Retriever

retriever = BM25Retriever()  # downloads + indexes on first run, cached after
retriever.text_to_item_retrieval("upbeat pop song", topk=5)
retriever.batch_text_to_item_retrieval(["upbeat pop song", "sad piano ballad"], topk=5)
```

## Evaluation

Your method will be evaluated on the `test` split (the devset):

```bash
python3 -m retrieval.run_bm25_baseline --split test --topk 20 --output predictions.json   # or your own runner
python3 -m retrieval.evaluation.evaluate --predictions predictions.json --ground_truth ground_truth.json --catalog_size 47071
```

### Ground truth

`talkpl-ai/TalkPlayData-Challenge-Dataset` has `train` (15,199 sessions) and
`test` (1,000 sessions, used as the devset) splits. Each session has 8 turns;
each turn embeds the gold track ID and gold response directly in its
`conversations` field. Extract these into the format `evaluate.py` expects:

```bash
python3 -m retrieval.evaluation.make_ground_truth --split test --output path/to/ground_truth.json
```

### Generating BM25 predictions

`run_bm25_baseline.py` retrieves top-k track IDs for every session x turn in a
dialogue dataset split. The query is the full conversation history up to and
including that turn's user message (prior `music` turns are expanded to
track metadata), matching the retrieval input built in
`music-crs-baselines/mcrs/crs_baseline.py`. There is no LLM query rewriting
or response generation — retrieval only, so `predicted_response` is left
empty:

```bash
python3 -m retrieval.run_bm25_baseline --split test --topk 20 --output path/to/predictions.json
```

### Scoring predictions

`evaluate()` computes macro-averaged nDCG@{1,10,20}, item (catalog)
diversity, and a combined **final score**, given predictions and ground
truth in the challenge format:

  - Predictions: list of `{session_id, turn_number, predicted_track_ids, predicted_response}`
  - Ground truth: list of `{session_id, turn_number, ground_truth_track_id}`

```bash
python3 -m retrieval.evaluation.evaluate \
  --predictions path/to/predictions.json \
  --ground_truth path/to/ground_truth.json \
  --catalog_size 47071 \
  --output path/to/results.json
```

Or from Python:

```python
from retrieval.evaluation import compute_ndcg_metrics, compute_catalog_diversity
```

Use this (or `retrieval.evaluation.evaluate()` from Python) to compute your score
yourself during development — this is the exact function used for grading.

#### Final score

Your ranking is scored on a single combined metric:

```
final_score = 0.8 * nDCG@20 + 0.2 * catalog_diversity
```

  - **nDCG@20** — how highly you rank the ground truth track within your top 20
  predictions, averaged over all turns. This is the primary signal: getting the
  right track ranked highly matters most.
  - **catalog_diversity** — the fraction of the track catalog covered by your
  recommendations across all turns (unique recommended tracks / catalog size).
  This penalizes methods that recommend the same small set of popular tracks to
  everyone, regardless of the query.

### Baseline to beat (test split, 1,000 sessions x 8 turns)

| Metric | BM25 baseline |
| --- | ---: |
| nDCG@1 | 0.0101 |
| nDCG@10 | 0.0644 |
| nDCG@20 | 0.0828 |
| Catalog diversity | 0.3906 |
| **Final score** | **0.1444** |

nDCG is in line with the `music-crs-evaluator` baseline's LLaMA-1B+BM25
numbers (nDCG@10 = 0.0627, nDCG@20 = 0.0815); the small gap is expected
since that baseline additionally uses an LLM for response generation
while ours is retrieval only.

## Notes

  - You do not need to generate a text response for this task — leave
  `predicted_response` empty in your predictions; it isn't scored.
  - Ground truth for the `test` split can be extracted yourself with
  `retrieval.evaluation.make_ground_truth` for local development. Do not
  use it to fit your method (e.g. hard-coding gold tracks); it's for
  evaluation only.

## Final G1 retrieval in the chat backend

`MusicService` uses `FinalRetriever` for R6 and R8. There is no automatic
BM25-only fallback. R5 still answers catalog metadata questions, and R7 adds
tracks from the saved recommendation list without rerunning retrieval.

### Ranking stages

The adapter reuses the original G1 functions in this order:

1. `build_adaptive_query()` and `build_goal_aware_query()` construct queries.
2. BM25 plus Qwen3 dense retrieval, fused with weighted RRF, produce each
   branch's top 20. The original scripts use 50 candidates per component.
3. `weighted_rrf()` fuses adaptive and goal-aware rankings with the saved
   specificity-dependent weights.
4. `rerank_tracks()` applies the saved exact-metadata weights.
5. `rerank_prediction()` applies the saved initial diversity settings.
6. A separate adaptive hybrid retrieval uses 100 candidates per component
   to produce the top-100 reservoir.
7. `selective_rerank()` expands the lower positions using the saved settings.

The report's final selective configuration is `protect_top=12`,
`reservoir_start=11`, `reservoir_end=100`, `novelty_weight=1.35`, and
`original_bonus=0.14`. Commit `0e28727` records final score `0.223511684`.
The runtime loads the actual saved winning settings rather than retuning them.

The original top-20 branch is computed separately from the top-100 reservoir.
Taking the first 20 of a fusion with a larger candidate pool can change its
ranking and does not reproduce the original pipeline.

### Export the saved artifacts on the school server

If the original files cannot be recovered, regenerate them with the updated
checkout on a compute/GPU node, in the project's activated Python environment:

```bash
bash run_g1_artifacts.sh cache/g1-rerun
export MUSICCRS_RETRIEVAL_ARTIFACT="$PWD/cache/g1-rerun/final_retrieval.json"
python -m tests.full_music_smoke --quiet
```

The runner requires a new output directory, saves `pipeline.log` there, and
stops on any failing stage. It runs both hybrid branches, fusion tuning,
exact-match tuning, initial diversity tuning, the top-100 reservoir, and
the final selective fine search. It then evaluates the predictions and
exports the verified runtime artifact. This is a full G1 regeneration, not
a smoke test; downloads, embedding inference, and CPU tuning can take time.
Keep it in your school's compute job/session for the duration of the run.
If interrupted, completed files remain available; the runner itself does not
automatically resume stages. The goal-aware script supports chunk resumption.

The fine search stops at the first configuration meeting its score target,
or saves the best improvement if the search finishes below the target.
A rerun can choose different weights or produce different metrics; inspect
`results_selective_expansion_fine.json` and `results_final.json` rather than
assuming the report's exact score has been reproduced. The earlier coarse
candidate-expansion search is not required as input to the final fine search.

Use the updated checkout and its activated Python environment. Locate the
directory containing these G1 files:

```text
results_prediction_fusion.json
  (or prediction_fusion_tuning_results.json)
results_exact_specificity_tuning.json
results_diversity_tuning.json
results_selective_expansion_fine.json
predictions_exact_specificity_best.json
predictions_diversity_best.json
predictions_adaptive_goal_top100_full.json
predictions_selective_expansion_fine_best.json
```

Then, from the project root:

```bash
python -m retrieval.export_final --artifacts-dir /absolute/path/to/G1/results
```

This writes `cache/final_retrieval.json`. The exporter checks matching
session/turn keys and unique rankings, derives the two sets of recommendation
frequency statistics, and replays both diversity stages. Every result must
match the saved predictions exactly before it writes the artifact. Source
file hashes are stored with the configuration. No model or ground truth is
used for this export.

If the final fine-search predictions were renamed to `predictions.json`, use
`--final-predictions predictions.json`. If both fusion result names exist,
select the correct one with `--fusion-results results_prediction_fusion.json`.
Relative filenames are resolved within `--artifacts-dir`.

Hybrid settings default to those in `run_final.sh`: candidate depth 50,
dense weight 0.25, RRF constant 60, fusion RRF constant 60. If the winning run
used different values, supply `--candidate-k`, `--dense-weight`, `--rrf-k`,
and `--fusion-rrf-k` explicitly. The export verifies the saved diversity
stages; it does not prove that these upstream settings regenerate the saved
hybrid rankings.

`run_final.sh` currently ends with the earlier coarse candidate-expansion
search. Running it alone does not reproduce the report's final fine search
in `tune_selective_expansion.py`. Recover the existing winning artifacts
before starting any expensive rerun.

If an artifact is kept elsewhere, configure both the app and smoke test:

```bash
export MUSICCRS_RETRIEVAL_ARTIFACT=/absolute/path/to/final_retrieval.json
```

Do not substitute fixture settings or empty frequency counts for missing
G1 artifacts. Missing or malformed settings stop initialization explicitly.

### Run the actual pipeline

```bash
python -c 'import torch; print("CUDA available:", torch.cuda.is_available())'
python -m tests.full_music_smoke --quiet
python -m musiccrs.musiccrs
```

The real smoke test loads the saved settings, real catalog, precomputed track
embeddings, and Qwen query encoder. CUDA accelerates query encoding when
available; the current dense implementation ranks NumPy vectors on CPU.
Model/catalog setup time is reported separately from dialogue timings.
The chat server initializes retrieval before accepting connections.

### Live-conversation adaptation and limits

- Chat does not have the dataset's annotated specificity or user profiles.
  R6/R8 use the LL discovery branch. R6 supplies the actual request, a goal
  derived from playlist titles/artists/tags, the two latest successful
  retrieval requests, and metadata for two recent playlist tracks.
- R8 starts fresh context using the new description. It replaces the playlist
  only after candidates are available. Clearing the playlist clears recent
  retrieval requests as well. Context remains local to the connected user.
- Novelty uses frozen G1 recommendation frequencies, not catalog popularity
  or mutable counts shared between live users.
- The final result contains at most 20 tracks. Playlist exclusions are applied
  after ranking, so R6 may return fewer than five if too few eligible tracks
  remain. R8 chooses a duration-based playlist from these results, now at
  most 20 songs. It does not append unreranked reservoir tracks.
- Protecting the first 12 means final selective expansion normally cannot
  change R6's first five suggestions. Exclusions can expose lower positions.
  The protected-prefix policy has not been retuned for a five-song display.
- The reported G1 score belongs to the saved development evaluation. It is
  not a measured score for new chat requests or proof of recommendation quality.

See [MusicCRS verification](#musiccrs-verification) below for offline and browser checks.

## MusicCRS verification

Run from the project root in the activated Python environment:

```bash
python -m unittest tests.test_final_retrieval tests.test_music_intelligence tests.test_bm25_cache
python -m tests.full_music_smoke --offline --quiet
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
[the final retrieval setup](#final-g1-retrieval-in-the-chat-backend), then run:

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
