# R5 NLU progress

## Why we changed the R5 router

R5 originally relied on fixed regular expressions in `musiccrs/music_intelligence.py` to recognize questions about a track and to list songs by an artist. Small wording changes such as “other songs by Linkin Park” were not consistently recognized. The goal is to handle varied natural phrasing within the 3–5 second response budget, without sending each message to a hosted generative model.

## Current implementation

R5 now uses a local CPU sentence encoder (`sentence-transformers/all-MiniLM-L6-v2`) with a lightweight logistic-regression intent classifier. The classifier is fitted from the examples in `musiccrs/nlu_data/r5_train.json` when the service starts. It recognizes artist, album, release-date, tag, and artist-track-listing questions, with an `other` class for unrelated messages.

The router masks catalog names before classifying the intent. A shared catalog index finds track and artist names; RapidFuzz is used for a second, typo-tolerant lookup when an R5 request has been recognized but an exact name was not found. Answers are still read from catalog metadata. R5 remembers the last discussed track and artist for follow-ups such as “when did that come out?” Duplicate titles are clarified rather than guessed. Recommendation requests continue to use the final G1 retriever; playlist mutations continue through their existing handlers.

`MusicService.warm_r5()` loads the router and entity index at backend startup so the first browser question does not pay the model-loading cost. New dependencies are `scikit-learn` and `rapidfuzz`. The MiniLM model is cached under `cache/nlu/models`.

## Manual CPU-only check

The R5 command-line check instantiates `MusicService` and `MusicIntelligence` directly. R5 answers do not call the Qwen retriever, so this check can run in a Slurm CPU allocation before starting the GPU browser backend. The first run downloads MiniLM; the next runs should use the cache. Run it from the project root with `.venv` active:

```bash
python - <<'PY'
from musiccrs.music_service import MusicService, ConversationState
from musiccrs.music_intelligence import MusicIntelligence

service = MusicService()
chat = MusicIntelligence(service)
state = ConversationState()

for phrase in [
    "Who made Numb by Linkin Park?",
    "What album is Numb by Linkin Park on?",
    "What year did Numb come out?",
    "How is Numb tagged?",
    "anything else from Linkin Park?",
    "when did that come out?",
]:
    result = chat.handle(phrase, state)
    print(f"\nYou: {phrase}\nMusicCRS: {result[0] if result else 'UNHANDLED'}")
PY
```

## Results so far

The first manual CPU run loaded the model and produced these outcomes:

- “Who made Numb?” was incorrectly resolved to a catalog title called “Made.”
- “What album is Numb by Linkin Park on?” correctly returned Meteora.
- “What year did Numb come out?” asked for disambiguation because the catalog has several tracks named Numb.
- “How is Numb tagged?” asked for disambiguation while that title choice was pending.
- “Anything else from Linkin Park?” correctly listed other tracks.
- “When did that come out?” correctly used the previously discussed Linkin Park track.

We added “made” to the grammatical stop words and made a repeated title inherit the artist only when it matches the previous R5 track. These changes were pushed in commit `baccbec`.

The follow-up run changed the first question to “Who made Numb by Linkin Park?” but it returned `UNHANDLED`. We added training examples for questions containing both a track and artist; this was pushed in commit `8bf5cd1`. **That last fix has not been manually rerun yet.** The next R5 check should pull the latest `ft/core-fn-impl` branch and rerun the CPU-only command above.

The next browser conversation exposed a missing R5 case: “what other songs are on the album?” was treated as an artist-track request. The router now has an `album_tracks` intent and album-name lookup, including distinctive album-word aliases such as “Issa” for “Issa Album.” Adding a track also records it as the current conversational reference, so “other songs on the track” can use that track's album. This album-listing change still needs a manual rerun.

The prior test downloaded about 85 MB of model weights. It did not report per-message latency, so the 3–5 second target still needs to be checked in the browser after the router is warm.

## Browser backend restart

The browser backend job previously used job ID `485443`, running the old code on `gorina9` at port `25443`. To try the pushed R5 changes in the browser, the service must be restarted from `gorina11` using the updated checkout:

```bash
scancel 485443
cd /mnt/beegfs/home/otogun/dat640-2026-MusicCRS
git pull --ff-only origin ft/core-fn-impl
mkdir -p logs
sbatch run_backend.sbatch cache/g1-485101/final_retrieval.json
```

Use the newly printed job ID with `squeue -u otogun` and inspect `logs/backend-JOBID.out` for the assigned node and port. Recreate the `gorina11 → ssh1` reverse tunnel using those new values. Keep the Mac’s local tunnel to port `5050` open and open the frontend at `http://localhost:3000/build`.

The updated code has been committed and pushed to `origin/ft/core-fn-impl` through `8bf5cd1`. The latest known instruction was to restart the browser backend; no confirmation of that restart has been received yet.
