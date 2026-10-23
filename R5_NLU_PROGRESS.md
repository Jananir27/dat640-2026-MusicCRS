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

The first browser rerun after that change used “what other songs are in this album?” and “what other songs are in the issa album”; both remained unhandled. The album intent examples had emphasized “on” rather than “in.” Training examples now include “in this album,” “in the album,” and “from this album” phrasings. This wording update still needs a browser rerun.

Additional manual feedback recommends natural pronoun-only follow-ups and stronger `other` examples for unsupported catalog questions. Training examples now cover short “who sings it?”, “what year?”, and “anything else by them?” forms, plus duration, explicitness, playback controls, awards, and musician-role questions as unsupported intents. When R5 recognizes a supported question but no entity is named, it now uses the last discussed track. This update has not been tested yet.

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

## Laptop browser check and context-aware phrasing fix (2026-10-12)

The browser backend was later started on the laptop at `127.0.0.1:5000`; the frontend initially pointed to port `5050` and showed `ERR_CONNECTION_REFUSED`. Restarting the frontend with `REACT_APP_BACKEND_URL=http://127.0.0.1:5000` restored the connection.

With `Numb — 21 Savage` in the conversation, the browser returned useful answers for “what other tracks are on this album?” (tracks from *Issa Album*), “Who performs it?”, “What kind of music is it?”, “What year?”, and “And what album is it on?”. An explicit “Other songs by 21 Savage?” also listed catalog tracks.

Three phrasings exposed ambiguity that the intent model alone did not resolve: “When did it drop?” matched tracks titled *Drop*; “Anything else by them?” matched an album alias and asked which album; and “Who made Numb?” was unhandled. Do not add `drop` or `them` to a global stop-word list: either can be a genuine catalog name. The current code canonicalizes these specific constructions before entity lookup: a release question ending in “did … drop” becomes a release-date question, “by/from them” uses the remembered artist, and “who made …” becomes “who performs …” because the catalog provides performer metadata rather than songwriter/producer credits. Explicit title-position queries such as “What album is Drop on?” and “What album is Them on?” are left unchanged.

A targeted unit test now checks these rewrites and confirms title-position names remain intact; it passes. I also ran the cached MiniLM classifier against a small fixture catalog in local-only mode. It answered all three previously failing phrasings and preserved exact title lookups for *Drop* and *Them*. The repository `.venv` lacks RapidFuzz, so that classifier check stubbed only the optional fuzzy-search call; exact matching and the intent classifier were real. The laptop backend must be restarted to load this code before repeating the browser checks.

### Follow-up browser rerun

The next browser rerun used the laptop backend at `127.0.0.1:5000` with *Numb — 21 Savage* in the playlist. “Who made Numb?” now answers “Numb is by 21 Savage.” “What album is Drop on?” asks the user to disambiguate catalog tracks named *Drop*; this is correct because *Drop* can be a track title. “When did it drop?” was unhandled, and “What album is Them on?” fuzzy-matched *Theme* and answered with that track’s album. This shows that preserving the word is not enough when fuzzy matching substitutes a different title.

I added a direct remembered-track release follow-up path for “When did it drop?” plus a unit test asserting it returns the previous track’s date; that targeted test passes. After the user reported restarting, the browser still returned the unhandled response. The listener on `127.0.0.1:5000` was still PID `50164`, unchanged from the prior process check, so the updated code is not yet confirmed as running. Tighten title fuzzy matching so a near-match like *Them* → *Theme* is clarified or rejected rather than answered as if it were exact.

The user restarted the backend again. The port-5000 listener changed to PID `51255`, confirming a process restart. A direct Socket.IO polling request with the browser origin and referer received HTTP 200 and `Access-Control-Allow-Origin: http://127.0.0.1:3000`. The browser still returned 403, and its response headers identified `server: AirTunes/940.23.1`; the request was reaching macOS AirTunes on port 5000 rather than the Flask service. The backend now accepts `MUSICCRS_HOST` and `MUSICCRS_PORT` environment variables when started with `python -m musiccrs.musiccrs`, so use a free port such as 5051 and point the frontend at that same port. This browser failure is a port collision/routing issue, not an R5 response.

After moving the backend to port 5051, the browser answered “When did it drop?” with “Numb was released on 2017-09-23.” The next “Anything else by them?” request crashed in `CatalogEntities.mentions()` because the active Python 3.14 virtual environment lacked `rapidfuzz`. Remembered-artist follow-ups now use `last_question_artist` without fuzzy matching, and fuzzy lookup degrades gracefully if RapidFuzz is absent. On the next restart, “Anything else by them?” correctly listed tracks by 21 Savage. A further browser check found that “What album is Them on?” still fuzzy-matched *Theme* and returned its album. I tightened fuzzy matching for short title spans and prevented unmatched explicit title words from falling back to the previously discussed track. Three focused regression tests pass; restart the backend once more to verify this final title safeguard in the browser. Installing `rapidfuzz` from `requirements.txt` remains necessary for typo-tolerant entity lookup.

### Browser rerun after title-matching safeguard (2026-10-12)

Confirmed the local backend had restarted, then tested in the browser with *Numb — 21 Savage* in the playlist:

- “What album is Them on?” now asks “Which song or track should I look up?” instead of incorrectly matching *Theme*.
- “When did it drop?” correctly returns “Numb was released on 2017-09-23.”
- “What album is Drop on?” presents the matching *Drop* tracks for disambiguation, preserving the intended behavior for real track titles.
- “Anything else by them?” correctly lists tracks by 21 Savage.
- “Who performs Numb by 21 Savage?” correctly returns “Numb is by 21 Savage.”

The short-name safeguard avoids the false *Them* → *Theme* answer, but this run did not establish that *Them* is an exact catalog title; the bot appropriately requests clarification. The playlist still contains *Numb — 21 Savage* after these checks.

### Independent browser rerun (2026-10-12)

Repeated the checks in the live browser after finding the chat had reset to an empty conversation. Added *Numb — 21 Savage* again, then observed: “What album is Them on?” → “Which song or track should I look up?”; “When did it drop?” → “Numb was released on 2017-09-23.”; “What album is Drop on?” → a four-track disambiguation list; “Anything else by them?” → a list of 21 Savage tracks; and “Who performs Numb by 21 Savage?” → “Numb is by 21 Savage.” The live responses matched the prior report.

### RapidFuzz and latency verification (2026-10-12)

Confirmed the running backend uses the `dat640` Conda interpreter (`/opt/homebrew/Caskroom/miniconda/base/envs/dat640/bin/python`) and RapidFuzz 3.14.6 is installed there. Also installed RapidFuzz 3.14.6 in the repository `.venv` for local checks. A focused `CatalogEntities` check with RapidFuzz matched the misspelled title *Numbb* to *Numb* and artist *Linkin Prk* to *Linkin Park*.

Live browser results show partial end-to-end typo handling: “What album is Numb by Linkin Prk on?” correctly returned the Meteora album, while “What album is Numbb by Linkin Park on?” and “Other songs by Linkin Prk?” returned the generic “I'm sorry, I don't understand that request.” The matching library works, but typo recognition is not reliable across intent routes; R5 still needs a fallback that can identify the supported intent before fuzzy entity resolution when the initial classifier rejects a typo-bearing query.

Measured three warmed browser replies from send to visible answer: exact album lookup 211 ms, album lookup with misspelled artist 228 ms, and artist lookup 161 ms. These observed responses are comfortably below the 3–5 second target; this small sample is not a full latency benchmark.
