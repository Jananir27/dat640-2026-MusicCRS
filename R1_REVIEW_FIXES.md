# R1 exclusion review and encoder decision

## Exclusion and routing fixes

The review identified inconsistent legacy paths in addition to cases already
handled by the production R2 adapter. The standard route, production LLM route
and candidate-only legacy LLM route now share the catalog resolver in
`musiccrs/request_constraints.py`.

- Both `Recommend metal without Metallica` and `Recommend metal, no Metallica`
  enforce the same artist ban. Configured advanced recommendations run with an
  empty playlist, including providers exposing only the original interface.
  Bare `/recommend` retains its existing non-LLM route.
- A literal negative clause takes precedence over the LLM's extraction. For
  `exclude Numb by Linkin Park`, both `Numb` and `Numb by Linkin Park` in the LLM
  field resolve to Linkin Park track IDs. Even an incorrect `Numb by 21 Savage`
  cannot override the explicit user qualifier. Matching remasters are included;
  21 Savage's recording remains eligible unless separately forbidden.
- A genuinely ambiguous bare title asks for its artist. Resolution completes
  before persistent preferences change, so compound invalid exclusions are atomic.
- Artist filtering compares normalized identities, including individual list
  credits and explicit `feat.`, `ft.` and `featuring` credits. Banning War does
  not ban The War on Drugs or Warpaint. Literal band names containing commas,
  `&` or `with` are preserved rather than guessed to be collaborations.
- The legacy fallback now uses resolved track IDs instead of title-only matches.
  Its candidate snapshot may omit an excluded recording; that absence does not
  justify excluding a different artist's same-title recording. The full catalog
  path remains strict about unknown or ambiguous references.
- When the LLM extracts no positive preferences, the fallback retrieval query
  uses the positive request with the negative clause removed. Excluded identities
  are not used as positive retrieval seeds.

The LLM schema remains compatible: `excluded_tracks` is still a list of strings.
Its extraction instructions now explicitly preserve `title by artist`; identity
enforcement does not depend on the model following that instruction perfectly.

## Regression checks

`tests/test_r1_exclusion_regressions.py` covers both service interfaces, both
exclusion wordings, seeded/empty playlists, actual MusicCRS agent wiring,
qualified/ambiguous titles, wrong or omitted LLM qualifiers, remasters, compound
artist credits and incompatible encoder configuration.

The full Python suite passes **212 tests**. The existing offline integrated
smoke suite passes **18 scenarios over 74 turns**. Those offline scenarios use
synthetic dense rankings and are not musical-relevance measurements.

The real catalog passes **11 additional cases**: both Metallica wordings with
seeded/empty playlists in both routes, plus three variants of the parsed Numb
exclusion. Three live Ollama calls also pass extraction and deterministic
catalog-identity checks. These live extraction checks do not measure retrieval
quality or response latency.

All **200 frozen R4 rankings and ranking scores replay unchanged**, with maximum
score difference **0.0**. No relevance grades or timing samples were changed.
The consolidated evidence is `evaluation/part4/r1_review_validation.json`.

```bash
MUSICCRS_NLU_LOCAL_ONLY=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv/bin/python -m unittest tests.test_r1_exclusion_regressions

MUSICCRS_NLU_LOCAL_ONLY=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

## Confirmed team model decision

The approved proposal names **Sentence-BERT**. The implementation uses
**Qwen/Qwen3-Embedding-0.6B**, configured in `retrieval/dense.py`, with the
TalkPlay field `metadata-qwen3_embedding_0.6b`. The stored vectors have 1,024
dimensions. Loading Qwen through the `SentenceTransformers` library does not
make the encoder Sentence-BERT.

On 2026-11-03, Tobi confirmed after discussing this with Janani that the team had
moved to Qwen. This records the team's revised encoder choice and closes review
item 4. Reports should identify the semantic encoder as Qwen, retaining
Sentence-BERT only when describing the original proposal.

The implementation retains matching Qwen query and track embeddings. Its
configuration guard prevents the default cached Qwen field from being paired
with a different query encoder. No encoder replacement or new R4 run is needed
for this clarification: the recorded evaluation already uses Qwen.
