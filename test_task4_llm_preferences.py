"""Live test for UiS-hosted Llama. Requires OLLAMA_API_KEY in the environment."""

import json
import time

from musiccrs.llm_preference_parser import (
    LLMPreferenceParser,
    PreferenceExtractionError,
)

QUERY = (
    "Recommend energetic rock songs similar to my playlist, "
    "but exclude Linkin Park and suggest something like Green Day."
)


def main():
    parser = LLMPreferenceParser()
    start = time.perf_counter()
    parsed = parser.parse(QUERY)
    elapsed = time.perf_counter() - start
    print("\n--- LLM STRUCTURED MUSIC PREFERENCES ---")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
    print(f"\nLLM extraction time: {elapsed:.2f} seconds")
    # These are critical test constraints, not assumptions about the model.
    assert "Linkin Park" in parsed["excluded_artists"], "Artist exclusion missed"
    assert "Green Day" in parsed["preferred_artists"], "Preferred artist missed"
    assert parsed["use_playlist_context"] is True, "Playlist context missed"
    print("\nTask 4 LLM preference extraction check passed!")


if __name__ == "__main__":
    try:
        main()
    except PreferenceExtractionError as exc:
        raise SystemExit(f"Could not complete LLM test: {exc}")

