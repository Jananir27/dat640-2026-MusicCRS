"""Backward-compatible entry point for the full R1–R8 backend smoke checks.

Run ``python -m tests.live_music_smoke`` from the repository root.
See ``retrieval/README.md#musiccrs-verification`` for the browser checklist and offline mode.
"""

from tests.full_music_smoke import main


if __name__ == "__main__":
    main()
