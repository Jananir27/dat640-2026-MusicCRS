"""Generate BM25 + tags predictions for a dataset split.

Uses extended track metadata:
- track_name
- artist_name
- album_name
- tag_list

Supports limiting the number of sessions for development/testing so that
BM25 + tags can be compared fairly with dense and hybrid retrieval on
the same subset.
"""

import argparse
import json

from datasets import load_dataset

from .bm25 import BM25Retriever
from .data_loader import MusicCatalogLoader


NUM_TURNS = 8

# Metadata fields used by our improved BM25 experiment.
CORPUS_TYPES = [
    "track_name",
    "artist_name",
    "album_name",
    "tag_list",
]


def _build_retrieval_input(
    conversations: list[dict],
    target_turn_number: int,
    catalog: MusicCatalogLoader,
    corpus_types: list[str],
) -> str:
    """Build retrieval query from conversation history.

    Includes all messages up to and including the current user's
    message. Previous music recommendations are expanded to their
    track metadata.
    """

    lines = []

    for message in conversations:

        # Ignore messages belonging to future turns.
        if message["turn_number"] > target_turn_number:
            break

        # For the current turn, stop after the user's message.
        if (
            message["turn_number"] == target_turn_number
            and message["role"] != "user"
        ):
            break

        role = message["role"]
        content = message["content"]

        # Convert previous music track IDs into readable metadata.
        if role == "music":
            role = "assistant"

            content = catalog.id_to_metadata_str(
                content,
                corpus_types,
            )

        lines.append(
            f"{role}: {content}"
        )

    return "\n".join(lines)


def run_baseline(
    retriever: BM25Retriever,
    catalog: MusicCatalogLoader,
    dataset_name: str,
    split: str,
    topk: int,
    corpus_types: list[str],
    max_sessions: int | None = None,
) -> list[dict]:
    """Run BM25 + tags retrieval.

    Args:
        retriever:
            BM25 retriever.

        catalog:
            Track metadata catalog.

        dataset_name:
            Hugging Face dialogue dataset.

        split:
            Dataset split, for example "test".

        topk:
            Number of tracks to retrieve.

        corpus_types:
            Metadata fields used for retrieval.

        max_sessions:
            Optional number of sessions to process.
            If None, the complete dataset split is used.

    Returns:
        Predictions in MusicCRS evaluation format.
    """

    print()
    print("Loading dialogue dataset...")

    dataset = load_dataset(
        dataset_name,
        split=split,
    )

    # ---------------------------------------------------------
    # Optional subset
    # ---------------------------------------------------------

    if max_sessions is not None:

        number_of_sessions = min(
            max_sessions,
            len(dataset),
        )

        dataset = dataset.select(
            range(number_of_sessions)
        )

    print(
        f"Using {len(dataset)} sessions"
    )

    print(
        f"Expected predictions: "
        f"{len(dataset) * NUM_TURNS}"
    )

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

    predictions = []

    for session_index, item in enumerate(dataset):

        for turn_number in range(
            1,
            NUM_TURNS + 1,
        ):

            query = _build_retrieval_input(
                item["conversations"],
                turn_number,
                catalog,
                corpus_types,
            )

            predicted_track_ids = (
                retriever.text_to_item_retrieval(
                    query,
                    topk,
                )
            )

            predictions.append(
                {
                    "session_id":
                        item["session_id"],

                    "turn_number":
                        turn_number,

                    "predicted_track_ids":
                        predicted_track_ids,

                    "predicted_response":
                        "",
                }
            )

        # Progress indicator
        if (
            (session_index + 1) % 100 == 0
            or session_index + 1 == len(dataset)
        ):
            print(
                f"Processed "
                f"{session_index + 1}/"
                f"{len(dataset)} sessions"
            )

    return predictions


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Run BM25 retrieval with "
            "track name + artist + album + tags"
        )
    )

    parser.add_argument(
        "--dialogue_dataset",
        default=(
            "talkpl-ai/"
            "TalkPlayData-Challenge-Dataset"
        ),
        help="Hugging Face dialogue dataset",
    )

    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Number of tracks to retrieve",
    )

    # ---------------------------------------------------------
    # NEW: allows us to evaluate only first N sessions
    # ---------------------------------------------------------

    parser.add_argument(
        "--max_sessions",
        type=int,
        default=None,
        help=(
            "Limit the number of sessions. "
            "Example: --max_sessions 100. "
            "If omitted, all sessions are processed."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to write predictions JSON",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Experiment information
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("MusicCRS - BM25 + Tags Retrieval")
    print("=" * 60)

    print(
        f"Split       : {args.split}"
    )

    print(
        f"Top-k       : {args.topk}"
    )

    if args.max_sessions is None:
        print(
            "Sessions    : ALL"
        )
    else:
        print(
            f"Sessions    : "
            f"{args.max_sessions}"
        )

    print(
        "Metadata     : "
        "track_name + artist_name + "
        "album_name + tag_list"
    )

    print(
        f"Output      : {args.output}"
    )

    # ---------------------------------------------------------
    # Initialize BM25 + tags
    # ---------------------------------------------------------

    print()
    print(
        "Initializing BM25 + tags retriever..."
    )

    retriever = BM25Retriever(
        corpus_types=CORPUS_TYPES,
    )

    # ---------------------------------------------------------
    # Track metadata catalog
    # ---------------------------------------------------------

    print(
        "Loading music catalog..."
    )

    catalog = MusicCatalogLoader()

    # ---------------------------------------------------------
    # Generate predictions
    # ---------------------------------------------------------

    predictions = run_baseline(
        retriever=retriever,
        catalog=catalog,
        dataset_name=args.dialogue_dataset,
        split=args.split,
        topk=args.topk,
        corpus_types=CORPUS_TYPES,
        max_sessions=args.max_sessions,
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    print()
    print(
        "Saving predictions..."
    )

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            predictions,
            f,
            indent=2,
        )

    print()
    print("=" * 60)
    print(
        "BM25 + tags retrieval completed"
    )
    print("=" * 60)

    print(
        f"Predictions : "
        f"{len(predictions)}"
    )

    print(
        f"Saved to    : "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
