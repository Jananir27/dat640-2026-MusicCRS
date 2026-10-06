"""Generate BM25 + tags predictions for the MusicCRS dataset.

This runner extends the original BM25 baseline by indexing and retrieving
over the following track metadata fields:

    - track_name
    - artist_name
    - album_name
    - tag_list

The resulting predictions are compatible with
retrieval.evaluation.evaluate.
"""

import argparse
import json

from datasets import load_dataset

from .bm25 import BM25Retriever
from .data_loader import MusicCatalogLoader


NUM_TURNS = 8


# Metadata fields used by the improved BM25 retriever.
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

    The query contains all dialogue messages up to and including the
    current user's message.

    Previous music recommendations are expanded into track metadata.

    Args:
        conversations:
            Full conversation for the session.

        target_turn_number:
            Current turn to retrieve tracks for.

        catalog:
            Track metadata catalog.

        corpus_types:
            Metadata fields used when expanding previous music turns.

    Returns:
        Conversation history formatted as a retrieval query.
    """

    lines = []

    for message in conversations:

        # Stop once we pass the target turn.
        if message["turn_number"] > target_turn_number:
            break

        # For the current turn, only include the user's message.
        if (
            message["turn_number"] == target_turn_number
            and message["role"] != "user"
        ):
            break

        role = message["role"]
        content = message["content"]

        # Expand previous music track IDs into metadata.
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


def run_bm25_tags(
    retriever: BM25Retriever,
    catalog: MusicCatalogLoader,
    dataset_name: str,
    split: str,
    topk: int,
    max_sessions: int | None = None,
) -> list[dict]:
    """Run BM25 + tags retrieval over a dialogue dataset split."""

    print("Loading dialogue dataset...")

    dataset = load_dataset(
        dataset_name,
        split=split,
    )

    # Optional small test run.
    if max_sessions is not None:

        max_sessions = min(
            max_sessions,
            len(dataset),
        )

        dataset = dataset.select(
            range(max_sessions)
        )

    print(
        f"Sessions: {len(dataset)}"
    )

    predictions = []

    # ---------------------------------------------------------
    # Process every session
    # ---------------------------------------------------------

    for session_index, item in enumerate(dataset):

        # -----------------------------------------------------
        # Eight recommendation turns per session
        # -----------------------------------------------------

        for turn_number in range(
            1,
            NUM_TURNS + 1,
        ):

            query = _build_retrieval_input(
                conversations=item["conversations"],
                target_turn_number=turn_number,
                catalog=catalog,
                corpus_types=CORPUS_TYPES,
            )

            predicted_track_ids = (
                retriever.text_to_item_retrieval(
                    query=query,
                    topk=topk,
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

        # Progress display.
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
            "Run MusicCRS BM25 retrieval "
            "using track metadata and tags"
        )
    )

    parser.add_argument(
        "--dialogue_dataset",
        default=(
            "talkpl-ai/"
            "TalkPlayData-Challenge-Dataset"
        ),
    )

    parser.add_argument(
        "--split",
        default="test",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max_sessions",
        type=int,
        default=None,
        help=(
            "Optional number of sessions "
            "for quick experiments"
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to predictions JSON",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("MusicCRS BM25 + Tags Retrieval")
    print("=" * 60)

    print(
        f"Split      : {args.split}"
    )

    print(
        f"Top-k      : {args.topk}"
    )

    print(
        f"Sessions   : "
        f"{args.max_sessions or 'ALL'}"
    )

    print(
        f"Fields     : "
        f"{', '.join(CORPUS_TYPES)}"
    )

    print(
        f"Output     : {args.output}"
    )

    # ---------------------------------------------------------
    # Initialize BM25
    # ---------------------------------------------------------

    print()
    print("Initializing BM25 + tags index...")

    retriever = BM25Retriever(
        corpus_types=CORPUS_TYPES,
    )

    # ---------------------------------------------------------
    # Load catalog
    # ---------------------------------------------------------

    catalog = MusicCatalogLoader()

    # ---------------------------------------------------------
    # Run retrieval
    # ---------------------------------------------------------

    predictions = run_bm25_tags(
        retriever=retriever,
        catalog=catalog,
        dataset_name=args.dialogue_dataset,
        split=args.split,
        topk=args.topk,
        max_sessions=args.max_sessions,
    )

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Done
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("BM25 + Tags retrieval completed")
    print("=" * 60)

    print(
        f"Predictions : {len(predictions)}"
    )

    print(
        f"Saved to    : {args.output}"
    )


if __name__ == "__main__":
    main()
