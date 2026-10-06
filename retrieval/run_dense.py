"""Generate dense retrieval predictions for the MusicCRS dataset.

For every session and turn, builds the full conversation history up to
the current user's message and retrieves top-k tracks using DenseRetriever.

Supports limiting the number of sessions for quick development tests.
"""

import argparse
import json

from datasets import load_dataset

from .dense import DenseRetriever
from .data_loader import MusicCatalogLoader
from .run_bm25_baseline import _build_retrieval_input


NUM_TURNS = 8

# Use the metadata configuration that performed best in BM25 experiments.
CORPUS_TYPES = [
    "track_name",
    "artist_name",
    "album_name",
    "tag_list",
]


def run_dense(
    retriever: DenseRetriever,
    catalog: MusicCatalogLoader,
    dataset_name: str,
    split: str,
    topk: int,
    max_sessions: int | None = None,
) -> list[dict]:
    """Run dense retrieval for every session and turn."""

    # ---------------------------------------------------------
    # Load dialogue dataset
    # ---------------------------------------------------------

    print()
    print("Loading dialogue dataset...")

    dataset = load_dataset(
        dataset_name,
        split=split,
    )

    # Optionally restrict the number of sessions.
    if max_sessions is not None:

        number_of_sessions = min(
            max_sessions,
            len(dataset),
        )

        dataset = dataset.select(
            range(number_of_sessions)
        )

    print(f"Using {len(dataset)} sessions")
    print(
        f"Expected queries: "
        f"{len(dataset) * NUM_TURNS}"
    )

    # ---------------------------------------------------------
    # Build ALL queries first
    # ---------------------------------------------------------

    print()
    print("Building retrieval queries...")

    queries = []
    query_metadata = []

    for session_index, item in enumerate(dataset):

        for turn_number in range(
            1,
            NUM_TURNS + 1,
        ):

            query = _build_retrieval_input(
                item["conversations"],
                turn_number,
                catalog,
                CORPUS_TYPES,
            )

            queries.append(query)

            query_metadata.append(
                {
                    "session_id": item["session_id"],
                    "turn_number": turn_number,
                }
            )

        # Show query-building progress for large runs.
        if (session_index + 1) % 100 == 0:
            print(
                f"Built queries for "
                f"{session_index + 1}/"
                f"{len(dataset)} sessions"
            )

    print()
    print(
        f"Built {len(queries)} queries successfully."
    )

    # ---------------------------------------------------------
    # Dense retrieval
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("Starting dense retrieval")
    print("=" * 60)

    retrieved_tracks = (
        retriever.batch_text_to_item_retrieval(
            queries,
            topk,
        )
    )

    # ---------------------------------------------------------
    # Convert to challenge prediction format
    # ---------------------------------------------------------

    print()
    print("Creating prediction records...")

    predictions = []

    for metadata, track_ids in zip(
        query_metadata,
        retrieved_tracks,
    ):

        predictions.append(
            {
                "session_id": metadata["session_id"],
                "turn_number": metadata["turn_number"],
                "predicted_track_ids": track_ids,
                "predicted_response": "",
            }
        )

    print(
        f"Created {len(predictions)} predictions."
    )

    return predictions


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen3 dense metadata retrieval "
            "for MusicCRS"
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
        help="Dataset split to evaluate",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Number of tracks to retrieve",
    )

    parser.add_argument(
        "--max_sessions",
        type=int,
        default=None,
        help=(
            "Limit number of sessions for testing. "
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
    # Display experiment configuration
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("MusicCRS Dense Retrieval")
    print("=" * 60)

    print(f"Split        : {args.split}")
    print(f"Top-k        : {args.topk}")

    if args.max_sessions is None:
        print("Sessions     : ALL")
    else:
        print(
            f"Sessions     : {args.max_sessions}"
        )

    print(f"Output       : {args.output}")

    # ---------------------------------------------------------
    # Initialize DenseRetriever
    # ---------------------------------------------------------

    print()
    print("Initializing DenseRetriever...")

    retriever = DenseRetriever()

    # DenseRetriever automatically selects:
    #
    # CUDA -> if NVIDIA GPU is available
    # CPU  -> otherwise

    # ---------------------------------------------------------
    # Load track metadata
    # ---------------------------------------------------------

    print()
    print("Loading music catalog...")

    catalog = MusicCatalogLoader()

    # ---------------------------------------------------------
    # Run retrieval
    # ---------------------------------------------------------

    predictions = run_dense(
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

    print()
    print("Saving predictions...")

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
    print("Dense retrieval completed")
    print("=" * 60)

    print(
        f"Predictions : {len(predictions)}"
    )

    print(
        f"Saved to    : {args.output}"
    )


if __name__ == "__main__":
    main()
