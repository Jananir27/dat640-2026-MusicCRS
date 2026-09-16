"""Generate hybrid BM25 + dense predictions for MusicCRS."""

import argparse
import json

from datasets import load_dataset

from .hybrid import HybridRetriever
from .data_loader import MusicCatalogLoader
from .run_bm25_baseline import _build_retrieval_input


NUM_TURNS = 8

CORPUS_TYPES = [
    "track_name",
    "artist_name",
    "album_name",
    "tag_list",
]


def run_hybrid(
    retriever: HybridRetriever,
    catalog: MusicCatalogLoader,
    dataset_name: str,
    split: str,
    topk: int,
    max_sessions: int | None = None,
) -> list[dict]:

    print()
    print("Loading dialogue dataset...")

    dataset = load_dataset(
        dataset_name,
        split=split,
    )

    if max_sessions is not None:

        number_of_sessions = min(
            max_sessions,
            len(dataset),
        )

        dataset = dataset.select(
            range(number_of_sessions)
        )

    print(f"Using {len(dataset)} sessions")

    # --------------------------------------------
    # Build queries
    # --------------------------------------------

    queries = []
    query_metadata = []

    print("Building retrieval queries...")

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

        if (session_index + 1) % 100 == 0:
            print(
                f"Built queries for "
                f"{session_index + 1}/"
                f"{len(dataset)} sessions"
            )

    print(
        f"Built {len(queries)} queries."
    )

    # --------------------------------------------
    # Hybrid retrieval
    # --------------------------------------------

    retrieved_tracks = (
        retriever.batch_text_to_item_retrieval(
            queries,
            topk,
        )
    )

    # --------------------------------------------
    # Prediction format
    # --------------------------------------------

    predictions = []

    for metadata, track_ids in zip(
        query_metadata,
        retrieved_tracks,
    ):

        predictions.append(
            {
                "session_id":
                    metadata["session_id"],

                "turn_number":
                    metadata["turn_number"],

                "predicted_track_ids":
                    track_ids,

                "predicted_response": "",
            }
        )

    return predictions


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run hybrid BM25 + Qwen dense retrieval"
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
    )

    parser.add_argument(
        "--candidate_k",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--dense_weight",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--rrf_k",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("MusicCRS Hybrid Retrieval")
    print("=" * 60)

    print(f"Split        : {args.split}")
    print(f"Top-k        : {args.topk}")
    print(f"Candidate-k  : {args.candidate_k}")
    print(f"Dense weight : {args.dense_weight}")
    print(f"RRF k        : {args.rrf_k}")

    if args.max_sessions:
        print(
            f"Sessions     : {args.max_sessions}"
        )
    else:
        print("Sessions     : ALL")

    # --------------------------------------------
    # Initialize retriever
    # --------------------------------------------

    retriever = HybridRetriever(
        candidate_k=args.candidate_k,
        dense_weight=args.dense_weight,
        rrf_k=args.rrf_k,
    )

    catalog = MusicCatalogLoader()

    # --------------------------------------------
    # Run
    # --------------------------------------------

    predictions = run_hybrid(
        retriever=retriever,
        catalog=catalog,
        dataset_name=args.dialogue_dataset,
        split=args.split,
        topk=args.topk,
        max_sessions=args.max_sessions,
    )

    # --------------------------------------------
    # Save
    # --------------------------------------------

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
    print("Hybrid retrieval completed")
    print("=" * 60)

    print(
        f"Predictions : {len(predictions)}"
    )

    print(
        f"Saved to    : {args.output}"
    )


if __name__ == "__main__":
    main()
