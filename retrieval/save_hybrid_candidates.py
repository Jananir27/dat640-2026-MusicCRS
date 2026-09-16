"""Generate and cache BM25 + dense candidate rankings.

The expensive dense retrieval is performed only once. The resulting
BM25 and dense rankings can then be reused to tune different RRF
weights without re-encoding the queries with Qwen.
"""

import argparse
import json

from datasets import load_dataset

from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .data_loader import MusicCatalogLoader
from .run_bm25_baseline import _build_retrieval_input


NUM_TURNS = 8

CORPUS_TYPES = [
    "track_name",
    "artist_name",
    "album_name",
    "tag_list",
]


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Cache BM25 and dense candidate rankings"
    )

    parser.add_argument(
        "--dialogue_dataset",
        default="talkpl-ai/TalkPlayData-Challenge-Dataset",
    )

    parser.add_argument(
        "--split",
        default="test",
    )

    parser.add_argument(
        "--candidate_k",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--max_sessions",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MusicCRS - Generate Hybrid Candidates")
    print("=" * 60)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    print("Loading dialogue dataset...")

    dataset = load_dataset(
        args.dialogue_dataset,
        split=args.split,
    )

    if args.max_sessions is not None:

        count = min(
            args.max_sessions,
            len(dataset),
        )

        dataset = dataset.select(
            range(count)
        )

    print(f"Sessions: {len(dataset)}")

    # ---------------------------------------------------------
    # Catalog
    # ---------------------------------------------------------

    print("Loading music catalog...")

    catalog = MusicCatalogLoader()

    # ---------------------------------------------------------
    # Build all queries
    # ---------------------------------------------------------

    queries = []
    metadata = []

    print("Building queries...")

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

            metadata.append(
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

    print(f"Total queries: {len(queries)}")

    # ---------------------------------------------------------
    # BM25 + tags
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("Running BM25 + tags")
    print("=" * 60)

    bm25 = BM25Retriever(
        corpus_types=CORPUS_TYPES,
    )

    bm25_results = (
        bm25.batch_text_to_item_retrieval(
            queries,
            args.candidate_k,
        )
    )

    # ---------------------------------------------------------
    # Dense
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("Running Qwen dense retrieval")
    print("=" * 60)

    dense = DenseRetriever()

    dense_results = (
        dense.batch_text_to_item_retrieval(
            queries,
            args.candidate_k,
        )
    )

    # ---------------------------------------------------------
    # Save candidates
    # ---------------------------------------------------------

    records = []

    for info, bm25_tracks, dense_tracks in zip(
        metadata,
        bm25_results,
        dense_results,
    ):

        records.append(
            {
                "session_id":
                    info["session_id"],

                "turn_number":
                    info["turn_number"],

                "bm25_track_ids":
                    bm25_tracks,

                "dense_track_ids":
                    dense_tracks,
            }
        )

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            records,
            f,
            indent=2,
        )

    print()
    print("=" * 60)
    print("Candidate generation completed")
    print("=" * 60)

    print(f"Records : {len(records)}")
    print(f"Saved   : {args.output}")


if __name__ == "__main__":
    main()
