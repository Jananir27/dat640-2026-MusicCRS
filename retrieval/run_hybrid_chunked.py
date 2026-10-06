"""Run the full MusicCRS hybrid retrieval experiment in resumable chunks.

One command will:

1. Load the MusicCRS test dataset.
2. Process sessions in chunks.
3. Run BM25 + tags and Qwen dense retrieval.
4. Fuse rankings using weighted RRF.
5. Save every completed chunk.
6. Skip completed chunks if the command is restarted.
7. Combine all chunks.
8. Evaluate the complete predictions.
9. Save the final metrics.

Default tuned configuration:

    BM25 weight = 1.0
    Dense weight = 0.25
    RRF k = 60
    Candidate k = 50
    Final top-k = 20
"""

import argparse
import json
import os
import time

from datasets import load_dataset

from .hybrid import HybridRetriever
from .data_loader import MusicCatalogLoader
from .run_bm25_baseline import _build_retrieval_input
from .evaluation.evaluate import evaluate


NUM_TURNS = 8

CORPUS_TYPES = [
    "track_name",
    "artist_name",
    "album_name",
    "tag_list",
]


def process_chunk(
    retriever,
    catalog,
    dataset,
    start_session,
    end_session,
    topk,
):
    """Process one session chunk."""

    print()
    print("=" * 70)
    print(
        f"Processing sessions "
        f"{start_session} - {end_session - 1}"
    )
    print("=" * 70)

    chunk_dataset = dataset.select(
        range(start_session, end_session)
    )

    queries = []
    metadata = []

    # -----------------------------------------------------
    # Build queries
    # -----------------------------------------------------

    print("Building retrieval queries...")

    for item in chunk_dataset:

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
                    "session_id":
                        item["session_id"],

                    "turn_number":
                        turn_number,
                }
            )

    print(
        f"Built {len(queries)} queries."
    )

    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------

    start_time = time.time()

    retrieved_tracks = (
        retriever.batch_text_to_item_retrieval(
            queries,
            topk,
        )
    )

    elapsed = time.time() - start_time

    print(
        f"Chunk retrieval time: "
        f"{elapsed / 60:.2f} minutes"
    )

    # -----------------------------------------------------
    # Build predictions
    # -----------------------------------------------------

    predictions = []

    for info, tracks in zip(
        metadata,
        retrieved_tracks,
    ):

        predictions.append(
            {
                "session_id":
                    info["session_id"],

                "turn_number":
                    info["turn_number"],

                "predicted_track_ids":
                    tracks,

                "predicted_response":
                    "",
            }
        )

    return predictions


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run full MusicCRS hybrid retrieval "
            "using resumable chunks"
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
        "--chunk_size",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=20,
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
        "--catalog_size",
        type=int,
        default=47071,
    )

    parser.add_argument(
        "--ground_truth",
        default="ground_truth.json",
    )

    parser.add_argument(
        "--output_dir",
        default="predictions/hybrid_full",
    )

    parser.add_argument(
        "--final_predictions",
        default="predictions_hybrid_full.json",
    )

    parser.add_argument(
        "--results",
        default="results_hybrid_full.json",
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("MusicCRS FULL HYBRID RETRIEVAL")
    print("=" * 70)

    print(
        f"Split          : {args.split}"
    )

    print(
        f"Chunk size     : {args.chunk_size}"
    )

    print(
        f"Top-k          : {args.topk}"
    )

    print(
        f"Candidate-k    : {args.candidate_k}"
    )

    print(
        "BM25 weight    : 1.0"
    )

    print(
        f"Dense weight   : {args.dense_weight}"
    )

    print(
        f"RRF k          : {args.rrf_k}"
    )

    print(
        f"Output dir     : {args.output_dir}"
    )

    # -----------------------------------------------------
    # Create output directory
    # -----------------------------------------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    print()
    print("Loading dialogue dataset...")

    dataset = load_dataset(
        args.dialogue_dataset,
        split=args.split,
    )

    total_sessions = len(dataset)

    print(
        f"Total sessions : {total_sessions}"
    )

    print(
        f"Total queries  : "
        f"{total_sessions * NUM_TURNS}"
    )

    # -----------------------------------------------------
    # Load catalog
    # -----------------------------------------------------

    print()
    print("Loading music catalog...")

    catalog = MusicCatalogLoader()

    # -----------------------------------------------------
    # Initialize hybrid retriever ONCE
    #
    # Important:
    # Qwen should not be reloaded for every chunk.
    # -----------------------------------------------------

    print()
    print("Initializing HybridRetriever...")

    retriever = HybridRetriever(
        candidate_k=args.candidate_k,
        dense_weight=args.dense_weight,
        rrf_k=args.rrf_k,
    )

    # -----------------------------------------------------
    # Process all chunks
    # -----------------------------------------------------

    total_start_time = time.time()

    chunk_files = []

    for start_session in range(
        0,
        total_sessions,
        args.chunk_size,
    ):

        end_session = min(
            start_session + args.chunk_size,
            total_sessions,
        )

        chunk_filename = (
            f"chunk_"
            f"{start_session:04d}_"
            f"{end_session - 1:04d}.json"
        )

        chunk_path = os.path.join(
            args.output_dir,
            chunk_filename,
        )

        chunk_files.append(
            chunk_path
        )

        # -------------------------------------------------
        # Resume support
        # -------------------------------------------------

        if os.path.exists(chunk_path):

            print()
            print(
                f"✓ Chunk already exists: "
                f"{chunk_filename}"
            )

            print(
                "  Skipping this chunk."
            )

            continue

        # -------------------------------------------------
        # Run chunk
        # -------------------------------------------------

        predictions = process_chunk(
            retriever=retriever,
            catalog=catalog,
            dataset=dataset,
            start_session=start_session,
            end_session=end_session,
            topk=args.topk,
        )

        # -------------------------------------------------
        # Validate chunk
        # -------------------------------------------------

        expected = (
            (end_session - start_session)
            * NUM_TURNS
        )

        if len(predictions) != expected:

            raise RuntimeError(
                f"Expected {expected} predictions "
                f"but generated {len(predictions)}"
            )

        # -------------------------------------------------
        # Save chunk immediately
        # -------------------------------------------------

        with open(
            chunk_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                predictions,
                f,
                indent=2,
            )

        print(
            f"✓ Saved {len(predictions)} "
            f"predictions"
        )

        print(
            f"  {chunk_path}"
        )

    # -----------------------------------------------------
    # Combine chunks
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("COMBINING CHUNKS")
    print("=" * 70)

    all_predictions = []

    for chunk_path in chunk_files:

        if not os.path.exists(chunk_path):

            raise RuntimeError(
                f"Missing chunk: {chunk_path}"
            )

        with open(
            chunk_path,
            "r",
            encoding="utf-8",
        ) as f:

            chunk_predictions = json.load(f)

        all_predictions.extend(
            chunk_predictions
        )

        print(
            f"Loaded "
            f"{os.path.basename(chunk_path)} "
            f"({len(chunk_predictions)} predictions)"
        )

    expected_total = (
        total_sessions
        * NUM_TURNS
    )

    if len(all_predictions) != expected_total:

        raise RuntimeError(
            f"Expected {expected_total} total "
            f"predictions but found "
            f"{len(all_predictions)}"
        )

    # -----------------------------------------------------
    # Save combined predictions
    # -----------------------------------------------------

    with open(
        args.final_predictions,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            all_predictions,
            f,
            indent=2,
        )

    print()
    print(
        f"✓ Combined predictions saved to "
        f"{args.final_predictions}"
    )

    print(
        f"✓ Total predictions: "
        f"{len(all_predictions)}"
    )

    # -----------------------------------------------------
    # Load ground truth
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)

    if not os.path.exists(
        args.ground_truth
    ):

        print(
            f"Ground truth file not found: "
            f"{args.ground_truth}"
        )

        print(
            "Predictions were generated successfully, "
            "but automatic evaluation was skipped."
        )

        return

    with open(
        args.ground_truth,
        "r",
        encoding="utf-8",
    ) as f:

        ground_truth = json.load(f)

    print(
        f"Ground truth records: "
        f"{len(ground_truth)}"
    )

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    results = evaluate(
        predictions=all_predictions,
        ground_truth=ground_truth,
        catalog_size=args.catalog_size,
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    with open(
        args.results,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    total_elapsed = (
        time.time()
        - total_start_time
    )

    print()
    print("=" * 70)
    print("FULL HYBRID RETRIEVAL COMPLETED")
    print("=" * 70)

    print(
        f"nDCG@1           : "
        f"{results['ndcg@1']:.6f}"
    )

    print(
        f"nDCG@10          : "
        f"{results['ndcg@10']:.6f}"
    )

    print(
        f"nDCG@20          : "
        f"{results['ndcg@20']:.6f}"
    )

    print(
        f"Catalog diversity: "
        f"{results['catalog_diversity']:.6f}"
    )

    print(
        f"FINAL SCORE      : "
        f"{results['final_score']:.6f}"
    )

    print()
    print(
        f"Predictions      : "
        f"{args.final_predictions}"
    )

    print(
        f"Results          : "
        f"{args.results}"
    )

    print(
        f"Total runtime    : "
        f"{total_elapsed / 60:.2f} minutes"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
