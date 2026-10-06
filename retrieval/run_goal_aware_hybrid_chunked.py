"""Run goal-aware hybrid retrieval over the full MusicCRS test set in chunks.

Pipeline:
    Goal-aware query
        -> BM25 + Tags
        -> Qwen Dense
        -> Weighted RRF
        -> Top-20

The experiment is processed in resumable chunks. Each completed chunk is
saved immediately. After all chunks finish, predictions are combined and
evaluated automatically.
"""

import argparse
import json
import os
import time

from datasets import load_dataset

from .data_loader import MusicCatalogLoader
from .hybrid import HybridRetriever
from .run_goal_aware_hybrid import build_goal_aware_query
from .evaluation.evaluate import evaluate


NUM_TURNS = 8


def process_chunk(
    retriever: HybridRetriever,
    catalog: MusicCatalogLoader,
    dataset,
    start_session: int,
    end_session: int,
    topk: int,
) -> list[dict]:
    """Process one chunk of dialogue sessions."""

    print()
    print("=" * 72)
    print(
        f"PROCESSING SESSIONS "
        f"{start_session:04d}-{end_session - 1:04d}"
    )
    print("=" * 72)

    chunk_dataset = dataset.select(
        range(start_session, end_session)
    )

    queries = []
    metadata = []

    # ---------------------------------------------------------
    # Build goal-aware queries
    # ---------------------------------------------------------

    print("Building goal-aware queries...")

    for item in chunk_dataset:

        for turn_number in range(
            1,
            NUM_TURNS + 1,
        ):

            query = build_goal_aware_query(
                item=item,
                target_turn_number=turn_number,
                catalog=catalog,
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

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

    print()
    print("Starting hybrid retrieval...")

    start_time = time.time()

    retrieved_tracks = (
        retriever.batch_text_to_item_retrieval(
            queries,
            topk,
        )
    )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print(
        f"Chunk retrieval completed in "
        f"{elapsed / 60:.2f} minutes"
    )

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

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


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run full goal-aware MusicCRS "
            "hybrid retrieval in resumable chunks"
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
        default=(
            "predictions/"
            "goal_aware_hybrid_full"
        ),
    )

    parser.add_argument(
        "--final_predictions",
        default=(
            "predictions_"
            "goal_aware_hybrid_full.json"
        ),
    )

    parser.add_argument(
        "--results",
        default=(
            "results_"
            "goal_aware_hybrid_full.json"
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print("MusicCRS FULL GOAL-AWARE HYBRID RETRIEVAL")
    print("=" * 72)

    print(
        f"Split          : {args.split}"
    )

    print(
        f"Chunk size     : {args.chunk_size}"
    )

    print(
        f"Final top-k    : {args.topk}"
    )

    print(
        f"Candidate-k    : {args.candidate_k}"
    )

    print(
        "BM25 weight    : 1.0"
    )

    print(
        f"Dense weight   : "
        f"{args.dense_weight}"
    )

    print(
        f"RRF k          : "
        f"{args.rrf_k}"
    )

    print(
        f"Output dir     : "
        f"{args.output_dir}"
    )

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    print()
    print("Loading dialogue dataset...")

    dataset = load_dataset(
        args.dialogue_dataset,
        split=args.split,
    )

    total_sessions = len(dataset)

    print(
        f"Total sessions : "
        f"{total_sessions}"
    )

    print(
        f"Total queries  : "
        f"{total_sessions * NUM_TURNS}"
    )

    # ---------------------------------------------------------
    # Catalog
    # ---------------------------------------------------------

    print()
    print("Loading music catalog...")

    catalog = MusicCatalogLoader()

    # ---------------------------------------------------------
    # Hybrid retriever
    #
    # Load only ONCE.
    # The Qwen model must not be reloaded for each chunk.
    # ---------------------------------------------------------

    print()
    print("Initializing HybridRetriever...")

    retriever = HybridRetriever(
        candidate_k=args.candidate_k,
        dense_weight=args.dense_weight,
        rrf_k=args.rrf_k,
    )

    # ---------------------------------------------------------
    # Process chunks
    # ---------------------------------------------------------

    total_start = time.time()

    chunk_files = []

    for start_session in range(
        0,
        total_sessions,
        args.chunk_size,
    ):

        end_session = min(
            start_session
            + args.chunk_size,
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

        # -----------------------------------------------------
        # Resume support
        # -----------------------------------------------------

        if os.path.exists(
            chunk_path
        ):

            print()
            print(
                f"✓ Found completed chunk: "
                f"{chunk_filename}"
            )

            print(
                "  Skipping."
            )

            continue

        # -----------------------------------------------------
        # Process
        # -----------------------------------------------------

        predictions = process_chunk(
            retriever=retriever,
            catalog=catalog,
            dataset=dataset,
            start_session=start_session,
            end_session=end_session,
            topk=args.topk,
        )

        expected = (
            (end_session - start_session)
            * NUM_TURNS
        )

        if len(predictions) != expected:

            raise RuntimeError(
                f"Chunk generated "
                f"{len(predictions)} predictions; "
                f"expected {expected}."
            )

        # -----------------------------------------------------
        # Save immediately
        # -----------------------------------------------------

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

        print()
        print(
            f"✓ Saved {len(predictions)} "
            f"predictions"
        )

        print(
            f"  {chunk_path}"
        )

    # ---------------------------------------------------------
    # Combine chunks
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print("COMBINING CHUNKS")
    print("=" * 72)

    all_predictions = []

    for chunk_path in chunk_files:

        if not os.path.exists(
            chunk_path
        ):

            raise RuntimeError(
                f"Missing chunk: "
                f"{chunk_path}"
            )

        with open(
            chunk_path,
            "r",
            encoding="utf-8",
        ) as f:

            chunk_predictions = (
                json.load(f)
            )

        all_predictions.extend(
            chunk_predictions
        )

        print(
            f"Loaded "
            f"{os.path.basename(chunk_path)} "
            f"-> "
            f"{len(chunk_predictions)} "
            f"predictions"
        )

    expected_total = (
        total_sessions
        * NUM_TURNS
    )

    if (
        len(all_predictions)
        != expected_total
    ):

        raise RuntimeError(
            f"Combined prediction count "
            f"is {len(all_predictions)}, "
            f"expected {expected_total}."
        )

    # ---------------------------------------------------------
    # Save final predictions
    # ---------------------------------------------------------

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
        f"✓ Combined predictions: "
        f"{len(all_predictions)}"
    )

    print(
        f"✓ Saved to: "
        f"{args.final_predictions}"
    )

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print("EVALUATION")
    print("=" * 72)

    if not os.path.exists(
        args.ground_truth
    ):

        print(
            f"Ground truth not found: "
            f"{args.ground_truth}"
        )

        print(
            "Retrieval completed, but "
            "automatic evaluation was skipped."
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

    results = evaluate(
        predictions=all_predictions,
        ground_truth=ground_truth,
        catalog_size=args.catalog_size,
    )

    # ---------------------------------------------------------
    # Save evaluation
    # ---------------------------------------------------------

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

    total_elapsed = (
        time.time()
        - total_start
    )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print("FULL GOAL-AWARE HYBRID COMPLETED")
    print("=" * 72)

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
        f"{total_elapsed / 60:.2f} "
        f"minutes"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
