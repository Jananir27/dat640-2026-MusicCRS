"""Extract ground truth track IDs from the MusicCRS challenge dialogue dataset.

Each session in `talkpl-ai/TalkPlayData-Challenge-Dataset` has 8 turns; each
turn's conversation has a `user` message, a `music` message (the gold track
ID), and an `assistant` response. This script extracts, for every turn, the
gold track ID and response, in the format expected by `evaluate.py`.

Splits: `train` (15,199 sessions) for training/dev-tuning, `test` (1,000
sessions) as the held-out devset. Blind A/B sets do not have public ground
truth (they're scored server-side on the challenge leaderboard).
"""
import argparse
import json

from datasets import load_dataset

DEFAULT_DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Dataset"
NUM_TURNS = 8


def _parse_turn(conversations: list[dict], turn_number: int) -> tuple[str, str]:
    """Extracts (gold_track_id, gold_response) for one turn's conversation."""
    turn_msgs = [c for c in conversations if c["turn_number"] == turn_number]
    gold_track_id = turn_msgs[1]["content"]
    gold_response = turn_msgs[2]["content"]
    return gold_track_id, gold_response


def make_ground_truth(
    dataset_name: str = DEFAULT_DATASET_NAME, split: str = "test"
) -> list[dict]:
    """Builds ground truth entries for every session x turn in a split.

    Args:
        dataset_name: Hugging Face dataset name.
        split: Dataset split to use (e.g. "train" or "test").

    Returns:
        A list of dicts with session_id, user_id, turn_number,
        ground_truth_track_id, ground_truth_response.
    """
    dataset = load_dataset(dataset_name, split=split)
    ground_truth = []
    for item in dataset:
        for turn_number in range(1, NUM_TURNS + 1):
            gold_track_id, gold_response = _parse_turn(
                item["conversations"], turn_number
            )
            ground_truth.append(
                {
                    "session_id": item["session_id"],
                    "user_id": item["user_id"],
                    "turn_number": turn_number,
                    "ground_truth_track_id": gold_track_id,
                    "ground_truth_response": gold_response,
                }
            )
    return ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MusicCRS ground truth")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True, help="Path to write ground truth JSON")
    args = parser.parse_args()

    ground_truth = make_ground_truth(args.dataset_name, args.split)
    with open(args.output, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Wrote {len(ground_truth)} ground truth entries to {args.output}")


if __name__ == "__main__":
    main()
