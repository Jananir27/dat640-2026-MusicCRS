"""Goal-aware hybrid retrieval for MusicCRS.

This experiment improves query construction rather than changing the
underlying BM25 + Dense hybrid retriever.

The retrieval query combines:

    1. Current user request       -> strongest signal
    2. Conversation goal         -> task/intent context
    3. Previous user requests    -> conversational preference context
    4. Previous recommended music -> compact track metadata
    5. User musical culture      -> weak personalization context

The query is then passed to the existing HybridRetriever:

    BM25 + Tags
         +
    Qwen Dense
         |
         v
    Weighted RRF
         |
         v
       Top-K
"""

import argparse
import json
import time

from datasets import load_dataset

from .data_loader import MusicCatalogLoader
from .hybrid import HybridRetriever


NUM_TURNS = 8

CORPUS_TYPES = [
    "track_name",
    "artist_name",
    "album_name",
    "tag_list",
]


# ============================================================
# Helper
# ============================================================

def safe_text(value) -> str:
    """Convert optional values into clean strings."""

    if value is None:
        return ""

    if isinstance(value, list):
        return ", ".join(str(v) for v in value)

    return str(value)


# ============================================================
# Current user message
# ============================================================

def get_current_user_message(
    conversations: list[dict],
    target_turn_number: int,
) -> str:
    """Return the current user request for the target turn."""

    for message in conversations:

        if (
            message["turn_number"] == target_turn_number
            and message["role"] == "user"
        ):
            return safe_text(
                message["content"]
            )

    return ""


# ============================================================
# Previous user requests
# ============================================================

def get_previous_user_messages(
    conversations: list[dict],
    target_turn_number: int,
    max_previous: int = 2,
) -> list[str]:
    """Return the most recent previous user requests.

    We intentionally keep only a small amount of history because old
    conversation turns can introduce stale artists, tracks and genres.
    """

    previous = []

    for message in conversations:

        turn = message["turn_number"]

        if turn >= target_turn_number:
            continue

        if message["role"] != "user":
            continue

        content = safe_text(
            message["content"]
        ).strip()

        if content:
            previous.append(content)

    return previous[-max_previous:]


# ============================================================
# Previous music
# ============================================================

def get_previous_music(
    conversations: list[dict],
    target_turn_number: int,
    catalog: MusicCatalogLoader,
    max_previous: int = 2,
) -> list[str]:
    """Expand recent previous music recommendations into metadata."""

    music_tracks = []

    for message in conversations:

        turn = message["turn_number"]

        if turn >= target_turn_number:
            continue

        if message["role"] != "music":
            continue

        track_id = message["content"]

        try:

            metadata = catalog.id_to_metadata(
                track_id
            )

        except KeyError:
            continue

        parts = []

        # Track name
        track_name = safe_text(
            metadata.get("track_name")
        )

        if track_name:
            parts.append(
                f"track {track_name}"
            )

        # Artist
        artist_name = safe_text(
            metadata.get("artist_name")
        )

        if artist_name:
            parts.append(
                f"artist {artist_name}"
            )

        # Album
        album_name = safe_text(
            metadata.get("album_name")
        )

        if album_name:
            parts.append(
                f"album {album_name}"
            )

        # Tags
        tags = metadata.get(
            "tag_list"
        )

        if tags:

            if isinstance(tags, list):

                # Limit tags to prevent huge queries.
                tags = tags[:10]

                tag_text = ", ".join(
                    str(tag)
                    for tag in tags
                )

            else:
                tag_text = str(tags)

            parts.append(
                f"tags {tag_text}"
            )

        if parts:

            music_tracks.append(
                "; ".join(parts)
            )

    return music_tracks[-max_previous:]


# ============================================================
# Goal information
# ============================================================

def get_goal_text(
    conversation_goal,
) -> str:
    """Convert the structured conversation goal into retrieval text."""

    if not conversation_goal:
        return ""

    parts = []

    listener_goal = conversation_goal.get(
        "listener_goal"
    )

    if listener_goal:

        parts.append(
            safe_text(listener_goal)
        )

    specificity = conversation_goal.get(
        "specificity"
    )

    if specificity:

        parts.append(
            f"specificity {specificity}"
        )

    return ". ".join(parts)


# ============================================================
# Profile information
# ============================================================

def get_profile_text(
    user_profile,
) -> str:
    """Extract music-relevant profile information.

    We intentionally avoid adding age/gender/country initially because
    those fields may add noise to text retrieval.

    Preferred musical culture is directly music-related and therefore
    more appropriate for this first experiment.
    """

    if not user_profile:
        return ""

    musical_culture = user_profile.get(
        "preferred_musical_culture"
    )

    if not musical_culture:
        return ""

    return (
        "preferred musical culture "
        + safe_text(musical_culture)
    )


# ============================================================
# Goal-aware query
# ============================================================

def build_goal_aware_query(
    item: dict,
    target_turn_number: int,
    catalog: MusicCatalogLoader,
) -> str:
    """Construct a focused retrieval query.

    Current request is deliberately repeated because it is the most
    important signal for identifying the track required at this turn.

    This is a simple lexical weighting technique: terms occurring in the
    current request receive more influence in BM25 while the same text
    still provides the dense retriever with the current intent.
    """

    conversations = item[
        "conversations"
    ]

    current_request = (
        get_current_user_message(
            conversations,
            target_turn_number,
        )
    )

    previous_requests = (
        get_previous_user_messages(
            conversations,
            target_turn_number,
            max_previous=2,
        )
    )

    previous_music = (
        get_previous_music(
            conversations,
            target_turn_number,
            catalog,
            max_previous=2,
        )
    )

    goal_text = get_goal_text(
        item.get("conversation_goal")
    )

    profile_text = get_profile_text(
        item.get("user_profile")
    )

    query_parts = []

    # --------------------------------------------------------
    # Strongest signal
    # --------------------------------------------------------

    if current_request:

        query_parts.append(
            "CURRENT USER REQUEST:\n"
            + current_request
        )

        # Repeat the current request once to increase its lexical weight.
        query_parts.append(
            "MAIN MUSIC REQUEST:\n"
            + current_request
        )

    # --------------------------------------------------------
    # Conversation goal
    # --------------------------------------------------------

    if goal_text:

        query_parts.append(
            "CONVERSATION GOAL:\n"
            + goal_text
        )

    # --------------------------------------------------------
    # Recent conversational context
    # --------------------------------------------------------

    if previous_requests:

        query_parts.append(
            "RECENT USER REQUESTS:\n"
            + "\n".join(
                previous_requests
            )
        )

    # --------------------------------------------------------
    # Previous recommendations
    # --------------------------------------------------------

    if previous_music:

        query_parts.append(
            "RECENT MUSIC CONTEXT:\n"
            + "\n".join(
                previous_music
            )
        )

    # --------------------------------------------------------
    # Weak profile signal
    # --------------------------------------------------------

    if profile_text:

        query_parts.append(
            "USER MUSIC PREFERENCE:\n"
            + profile_text
        )

    return "\n\n".join(
        query_parts
    )


# ============================================================
# Run retrieval
# ============================================================

def run_goal_aware_hybrid(
    retriever: HybridRetriever,
    catalog: MusicCatalogLoader,
    dataset,
    topk: int,
) -> list[dict]:
    """Run goal-aware hybrid retrieval."""

    print()
    print("=" * 70)
    print("BUILDING GOAL-AWARE QUERIES")
    print("=" * 70)

    queries = []
    metadata = []

    for session_index, item in enumerate(
        dataset
    ):

        for turn_number in range(
            1,
            NUM_TURNS + 1,
        ):

            query = build_goal_aware_query(
                item=item,
                target_turn_number=(
                    turn_number
                ),
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

        if (
            (session_index + 1) % 100 == 0
            or session_index + 1
            == len(dataset)
        ):

            print(
                f"Built queries for "
                f"{session_index + 1}/"
                f"{len(dataset)} sessions"
            )

    print()
    print(
        f"Built {len(queries)} "
        "goal-aware queries."
    )

    # --------------------------------------------------------
    # Example query
    # --------------------------------------------------------

    if queries:

        print()
        print("=" * 70)
        print("EXAMPLE QUERY")
        print("=" * 70)

        print(
            queries[0]
        )

        print("=" * 70)

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING GOAL-AWARE HYBRID RETRIEVAL")
    print("=" * 70)

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
        f"Retrieval completed in "
        f"{elapsed / 60:.2f} minutes"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

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


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run goal-aware hybrid retrieval "
            "for MusicCRS"
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
        "--max_sessions",
        type=int,
        default=None,
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
        "--output",
        required=True,
    )

    args = parser.parse_args()

    print()
    print("=" * 70)
    print("MusicCRS GOAL-AWARE HYBRID RETRIEVAL")
    print("=" * 70)

    print(
        f"Split          : "
        f"{args.split}"
    )

    print(
        f"Sessions       : "
        f"{args.max_sessions or 'ALL'}"
    )

    print(
        f"Top-k          : "
        f"{args.topk}"
    )

    print(
        f"Candidate-k    : "
        f"{args.candidate_k}"
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
        f"Output         : "
        f"{args.output}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print(
        "Loading dialogue dataset..."
    )

    dataset = load_dataset(
        args.dialogue_dataset,
        split=args.split,
    )

    if args.max_sessions is not None:

        number_sessions = min(
            args.max_sessions,
            len(dataset),
        )

        dataset = dataset.select(
            range(number_sessions)
        )

    print(
        f"Loaded {len(dataset)} "
        "sessions."
    )

    # --------------------------------------------------------
    # Catalog
    # --------------------------------------------------------

    print()
    print(
        "Loading music catalog..."
    )

    catalog = MusicCatalogLoader()

    # --------------------------------------------------------
    # Hybrid retriever
    # --------------------------------------------------------

    print()
    print(
        "Initializing HybridRetriever..."
    )

    retriever = HybridRetriever(
        candidate_k=args.candidate_k,
        dense_weight=args.dense_weight,
        rrf_k=args.rrf_k,
    )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    total_start = time.time()

    predictions = (
        run_goal_aware_hybrid(
            retriever=retriever,
            catalog=catalog,
            dataset=dataset,
            topk=args.topk,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

    elapsed = (
        time.time()
        - total_start
    )

    print()
    print("=" * 70)
    print(
        "GOAL-AWARE HYBRID RETRIEVAL COMPLETED"
    )
    print("=" * 70)

    print(
        f"Predictions : "
        f"{len(predictions)}"
    )

    print(
        f"Saved to    : "
        f"{args.output}"
    )

    print(
        f"Runtime     : "
        f"{elapsed / 60:.2f} minutes"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
