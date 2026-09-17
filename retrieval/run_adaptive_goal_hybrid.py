"""Adaptive goal-aware hybrid retrieval for MusicCRS.

The retrieval query is adapted according to conversation_goal["specificity"].

Strategy
--------
HH:
    Exact/highly specific request.
    Strongly emphasize current user message.
    Avoid unnecessary historical/profile noise.

HL:
    Detailed request for multiple recommendations.
    Emphasize current request while retaining goal and recent context.

LH:
    User is trying to identify a specific item from a vague description.
    Strongly emphasize the conversation goal because it may contain useful
    identifying information not present in the user utterance.

LL:
    Broad discovery/recommendation request.
    Use goal, conversational context, previous music and musical preference.

Retrieval:
    Adaptive query
        -> BM25 + Tags
        -> Qwen Dense
        -> Weighted RRF
        -> Top-K
"""

import argparse
import json
import time

from datasets import load_dataset

from .data_loader import MusicCatalogLoader
from .hybrid import HybridRetriever


NUM_TURNS = 8


# ============================================================
# Utility
# ============================================================

def safe_text(value) -> str:
    """Convert metadata values to text."""

    if value is None:
        return ""

    if isinstance(value, list):
        return ", ".join(str(v) for v in value)

    return str(value)


# ============================================================
# Current user request
# ============================================================

def get_current_user_message(
    conversations: list[dict],
    target_turn_number: int,
) -> str:
    """Get the user message for the current turn."""

    for message in conversations:

        if (
            message["turn_number"] == target_turn_number
            and message["role"] == "user"
        ):
            return safe_text(
                message["content"]
            ).strip()

    return ""


# ============================================================
# Previous user requests
# ============================================================

def get_previous_user_messages(
    conversations: list[dict],
    target_turn_number: int,
    max_previous: int,
) -> list[str]:
    """Get recent user messages before the current turn."""

    previous = []

    for message in conversations:

        if message["turn_number"] >= target_turn_number:
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
# Previous recommended music
# ============================================================

def get_previous_music(
    conversations: list[dict],
    target_turn_number: int,
    catalog: MusicCatalogLoader,
    max_previous: int,
) -> list[str]:
    """Get compact metadata for recently recommended tracks."""

    music = []

    for message in conversations:

        if message["turn_number"] >= target_turn_number:
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

        track_name = safe_text(
            metadata.get("track_name")
        )

        artist_name = safe_text(
            metadata.get("artist_name")
        )

        album_name = safe_text(
            metadata.get("album_name")
        )

        tags = metadata.get(
            "tag_list"
        )

        if track_name:
            parts.append(
                f"track {track_name}"
            )

        if artist_name:
            parts.append(
                f"artist {artist_name}"
            )

        if album_name:
            parts.append(
                f"album {album_name}"
            )

        if tags:

            if isinstance(tags, list):

                # Avoid enormous tag strings.
                tags = tags[:8]

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

            music.append(
                "; ".join(parts)
            )

    return music[-max_previous:]


# ============================================================
# Conversation goal
# ============================================================

def get_goal_text(
    conversation_goal: dict,
) -> str:
    """Extract listener goal text."""

    if not conversation_goal:
        return ""

    return safe_text(
        conversation_goal.get(
            "listener_goal"
        )
    ).strip()


# ============================================================
# User music preference
# ============================================================

def get_profile_text(
    user_profile: dict,
) -> str:
    """Extract music-relevant profile information."""

    if not user_profile:
        return ""

    preference = user_profile.get(
        "preferred_musical_culture"
    )

    if not preference:
        return ""

    return (
        "preferred musical culture: "
        + safe_text(preference)
    )


# ============================================================
# Repetition helper
# ============================================================

def repeat_text(
    label: str,
    text: str,
    repetitions: int,
) -> list[str]:
    """Repeat an important field for simple lexical weighting."""

    if not text:
        return []

    return [
        f"{label}:\n{text}"
        for _ in range(repetitions)
    ]


# ============================================================
# HH query
# ============================================================

def build_hh_query(
    current_request: str,
    goal_text: str,
) -> str:
    """Build query for highly explicit / exact requests.

    Examples:
        Play 'Africa' by Toto.
        Play the song with the exact lyrics ...

    Current request receives the strongest weight.
    """

    parts = []

    # Current request x3
    parts.extend(
        repeat_text(
            "CURRENT REQUEST",
            current_request,
            3,
        )
    )

    # Goal x1
    if goal_text:

        parts.append(
            "GOAL:\n"
            + goal_text
        )

    return "\n\n".join(parts)


# ============================================================
# HL query
# ============================================================

def build_hl_query(
    current_request: str,
    goal_text: str,
    previous_requests: list[str],
    previous_music: list[str],
) -> str:
    """Build query for detailed multi-item recommendations."""

    parts = []

    # Current request x2
    parts.extend(
        repeat_text(
            "CURRENT REQUEST",
            current_request,
            2,
        )
    )

    # Goal x1
    if goal_text:

        parts.append(
            "GOAL:\n"
            + goal_text
        )

    # Last user request
    if previous_requests:

        parts.append(
            "RECENT USER CONTEXT:\n"
            + "\n".join(
                previous_requests[-1:]
            )
        )

    # Recent music
    if previous_music:

        parts.append(
            "RECENT MUSIC:\n"
            + "\n".join(
                previous_music[-1:]
            )
        )

    return "\n\n".join(parts)


# ============================================================
# LH query
# ============================================================

def build_lh_query(
    current_request: str,
    goal_text: str,
    previous_requests: list[str],
    previous_music: list[str],
) -> str:
    """Build query for vague descriptions of a specific target.

    Goal receives strong weighting because these sessions may contain
    identifying information in the listener goal.
    """

    parts = []

    # Current request x2
    parts.extend(
        repeat_text(
            "CURRENT REQUEST",
            current_request,
            2,
        )
    )

    # Goal x2
    parts.extend(
        repeat_text(
            "TARGET GOAL",
            goal_text,
            2,
        )
    )

    if previous_requests:

        parts.append(
            "RECENT USER CONTEXT:\n"
            + "\n".join(
                previous_requests[-2:]
            )
        )

    if previous_music:

        parts.append(
            "RECENT MUSIC:\n"
            + "\n".join(
                previous_music[-2:]
            )
        )

    return "\n\n".join(parts)


# ============================================================
# LL query
# ============================================================

def build_ll_query(
    current_request: str,
    goal_text: str,
    previous_requests: list[str],
    previous_music: list[str],
    profile_text: str,
) -> str:
    """Build query for broad discovery/recommendation sessions."""

    parts = []

    # Current request x2
    parts.extend(
        repeat_text(
            "CURRENT REQUEST",
            current_request,
            2,
        )
    )

    # Goal x2
    parts.extend(
        repeat_text(
            "DISCOVERY GOAL",
            goal_text,
            2,
        )
    )

    if previous_requests:

        parts.append(
            "PREVIOUS USER PREFERENCES:\n"
            + "\n".join(
                previous_requests[-2:]
            )
        )

    if previous_music:

        parts.append(
            "RECENT MUSIC CONTEXT:\n"
            + "\n".join(
                previous_music[-2:]
            )
        )

    if profile_text:

        parts.append(
            "USER MUSIC PREFERENCE:\n"
            + profile_text
        )

    return "\n\n".join(parts)


# ============================================================
# Adaptive query builder
# ============================================================

def build_adaptive_query(
    item: dict,
    target_turn_number: int,
    catalog: MusicCatalogLoader,
) -> str:
    """Build a query according to the session specificity type."""

    conversations = item[
        "conversations"
    ]

    goal = (
        item.get("conversation_goal")
        or {}
    )

    specificity = (
        goal.get("specificity")
        or "LL"
    )

    current_request = (
        get_current_user_message(
            conversations,
            target_turn_number,
        )
    )

    goal_text = get_goal_text(
        goal
    )

    profile_text = get_profile_text(
        item.get("user_profile")
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

    # --------------------------------------------------------
    # Adaptive routing
    # --------------------------------------------------------

    if specificity == "HH":

        return build_hh_query(
            current_request=current_request,
            goal_text=goal_text,
        )

    if specificity == "HL":

        return build_hl_query(
            current_request=current_request,
            goal_text=goal_text,
            previous_requests=previous_requests,
            previous_music=previous_music,
        )

    if specificity == "LH":

        return build_lh_query(
            current_request=current_request,
            goal_text=goal_text,
            previous_requests=previous_requests,
            previous_music=previous_music,
        )

    # LL or unknown fallback
    return build_ll_query(
        current_request=current_request,
        goal_text=goal_text,
        previous_requests=previous_requests,
        previous_music=previous_music,
        profile_text=profile_text,
    )


# ============================================================
# Retrieval
# ============================================================

def run_adaptive_hybrid(
    retriever: HybridRetriever,
    catalog: MusicCatalogLoader,
    dataset,
    topk: int,
) -> list[dict]:
    """Run adaptive goal-aware hybrid retrieval."""

    print()
    print("=" * 72)
    print("BUILDING ADAPTIVE GOAL-AWARE QUERIES")
    print("=" * 72)

    queries = []
    metadata = []

    specificity_counts = {
        "HH": 0,
        "HL": 0,
        "LH": 0,
        "LL": 0,
    }

    for session_index, item in enumerate(
        dataset
    ):

        specificity = (
            item.get(
                "conversation_goal",
                {}
            ).get(
                "specificity",
                "LL",
            )
        )

        if specificity in specificity_counts:
            specificity_counts[
                specificity
            ] += 1

        for turn_number in range(
            1,
            NUM_TURNS + 1,
        ):

            query = build_adaptive_query(
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

                    "specificity":
                        specificity,
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
        f"Total queries: {len(queries)}"
    )

    print()
    print("Specificity distribution:")

    for key, value in (
        specificity_counts.items()
    ):
        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------
    # Show examples
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("ADAPTIVE QUERY EXAMPLES")
    print("=" * 72)

    shown = set()

    for query, info in zip(
        queries,
        metadata,
    ):

        specificity = info[
            "specificity"
        ]

        if specificity in shown:
            continue

        print()
        print(
            f"--- {specificity} ---"
        )

        print(query)

        shown.add(
            specificity
        )

        if len(shown) == 4:
            break

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("STARTING ADAPTIVE HYBRID RETRIEVAL")
    print("=" * 72)

    start = time.time()

    results = (
        retriever.batch_text_to_item_retrieval(
            queries,
            topk,
        )
    )

    elapsed = time.time() - start

    print()
    print(
        f"Retrieval completed in "
        f"{elapsed / 60:.2f} minutes"
    )

    predictions = []

    for info, tracks in zip(
        metadata,
        results,
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
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run adaptive goal-aware "
            "hybrid MusicCRS retrieval"
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
    print("=" * 72)
    print("MusicCRS ADAPTIVE GOAL-AWARE HYBRID")
    print("=" * 72)

    print(
        f"Split        : {args.split}"
    )

    print(
        f"Sessions     : "
        f"{args.max_sessions or 'ALL'}"
    )

    print(
        f"Top-k        : {args.topk}"
    )

    print(
        f"Candidate-k  : "
        f"{args.candidate_k}"
    )

    print(
        "BM25 weight  : 1.0"
    )

    print(
        f"Dense weight : "
        f"{args.dense_weight}"
    )

    print(
        f"RRF k        : "
        f"{args.rrf_k}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
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

    print(
        f"Loaded {len(dataset)} sessions."
    )

    # --------------------------------------------------------
    # Catalog
    # --------------------------------------------------------

    print()
    print("Loading catalog...")

    catalog = MusicCatalogLoader()

    # --------------------------------------------------------
    # Retriever
    # --------------------------------------------------------

    print()
    print("Initializing HybridRetriever...")

    retriever = HybridRetriever(
        candidate_k=args.candidate_k,
        dense_weight=args.dense_weight,
        rrf_k=args.rrf_k,
    )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    total_start = time.time()

    predictions = run_adaptive_hybrid(
        retriever=retriever,
        catalog=catalog,
        dataset=dataset,
        topk=args.topk,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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
    print("=" * 72)
    print("ADAPTIVE GOAL-AWARE HYBRID COMPLETED")
    print("=" * 72)

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

    print("=" * 72)


if __name__ == "__main__":
    main()
