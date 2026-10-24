"""Specificity-aware query optimization for MusicCRS.

Creates retrieval queries tailored to the conversation-goal specificity:

    HH - highly specific item / intent
    HL - specific constraints, broader result set
    LH - vague description of a relatively specific target
    LL - broad discovery / exploration

This module only constructs query text.
It does not perform retrieval and does not require a GPU.
"""

import re


SPECIFICITIES = {
    "HH",
    "HL",
    "LH",
    "LL",
}


def clean_text(text: str) -> str:
    """Normalize whitespace without removing useful punctuation."""

    if not text:
        return ""

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def get_user_messages(
    conversations: list[dict],
    target_turn_number: int,
) -> list[str]:
    """Return user messages up to the target turn."""

    messages = []

    for message in conversations:

        turn = message.get(
            "turn_number",
            0,
        )

        if turn > target_turn_number:
            break

        if (
            message.get("role") == "user"
            and turn <= target_turn_number
        ):

            content = clean_text(
                message.get(
                    "content",
                    "",
                )
            )

            if content:
                messages.append(content)

    return messages


def get_current_user_message(
    conversations: list[dict],
    target_turn_number: int,
) -> str:
    """Return current target-turn user message."""

    for message in conversations:

        if (
            message.get("turn_number")
            == target_turn_number
            and message.get("role")
            == "user"
        ):

            return clean_text(
                message.get(
                    "content",
                    "",
                )
            )

    return ""


def get_recent_user_context(
    conversations: list[dict],
    target_turn_number: int,
    max_messages: int = 3,
) -> str:
    """Return recent user requests up to current turn."""

    messages = get_user_messages(
        conversations,
        target_turn_number,
    )

    messages = messages[
        -max_messages:
    ]

    return " ".join(messages)


def optimize_query(
    conversations: list[dict],
    target_turn_number: int,
    specificity: str,
    conversation_goal: dict | None = None,
) -> str:
    """Construct specificity-aware retrieval query.

    HH:
        Current user message only.

        Useful for:
            exact song
            artist + song
            lyric phrase
            known item

    HL:
        Current user request + previous user request.

        Useful when the current turn refines an earlier request.

    LH:
        Current request + listener goal.

        Useful when the user describes a known target vaguely.

    LL:
        Recent user conversation + listener goal.

        Useful for broad discovery where context matters.
    """

    if specificity not in SPECIFICITIES:
        specificity = "LL"

    current = get_current_user_message(
        conversations,
        target_turn_number,
    )

    goal_text = ""

    if conversation_goal:

        goal_text = clean_text(
            conversation_goal.get(
                "listener_goal",
                "",
            )
        )

    # ========================================================
    # HH
    # ========================================================

    if specificity == "HH":

        # Exact/high-specificity queries should avoid
        # unnecessary conversation noise.
        return current

    # ========================================================
    # HL
    # ========================================================

    if specificity == "HL":

        recent = get_recent_user_context(
            conversations,
            target_turn_number,
            max_messages=2,
        )

        return recent or current

    # ========================================================
    # LH
    # ========================================================

    if specificity == "LH":

        parts = []

        if current:
            parts.append(current)

        if goal_text:
            parts.append(goal_text)

        return " ".join(parts)

    # ========================================================
    # LL
    # ========================================================

    recent = get_recent_user_context(
        conversations,
        target_turn_number,
        max_messages=3,
    )

    parts = []

    if recent:
        parts.append(recent)

    if goal_text:
        parts.append(goal_text)

    return " ".join(parts)
