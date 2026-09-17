"""Exact metadata matching for MusicCRS.

Provides lightweight lexical matching against:
    - track_name
    - artist_name
    - album_name

Designed primarily for highly specific (HH) requests such as:
    "Play 'Africa' by Toto."
    "Play Heart-Shaped Box by Nirvana."

This module does NOT perform dense retrieval and does not require a GPU.
"""

import re
import unicodedata

from .data_loader import MusicCatalogLoader


def normalize_text(text: str) -> str:
    """Normalize text for lexical matching."""

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(text),
    )

    text = text.lower()

    # Normalize apostrophes / quotes
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')

    # Replace punctuation with spaces
    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    # Collapse whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def token_set(text: str) -> set[str]:
    """Return normalized tokens."""

    normalized = normalize_text(text)

    if not normalized:
        return set()

    return set(
        normalized.split()
    )


def token_overlap(
    query: str,
    value: str,
) -> float:
    """Fraction of metadata tokens found in the query.

    Example:

        query:
            "play africa by toto"

        value:
            "africa"

        score:
            1.0
    """

    query_tokens = token_set(query)
    value_tokens = token_set(value)

    if not value_tokens:
        return 0.0

    overlap = (
        query_tokens
        & value_tokens
    )

    return (
        len(overlap)
        / len(value_tokens)
    )


def phrase_match(
    query: str,
    value: str,
) -> bool:
    """Check whether normalized metadata occurs as a phrase."""

    query_normalized = normalize_text(
        query
    )

    value_normalized = normalize_text(
        value
    )

    if not value_normalized:
        return False

    return (
        value_normalized
        in query_normalized
    )


class ExactMetadataMatcher:
    """Score tracks using exact/lexical metadata matching."""

    def __init__(
        self,
        catalog: MusicCatalogLoader | None = None,
    ) -> None:

        self.catalog = (
            catalog
            if catalog is not None
            else MusicCatalogLoader()
        )

    def score_track(
        self,
        query: str,
        track_id: str,
    ) -> float:
        """Calculate exact-match score for one track.

        Scoring:

        Track title:
            exact phrase     +6
            full token match +3

        Artist:
            exact phrase     +4
            full token match +2

        Album:
            exact phrase     +1

        Title + artist exact:
            additional       +5

        The combined title+artist bonus is particularly useful
        for known-song HH queries.
        """

        try:
            metadata = (
                self.catalog.id_to_metadata(
                    track_id
                )
            )

        except KeyError:
            return 0.0

        track_name = str(
            metadata.get(
                "track_name",
                ""
            )
            or ""
        )

        artist_name = str(
            metadata.get(
                "artist_name",
                ""
            )
            or ""
        )

        album_name = str(
            metadata.get(
                "album_name",
                ""
            )
            or ""
        )

        score = 0.0

        # ----------------------------------------------------
        # Track title
        # ----------------------------------------------------

        title_exact = phrase_match(
            query,
            track_name,
        )

        title_overlap = token_overlap(
            query,
            track_name,
        )

        if title_exact:
            score += 6.0

        elif title_overlap == 1.0:
            score += 3.0

        elif title_overlap >= 0.5:
            score += (
                1.5
                * title_overlap
            )

        # ----------------------------------------------------
        # Artist
        # ----------------------------------------------------

        artist_exact = phrase_match(
            query,
            artist_name,
        )

        artist_overlap = token_overlap(
            query,
            artist_name,
        )

        if artist_exact:
            score += 4.0

        elif artist_overlap == 1.0:
            score += 2.0

        elif artist_overlap >= 0.5:
            score += artist_overlap

        # ----------------------------------------------------
        # Album
        # ----------------------------------------------------

        album_exact = phrase_match(
            query,
            album_name,
        )

        if album_exact:
            score += 1.0

        # ----------------------------------------------------
        # Strong known-song bonus
        # ----------------------------------------------------

        if (
            title_exact
            and artist_exact
        ):
            score += 5.0

        return score

    def rerank(
        self,
        query: str,
        track_ids: list[str],
        topk: int | None = None,
        exact_weight: float = 1.0,
    ) -> list[str]:
        """Rerank an existing candidate list.

        The original rank is retained as a small base score, while
        exact metadata matches receive a boost.

        This method does NOT generate new candidates.
        """

        if not track_ids:
            return []

        candidate_count = len(
            track_ids
        )

        scored = []

        for rank, track_id in enumerate(
            track_ids,
            start=1,
        ):

            # Original ranking contribution.
            base_score = (
                candidate_count
                - rank
                + 1
            ) / candidate_count

            exact_score = (
                self.score_track(
                    query,
                    track_id,
                )
            )

            final_score = (
                base_score
                + exact_weight
                * exact_score
            )

            scored.append(
                (
                    track_id,
                    final_score,
                    rank,
                    exact_score,
                )
            )

        scored.sort(
            key=lambda x: (
                -x[1],
                x[2],
            )
        )

        ranked = [
            track_id
            for (
                track_id,
                _,
                _,
                _,
            ) in scored
        ]

        if topk is not None:
            ranked = ranked[:topk]

        return ranked

    def debug_scores(
        self,
        query: str,
        track_ids: list[str],
    ) -> list[dict]:
        """Return matching details for debugging."""

        results = []

        for rank, track_id in enumerate(
            track_ids,
            start=1,
        ):

            try:
                metadata = (
                    self.catalog.id_to_metadata(
                        track_id
                    )
                )

            except KeyError:
                continue

            results.append(
                {
                    "rank": rank,
                    "track_id": track_id,
                    "track_name":
                        metadata.get(
                            "track_name"
                        ),
                    "artist_name":
                        metadata.get(
                            "artist_name"
                        ),
                    "album_name":
                        metadata.get(
                            "album_name"
                        ),
                    "exact_score":
                        self.score_track(
                            query,
                            track_id,
                        ),
                }
            )

        results.sort(
            key=lambda x: (
                -x["exact_score"],
                x["rank"],
            )
        )

        return results
