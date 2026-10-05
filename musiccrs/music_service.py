"""Shared catalog and playlist services for the conversational agent.

The service is process-shared and contains only catalog/retrieval data. Mutable
playlist and pending-choice data lives in ``ConversationState`` on each agent
instance, which DialogueKit creates for each connected user.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import math
import re
from threading import Lock
from typing import Any

from retrieval.bm25 import BM25Retriever
from retrieval.data_loader import MusicCatalogLoader


_INDEX_FIELDS = ["track_name", "artist_name", "album_name", "tag_list"]
_TOKEN_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class ConversationState:
    """Mutable state belonging to one connected MusicCRS agent."""

    playlist_ids: list[str] = field(default_factory=list)
    pending_disambiguation_ids: list[str] = field(default_factory=list)
    pending_disambiguation_action: str | None = None
    pending_recommendation_ids: list[str] = field(default_factory=list)


def _values(value: Any) -> list[str]:
    """Return a metadata value as clean strings (catalog fields may be lists)."""
    if value is None:
        return []
    raw_values = value if isinstance(value, (list, tuple)) else [value]
    return [
        str(item).strip()
        for item in raw_values
        if item is not None and str(item).strip()
    ]


def _normalized(value: str) -> str:
    return " ".join(_TOKEN_RE.sub(" ", value.casefold()).split())


def _match_score(query: str, values: list[str]) -> float:
    """Score an exact, contained, or typo-tolerant match in one metadata field."""
    normalized_query = _normalized(query)
    if not normalized_query:
        return 0.0

    best = 0.0
    for value in values:
        normalized_value = _normalized(value)
        if not normalized_value:
            continue
        if normalized_query == normalized_value:
            score = 1.0
        elif normalized_query in normalized_value:
            score = 0.9 + 0.09 * len(normalized_query) / len(normalized_value)
        elif normalized_value in normalized_query:
            score = 0.72 + 0.17 * len(normalized_value) / len(normalized_query)
        else:
            score = SequenceMatcher(None, normalized_query, normalized_value).ratio()
        best = max(best, score)
    return best


def _popularity(metadata: dict) -> float:
    try:
        value = float(metadata.get("popularity") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


class MusicService:
    """Catalog lookup, candidate retrieval, and validated playlist operations."""

    def __init__(
        self,
        catalog: MusicCatalogLoader | None = None,
        retriever: BM25Retriever | None = None,
    ) -> None:
        self.catalog = catalog or MusicCatalogLoader()
        self._retriever = retriever
        self._retriever_lock = Lock()

    @staticmethod
    def display_value(value: Any) -> str:
        """Format a scalar or list-valued metadata field for a user."""
        return ", ".join(_values(value))

    def display_track(self, track: dict) -> str:
        """Format a catalog record as ``title — artist`` with optional album."""
        title = self.display_value(track.get("track_name")) or "Unknown title"
        artist = self.display_value(track.get("artist_name")) or "Unknown artist"
        album = self.display_value(track.get("album_name"))
        rendered = f"{title} — {artist}"
        return f"{rendered} ({album})" if album else rendered

    def display_record(self, track_id: str) -> dict | None:
        """Return a UI-friendly record with list-valued names formatted."""
        metadata = self.get_track(track_id)
        if metadata is None:
            return None
        record = {"track_id": track_id}
        for field_name in ("track_name", "artist_name", "album_name", "release_date"):
            value = metadata.get(field_name)
            if value is not None:
                record[field_name] = self.display_value(value)
        tags = _values(metadata.get("tag_list"))
        if tags:
            record["tag_list"] = tags
        if metadata.get("popularity") is not None:
            record["popularity"] = metadata["popularity"]
        return record

    def get_track(self, track_id: str) -> dict | None:
        return self.catalog.metadata_dict.get(track_id)

    def search_tracks(
        self,
        title: str | None = None,
        artist: str | None = None,
        query: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search the catalog and rank matches by name similarity then popularity.

        ``title`` and ``artist`` constrain their corresponding fields. A free
        ``query`` searches track, artist, album, and tag text, with track-title
        similarity receiving the strongest weight.
        """
        if limit <= 0:
            return []

        title = title.strip() if title else None
        artist = artist.strip() if artist else None
        query = query.strip() if query else None
        if not any((title, artist, query)):
            return []

        ranked: list[tuple[tuple[float, ...], dict]] = []
        for metadata in self.catalog.metadata_dict.values():
            title_score = (
                _match_score(title, _values(metadata.get("track_name")))
                if title
                else 0.0
            )
            artist_score = (
                _match_score(artist, _values(metadata.get("artist_name")))
                if artist
                else 0.0
            )

            if title and title_score < 0.42:
                continue
            if artist and artist_score < 0.42:
                continue

            query_title_score = 0.0
            query_artist_score = 0.0
            query_other_score = 0.0
            if query:
                query_title_score = _match_score(
                    query,
                    _values(metadata.get("track_name")),
                )
                query_artist_score = _match_score(
                    query,
                    _values(metadata.get("artist_name")),
                )
                other_values = (
                    _values(metadata.get("album_name"))
                    + _values(metadata.get("tag_list"))
                )
                query_other_score = _match_score(query, other_values)
                if max(query_title_score, query_artist_score, query_other_score) < 0.38:
                    continue

            rank = (
                title_score,
                artist_score,
                query_title_score,
                query_artist_score,
                query_other_score,
                _popularity(metadata),
            )
            ranked.append((rank, metadata))

        ranked.sort(key=lambda item: tuple(-score for score in item[0]))
        return [metadata for _, metadata in ranked[:limit]]

    def _get_retriever(self) -> BM25Retriever:
        if self._retriever is None:
            with self._retriever_lock:
                if self._retriever is None:
                    self._retriever = BM25Retriever(
                        corpus_types=_INDEX_FIELDS,
                        catalog=self.catalog,
                    )
        return self._retriever

    def recommend_tracks(
        self,
        query: str,
        exclude_ids: set[str] | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """Retrieve related catalog tracks with BM25 and remove excluded IDs."""
        if limit <= 0 or not query.strip():
            return []
        excluded = exclude_ids or set()
        retriever = self._get_retriever()
        candidate_count = min(
            len(self.catalog.metadata_dict),
            max(limit * 8 + len(excluded), 40),
        )
        candidate_ids = retriever.text_to_item_retrieval(query, candidate_count)

        tracks = []
        for track_id in candidate_ids:
            if track_id in excluded:
                continue
            metadata = self.get_track(track_id)
            if metadata is not None:
                tracks.append(metadata)
                if len(tracks) == limit:
                    break
        return tracks

    def recommend_from_playlist(
        self,
        state: ConversationState,
        limit: int = 5,
    ) -> list[dict]:
        """Build a compact catalog query from the playlist and retrieve matches."""
        seeds = [self.get_track(track_id) for track_id in state.playlist_ids]
        seeds = [track for track in seeds if track is not None]
        if not seeds or limit <= 0:
            return []

        titles: list[str] = []
        artists: list[str] = []
        tag_counts: Counter[str] = Counter()
        for track in seeds[-5:]:
            titles.extend(_values(track.get("track_name"))[:1])
            artists.extend(_values(track.get("artist_name"))[:2])
            tag_counts.update(tag.casefold() for tag in _values(track.get("tag_list")))

        common_tags = [tag for tag, _ in tag_counts.most_common(8)]
        query_parts = []
        if common_tags:
            query_parts.append("music styles: " + ", ".join(common_tags))
        if artists:
            query_parts.append("artists: " + ", ".join(dict.fromkeys(artists)))
        if titles:
            query_parts.append("tracks similar to: " + ", ".join(dict.fromkeys(titles)))

        return self.recommend_tracks(
            "; ".join(query_parts),
            exclude_ids=set(state.playlist_ids),
            limit=limit,
        )

    def get_playlist(self, state: ConversationState) -> list[str]:
        return list(state.playlist_ids)

    def get_playlist_records(self, state: ConversationState) -> list[dict]:
        return [
            record
            for track_id in state.playlist_ids
            if (record := self.display_record(track_id)) is not None
        ]

    def _validated_unique_ids(self, track_ids: list[str]) -> list[str]:
        unique_ids = list(dict.fromkeys(track_ids))
        unknown_ids = [
            track_id
            for track_id in unique_ids
            if track_id not in self.catalog.metadata_dict
        ]
        if unknown_ids:
            raise ValueError(f"Unknown track ID(s): {', '.join(unknown_ids)}")
        return unique_ids

    def add_tracks(self, state: ConversationState, track_ids: list[str]) -> list[str]:
        additions = self._validated_unique_ids(track_ids)
        state.playlist_ids = list(dict.fromkeys(state.playlist_ids + additions))
        return self.get_playlist(state)

    def remove_tracks(self, state: ConversationState, track_ids: list[str]) -> list[str]:
        removals = set(self._validated_unique_ids(track_ids))
        state.playlist_ids = [
            track_id
            for track_id in state.playlist_ids
            if track_id not in removals
        ]
        return self.get_playlist(state)

    def replace_playlist(self, state: ConversationState, track_ids: list[str]) -> list[str]:
        state.playlist_ids = self._validated_unique_ids(track_ids)
        return self.get_playlist(state)

    def clear_playlist(self, state: ConversationState) -> list[str]:
        state.playlist_ids.clear()
        return self.get_playlist(state)


_shared_service: MusicService | None = None
_shared_service_lock = Lock()


def get_shared_music_service() -> MusicService:
    """Return one read-only catalog/retrieval service for this server process."""
    global _shared_service
    if _shared_service is None:
        with _shared_service_lock:
            if _shared_service is None:
                _shared_service = MusicService()
    return _shared_service
