"""Shared catalog and playlist services for the conversational agent.

The service is process-shared and contains only catalog/retrieval data. Mutable
playlist and pending-choice data lives in ``ConversationState`` on each agent
instance, which DialogueKit creates for each connected user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import math
import re
from threading import Lock
from typing import Any

from retrieval.base import RetrievalModule
from retrieval.data_loader import MusicCatalogLoader
from retrieval.final import FinalRetriever
from musiccrs.music_features import extract_preferences, profile_tracks, select_diverse


_TOKEN_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class ConversationState:
    """Mutable state belonging to one connected MusicCRS agent."""

    playlist_ids: list[str] = field(default_factory=list)
    pending_disambiguation_ids: list[str] = field(default_factory=list)
    pending_disambiguation_action: str | None = None
    pending_recommendation_ids: list[str] = field(default_factory=list)
    recent_retrieval_requests: list[str] = field(default_factory=list)
    last_question_track_id: str | None = None
    last_question_artist: str | None = None


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
        retriever: RetrievalModule | None = None,
    ) -> None:
        self.catalog = catalog or MusicCatalogLoader()
        self._retriever = retriever
        self._retriever_lock = Lock()
        self._r5_entities = None
        self._r5_entities_lock = Lock()

    def get_r5_entities(self):
        """Build the read-only name index once for all connected users."""
        if self._r5_entities is None:
            with self._r5_entities_lock:
                if self._r5_entities is None:
                    from musiccrs.r5_nlu import CatalogEntities
                    self._r5_entities = CatalogEntities(self.catalog)
        return self._r5_entities

    def warm_r5(self):
        from musiccrs.r5_nlu import get_r5_router
        self.get_r5_entities()
        get_r5_router()

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

    def _get_retriever(self) -> RetrievalModule:
        if self._retriever is None:
            with self._retriever_lock:
                if self._retriever is None:
                    self._retriever = FinalRetriever(catalog=self.catalog)
        return self._retriever

    def recommend_tracks(
        self,
        query: str,
        exclude_ids: set[str] | None = None,
        limit: int = 5,
        *,
        state: ConversationState | None = None,
        request: str | None = None,
        new_playlist: bool = False,
        use_playlist_context: bool = True,
    ) -> list[dict]:
        """Run final G1 retrieval and filter playlist exclusions after ranking.

        Live R6/R8 requests use LL (discovery), since chat has no dataset goal
        labels. New playlist descriptions start a fresh retrieval context.
        """
        if limit <= 0 or not query.strip():
            return []
        excluded = exclude_ids or set()
        retriever = self._get_retriever()
        request = request or query
        if isinstance(retriever, FinalRetriever):
            history = [] if state is None or new_playlist else state.recent_retrieval_requests[-2:]
            playlist_ids = (
                [] if state is None or new_playlist or not use_playlist_context
                else state.playlist_ids[-2:]
            )
            conversations = [
                {"role": "user", "content": text, "turn_number": index}
                for index, text in enumerate(history, start=1)
            ]
            conversations.extend(
                {"role": "music", "content": track_id, "turn_number": index}
                for index, track_id in enumerate(playlist_ids, start=1)
            )
            conversations.append({"role": "user", "content": request, "turn_number": 3})
            candidate_ids = retriever.retrieve_context({
                "conversations": conversations,
                "conversation_goal": {"specificity": "LL", "listener_goal": query},
            }, 3, topk=20)
        else:
            # Explicitly injected retrievers are used by focused unit tests.
            candidate_ids = retriever.text_to_item_retrieval(query, min(20, len(self.catalog.metadata_dict)))

        tracks = []
        for track_id in dict.fromkeys(candidate_ids):
            if track_id in excluded:
                continue
            metadata = self.get_track(track_id)
            if metadata is not None:
                tracks.append(metadata)
                if len(tracks) == limit:
                    break
        if tracks and state is not None:
            previous = [] if new_playlist else state.recent_retrieval_requests
            state.recent_retrieval_requests = (previous + [request])[-2:]
        return tracks

    def recommend_from_playlist(
        self,
        state: ConversationState,
        limit: int = 5,
        request: str = "Recommend something based on my playlist",
    ) -> list[dict]:
        """Build a compact catalog query from the playlist and retrieve matches."""
        seeds = [self.get_track(track_id) for track_id in state.playlist_ids]
        seeds = [track for track in seeds if track is not None]
        if not seeds or limit <= 0:
            return []

        titles: list[str] = []
        artists: list[str] = []
        recent_seeds = seeds[-5:]
        genre_profile, mood_profile = profile_tracks(seeds)
        requested_genres, requested_moods = extract_preferences(request)
        genre_profile.update(requested_genres)
        mood_profile.update(requested_moods)
        for track in recent_seeds:
            titles.extend(_values(track.get("track_name"))[:1])
            artists.extend(_values(track.get("artist_name"))[:2])

        query_parts = []
        if genre_profile:
            query_parts.append("preferred genres: " + ", ".join(
                genre for genre, _ in genre_profile.most_common(5)
            ))
        if mood_profile:
            query_parts.append("preferred moods: " + ", ".join(
                mood for mood, _ in mood_profile.most_common(4)
            ))
        # When tags provide a usable preference profile, lead with those
        # features so one seed artist does not dominate the candidate pool.
        if not genre_profile and not mood_profile:
            if artists:
                query_parts.append("artists: " + ", ".join(dict.fromkeys(artists)))
            if titles:
                query_parts.append("tracks similar to: " + ", ".join(dict.fromkeys(titles)))

        candidates = self.recommend_tracks(
            "; ".join(query_parts),
            exclude_ids=set(state.playlist_ids),
            limit=min(20, len(self.catalog.metadata_dict)),
            state=state,
            request=request,
            use_playlist_context=False,
        )
        return select_diverse(candidates, limit, genre_profile, mood_profile)

    def recommend_for_playlist_description(
        self,
        query: str,
        limit: int = 20,
        request: str | None = None,
        state: ConversationState | None = None,
    ) -> list[dict]:
        """Retrieve and preference-rerank candidates for a new playlist."""
        genres, moods = extract_preferences(query)
        candidates = self.recommend_tracks(
            query,
            limit=min(20, len(self.catalog.metadata_dict)),
            state=state,
            request=request or query,
            new_playlist=True,
        )
        return select_diverse(candidates, limit, genres, moods)

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
        state.pending_recommendation_ids.clear()
        state.pending_disambiguation_ids.clear()
        state.pending_disambiguation_action = None
        state.recent_retrieval_requests.clear()
        state.last_question_track_id = None
        state.last_question_artist = None
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
