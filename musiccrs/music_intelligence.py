"""Natural-language handlers for track questions and music suggestions."""

from __future__ import annotations

import re

from musiccrs.music_service import ConversationState, MusicService


_WORD_NUMBERS = {
    "one": 1,
    "first": 1,
    "1st": 1,
    "two": 2,
    "second": 2,
    "2nd": 2,
    "three": 3,
    "third": 3,
    "3rd": 3,
    "four": 4,
    "fourth": 4,
    "4th": 4,
    "five": 5,
    "fifth": 5,
    "5th": 5,
    "last": -1,
}


def _clean_entity(text: str) -> str:
    text = text.strip().strip(" \t\r\n\"'“”‘’.,!?;:")
    text = re.sub(r"^(?:the song|song|track)\s+", "", text, flags=re.I)
    text = re.sub(r"\s+(?:song|track)$", "", text, flags=re.I)
    return text.strip()


def _exact_title_matches(query: str, tracks: list[dict], service: MusicService) -> list[dict]:
    normalized_query = service.display_value(query).casefold().strip()
    exact = [
        track
        for track in tracks
        if any(
            value.casefold().strip() == normalized_query
            for value in service.display_value(track.get("track_name")).split(", ")
        )
    ]
    return exact or tracks

def _reasonable_title_matches(
    query: str,
    tracks: list[dict],
    service: MusicService,
) -> list[dict]:
    """Reject obviously weak fuzzy matches while preserving useful loose references."""
    normalized_query = re.sub(r"[^a-z0-9]+", "", query.casefold())
    if not normalized_query:
        return []

    reasonable: list[dict] = []
    for track in tracks:
        for title in service.display_value(track.get("track_name")).split(", "):
            normalized_title = re.sub(r"[^a-z0-9]+", "", title.casefold())
            if not normalized_title:
                continue

            # Exact after ignoring case/punctuation.
            if normalized_query == normalized_title:
                reasonable.append(track)
                break

            # Permit small spelling differences, but not unrelated fragments
            # such as ABC / 123 / Son for "xyzabc123notasong".
            max_len = max(len(normalized_query), len(normalized_title))
            common_prefix = 0
            for left, right in zip(normalized_query, normalized_title):
                if left != right:
                    break
                common_prefix += 1

            # A close length plus a meaningful shared prefix is a conservative
            # loose-reference allowance.
            if (
                max_len >= 4
                and abs(len(normalized_query) - len(normalized_title)) <= 2
                and common_prefix >= max(3, min(len(normalized_query), len(normalized_title)) - 2)
            ):
                reasonable.append(track)
                break

    return reasonable



class MusicIntelligence:
    """Route MusicCRS natural-language requests for R2–R8."""

    def __init__(self, service: MusicService, r5_router=None) -> None:
        self.service = service
        self._r5_router = r5_router

    def handle(self, text: str, state: ConversationState) -> tuple[str, bool] | None:
        """Return response and whether the playlist changed; None if unhandled."""
        message = text.strip()
        if not message:
            return None

        # Existing pending disambiguation / recommendation selection.
        selection = self._handle_pending_selection(message, state)
        if selection is not None:
            return selection

        # R3: explain the available functionality in natural language.
        help_response = self._handle_help(message)
        if help_response is not None:
            return help_response, False

        # R2: natural-language playlist management.
        playlist_action = self._handle_playlist_action(message, state)
        if playlist_action is not None:
            return playlist_action

        # Keep conversational recommendation requests on the final G1 retriever.
        if self._is_recommendation_request(message):
            return self._recommend(message, state)

        # Existing R8: create a playlist from a description.
        if self._is_playlist_creation(message):
            return self._create_playlist(message, state)

        # R5 semantic intent routing and catalog-grounded answers.
        question = self._handle_question(message, state)
        if question is not None:
            return question, False

        return None

    def _handle_help(self, message: str) -> str | None:
        """R3: Explain MusicCRS functionality using natural language."""
        lowered = message.casefold().strip()

        help_patterns = [
            r"^(?:please\s+)?help(?:\s+me)?[.!?]*$",
            r"\bwhat can you do\b",
            r"\bhow do i use\b",
            r"\bhow can i use\b",
            r"\bwhat can i ask\b",
            r"\bwhat do you do\b",
        ]

        if not any(re.search(pattern, lowered) for pattern in help_patterns):
            return None

        return (
            "I can help you manage and explore music. You can ask me to:\n"
            "• add a song — for example, “Add Numb by Linkin Park”\n"
            "• remove a song — “Remove Numb from my playlist”\n"
            "• view your playlist — “Show my playlist”\n"
            "• clear your playlist — “Clear my playlist”\n"
            "• answer questions about tracks and artists\n"
            "• recommend songs based on your current playlist\n"
            "• create a playlist from a description, such as "
            "“Create a workout playlist”."
        )

    def _handle_playlist_action(
        self,
        message: str,
        state: ConversationState,
    ) -> tuple[str, bool] | None:
        """R2: Natural-language add/remove/view/clear playlist operations."""
        lowered = message.casefold().strip()

        # VIEW
        if re.search(
            r"\b(show|view|display|see|list|what(?:'s| is))\b.*\bplaylist\b",
            lowered,
        ):
            records = self.service.get_playlist_records(state)
            if not records:
                return "Your playlist is empty.", False

            rendered = "\n".join(
                f"{index}. {self.service.display_track(track)}"
                for index, track in enumerate(records, start=1)
            )
            return f"Your playlist:\n{rendered}", False

        # CLEAR
        if re.search(
            r"\b(clear|empty|reset|delete all|remove all)\b.*\bplaylist\b",
            lowered,
        ):
            self.service.clear_playlist(state)
            state.pending_disambiguation_ids.clear()
            state.pending_disambiguation_action = None
            state.pending_recommendation_ids.clear()
            return "Your playlist is now empty.", True

        # ADD
        add_match = re.match(
            r"^(?:please\s+)?(?:add|put|include)\s+(.+?)(?:\s+(?:to|in|into)\s+(?:my\s+|the\s+)?playlist)?[.!?]*$",
            message,
            flags=re.I,
        )
        if add_match:
            entity = _clean_entity(add_match.group(1))
            return self._add_track_from_text(entity, state)

        # REMOVE
        remove_match = re.match(
            r"^(?:please\s+)?(?:remove|delete|take out)\s+(.+?)(?:\s+from\s+(?:my\s+|the\s+)?playlist)?[.!?]*$",
            message,
            flags=re.I,
        )
        if remove_match:
            entity = _clean_entity(remove_match.group(1))
            return self._remove_track_from_text(entity, state)

        return None

    @staticmethod
    def _split_title_artist(entity: str) -> tuple[str, str | None]:
        """Split 'track by artist' while still allowing title-only requests."""
        match = re.match(r"^(.+?)\s+by\s+(.+)$", entity, flags=re.I)
        if match:
            return _clean_entity(match.group(1)), _clean_entity(match.group(2))
        return _clean_entity(entity), None

    def _remember_track(self, track_id: str, state: ConversationState) -> None:
        track = self.service.get_track(track_id)
        if track:
            state.last_question_track_id = track_id
            state.last_question_artist = self.service.display_value(track.get("artist_name"))

    def _add_track_from_text(
        self,
        entity: str,
        state: ConversationState,
    ) -> tuple[str, bool]:
        """Find a catalog track and add it, or ask for disambiguation."""
        title, artist = self._split_title_artist(entity)
        matches = self.service.search_tracks(
            title=title,
            artist=artist,
            limit=10,
        )
        matches = _reasonable_title_matches(title, matches, self.service)
        matches = _exact_title_matches(title, matches, self.service)

        if not matches:
            return (
                f"I couldn't find “{entity}” in the music catalog, "
                "so I didn't add anything.",
                False,
            )

        # A supplied artist normally makes the request specific enough.
        if artist:
            track = matches[0]
            track_id = track["track_id"]
            self.service.add_tracks(state, [track_id])
            self._remember_track(track_id, state)
            return (
                f"Added {self.service.display_track(track)} to your playlist.",
                True,
            )

        if len(matches) == 1:
            track = matches[0]
            track_id = track["track_id"]
            self.service.add_tracks(state, [track_id])
            self._remember_track(track_id, state)
            return (
                f"Added {self.service.display_track(track)} to your playlist.",
                True,
            )

        # R4: multiple same-title/fuzzy candidates. search_tracks() already
        # ranks them by similarity and popularity.
        choices = matches[:5]
        state.pending_disambiguation_ids = [
            track["track_id"] for track in choices
        ]
        state.pending_disambiguation_action = "add"
        state.pending_recommendation_ids.clear()

        lines = [
            f"{index}. {self.service.display_track(track)}"
            for index, track in enumerate(choices, start=1)
        ]
        return (
            f"I found several matches for “{title}”. Which one did you mean?\n"
            + "\n".join(lines)
            + "\nReply with a number, for example “1” or “the second one”.",
            False,
        )

    def _remove_track_from_text(
        self,
        entity: str,
        state: ConversationState,
    ) -> tuple[str, bool]:
        """Remove a matching track from the current playlist."""
        title, artist = self._split_title_artist(entity)

        if not state.playlist_ids:
            return "Your playlist is already empty.", False

        matches = self.service.search_tracks(
            title=title,
            artist=artist,
            limit=20,
        )
        playlist_ids = set(state.playlist_ids)
        matches = [
            track for track in matches
            if track["track_id"] in playlist_ids
        ]
        matches = _exact_title_matches(title, matches, self.service)

        if not matches:
            return (
                f"I couldn't find “{entity}” in your current playlist.",
                False,
            )

        if artist or len(matches) == 1:
            track = matches[0]
            track_id = track["track_id"]
            self.service.remove_tracks(state, [track_id])
            return (
                f"Removed {self.service.display_track(track)} from your playlist.",
                True,
            )

        choices = matches[:5]
        state.pending_disambiguation_ids = [
            track["track_id"] for track in choices
        ]
        state.pending_disambiguation_action = "remove"
        state.pending_recommendation_ids.clear()

        lines = [
            f"{index}. {self.service.display_track(track)}"
            for index, track in enumerate(choices, start=1)
        ]
        return (
            f"I found several matching tracks in your playlist. "
            f"Which “{title}” do you want to remove?\n"
            + "\n".join(lines)
            + "\nReply with a number.",
            False,
        )

    def _handle_pending_selection(
        self,
        message: str,
        state: ConversationState,
    ) -> tuple[str, bool] | None:
        lowered = message.casefold().strip()

        if state.pending_disambiguation_ids:
            selected = self._selected_positions(message, len(state.pending_disambiguation_ids))
            if selected is not None:
                if len(selected) != 1:
                    return "Please choose one option number for this track.", False
                track_id = state.pending_disambiguation_ids[selected[0]]
                action = state.pending_disambiguation_action
                state.pending_disambiguation_ids.clear()
                state.pending_disambiguation_action = None
                if action in {"add", "add_track", "ADD_TRACK"}:
                    self.service.add_tracks(state, [track_id])
                    track = self.service.get_track(track_id)
                    self._remember_track(track_id, state)
                    return f"Added {self.service.display_track(track)} to your playlist.", True
                if action in {"remove", "remove_track", "REMOVE_TRACK"}:
                    self.service.remove_tracks(state, [track_id])
                    track = self.service.get_track(track_id)
                    return f"Removed {self.service.display_track(track)} from your playlist.", True
                state.last_question_track_id = track_id
                track = self.service.get_track(track_id)
                state.last_question_artist = (
                    self.service.display_value(track.get("artist_name")) if track else None
                )
                return self._answer_track_action(track_id, action), False
            return None

        pending = state.pending_recommendation_ids
        if not pending:
            return None

        if re.search(r"\b(no|none|neither|don't|do not|not any|skip them)\b", lowered):
            state.pending_recommendation_ids.clear()
            return "Okay, I won't add any of those recommendations.", False

        except_match = re.search(
            r"\bexcept\s+(?:for\s+)?(?:the\s+)?(?:one\s+)?(?:by\s+)?(.+)$",
            message,
            flags=re.I,
        )
        if except_match:
            excluded_text = _clean_entity(except_match.group(1))
            excluded_ids = self._match_pending_description(excluded_text, pending)
            if len(excluded_ids) != 1:
                return (
                    "I couldn't identify exactly one recommendation to leave out. "
                    "Please name its artist or title.",
                    False,
                )
            selected_ids = [track_id for track_id in pending if track_id not in excluded_ids]
        elif re.search(r"\b(all of them|add all|add them all|all recommendations|everything)\b", lowered):
            selected_ids = list(pending)
        else:
            positions = self._selected_positions(message, len(pending))
            if positions is not None:
                selected_ids = [pending[index] for index in positions]
            else:
                artist_match = re.search(
                    r"\b(?:add|choose|select)\s+(?:the\s+)?(?:one\s+)?by\s+(.+)$",
                    message,
                    re.I,
                )
                if not artist_match:
                    return None
                matching = self._match_pending_description(
                    _clean_entity(artist_match.group(1)),
                    pending,
                    artist_only=True,
                )
                if len(matching) != 1:
                    return (
                        "I couldn't identify exactly one recommendation by that artist. "
                        "Please choose by number or give the song title.",
                        False,
                    )
                selected_ids = matching

        if not selected_ids:
            state.pending_recommendation_ids.clear()
            return "There are no recommendations left to add.", False

        playlist = self.service.add_tracks(state, selected_ids)
        state.pending_recommendation_ids.clear()
        added = [self.service.display_track(self.service.get_track(track_id)) for track_id in selected_ids]
        return (
            f"Added {len(selected_ids)} song(s): " + "; ".join(added)
            + f". Your playlist now has {len(playlist)} song(s).",
            True,
        )

    @staticmethod
    def _selected_positions(message: str, count: int) -> list[int] | None:
        lowered = message.casefold().strip()

        # Preserve every explicit position in selections such as "Add 1 and 3".
        number_list = re.fullmatch(
            r"(?:(?:add|choose|select|take)\s+)?(?:options?\s+|numbers?\s+)?"
            r"(\d+(?:\s*(?:,\s*(?:and\s+)?|and\s+)\d+)+)[.!]?",
            lowered,
        )
        if number_list:
            positions = list(dict.fromkeys(int(n) - 1 for n in re.findall(r"\d+", number_list.group(1))))
            return positions if all(0 <= p < count for p in positions) else None

        if (
            not re.search(
                r"\b(add|choose|select|take|want|option|number|first|second|third|fourth|fifth|last)\b",
                lowered,
            )
            and not re.fullmatch(r"\s*\d+\s*", lowered)
        ):
            return None

        # Multi-selection expressions used for recommendation follow-ups.
        if re.search(r"\bfirst\s+(?:two|2)\b", lowered):
            return list(range(min(2, count)))

        if re.search(r"\bfirst\s+(?:three|3)\b", lowered):
            return list(range(min(3, count)))

        first_number = re.search(r"\bfirst\s+(\d+)\b", lowered)
        if first_number:
            number = int(first_number.group(1))
            return list(range(min(number, count)))

        # Resolve ordinal phrases before generic words such as "one".
        ordinal_words = {
            "first": 0,
            "second": 1,
            "third": 2,
            "fourth": 3,
            "fifth": 4,
        }

        for word, index in ordinal_words.items():
            if re.search(rf"\b{word}\b", lowered):
                if index < count:
                    return [index]

        if re.search(r"\blast\b", lowered):
            return [count - 1] if count else None

        # Numeric forms: "2", "number 2", "option 2", "2nd".
        numeric = re.search(
            r"\b(?:number\s+|option\s+)?(\d+)(?:st|nd|rd|th)?\b",
            lowered,
        )
        if numeric:
            index = int(numeric.group(1)) - 1
            if 0 <= index < count:
                return [index]

        return None

    def _match_pending_description(
        self,
        description: str,
        pending: list[str],
        artist_only: bool = False,
    ) -> list[str]:
        needle = description.casefold()
        matches = []
        for track_id in pending:
            metadata = self.service.get_track(track_id)
            if metadata is None:
                continue
            fields = [metadata.get("artist_name")] if artist_only else [
                metadata.get("artist_name"),
                metadata.get("track_name"),
            ]
            values = [value.casefold() for field in fields for value in self.service.display_value(field).split(", ")]
            if any(needle in value for value in values if needle):
                matches.append(track_id)
        return matches

    def _handle_question(self, message: str, state: ConversationState) -> str | None:
        from musiccrs.r5_nlu import get_r5_router, normalize

        entities = self.service.get_r5_entities()
        title_mentions = entities.mentions(message, "title")
        artist_mentions = entities.mentions(message, "artist")
        album_mentions = entities.mentions(message, "album")
        masked_message = entities.mask(message, title_mentions, artist_mentions, album_mentions)
        router = self._r5_router or get_r5_router()
        prediction = router.predict(masked_message)
        if not prediction.accepted:
            return None

        # Spend fuzzy matching only on messages already classified as an R5
        # lookup. Ordinary chat and playlist actions use the fast exact index.
        if prediction.intent == "artist_tracks" and not artist_mentions:
            artist_mentions = entities.mentions(message, "artist", fuzzy=True)
        elif prediction.intent == "album_tracks" and not album_mentions:
            album_mentions = entities.mentions(message, "album", fuzzy=True)
        elif prediction.intent in {"artist", "album", "release", "tags"}:
            if not title_mentions:
                title_mentions = entities.mentions(message, "title", fuzzy=True)
            if not artist_mentions:
                artist_mentions = entities.mentions(message, "artist", fuzzy=True)
        if title_mentions or artist_mentions or album_mentions:
            masked_message = entities.mask(message, title_mentions, artist_mentions, album_mentions)
            prediction = router.predict(masked_message)
            if not prediction.accepted:
                return None

        if prediction.intent == "album_tracks":
            album_name = None
            album_ids = None
            if album_mentions:
                album_ids = set(album_mentions[0].ids)
                album_names = {}
                for candidate_id in album_ids:
                    candidate = self.service.get_track(candidate_id)
                    if candidate:
                        for value in self.service.display_value(candidate.get("album_name")).split(", "):
                            if value:
                                album_names[normalize(value)] = value
                if len(album_names) != 1:
                    return "Which album do you mean? Please name the album or a track on it."
                album_name = next(iter(album_names.values()))
            else:
                seed_id = None
                if title_mentions:
                    title_ids = {track_id for mention in title_mentions for track_id in mention.ids}
                    if state.last_question_track_id in title_ids:
                        seed_id = state.last_question_track_id
                    elif len(title_ids) == 1:
                        seed_id = next(iter(title_ids))
                    elif title_ids:
                        return "Which track do you mean? Please name the artist or choose a track first."
                if seed_id is None:
                    seed_id = state.last_question_track_id
                seed = self.service.get_track(seed_id) if seed_id else None
                if not seed or not self.service.display_value(seed.get("album_name")):
                    return "Which album or track should I use to find other songs?"
                album_name = self.service.display_value(seed.get("album_name")).split(", ")[0]
                album_ids = {
                    track_id for track_id, row in self.service.catalog.metadata_dict.items()
                    if normalize(album_name) in {
                        normalize(value) for value in self.service.display_value(row.get("album_name")).split(", ")
                    }
                }

            album_ids = album_ids or set()
            results = [
                row for track_id, row in self.service.catalog.metadata_dict.items()
                if track_id in album_ids
            ]
            if not results:
                return f"I couldn't find tracks from {album_name} in the catalog."
            if re.search(r"\b(other|else|more)\b", message, re.I):
                excluded = set(state.playlist_ids)
                if state.last_question_track_id:
                    excluded.add(state.last_question_track_id)
                results = [track for track in results if track["track_id"] not in excluded]
            if not results:
                return f"There are no other tracks from {album_name} beyond the ones already in context."
            rendered = [self.service.display_track(track) for track in results[:10]]
            return f"Tracks from {album_name}: " + "; ".join(rendered) + "."

        if prediction.intent == "artist_tracks":
            artist = artist_mentions[0] if artist_mentions else None
            artist_name = artist.name if artist else state.last_question_artist
            if not artist_name:
                return "Which artist do you mean?"
            exact_artist_ids = set(artist.ids) if artist else None
            results = self.service.search_tracks(artist=artist_name, limit=100)
            results = [
                track for track in results
                if (exact_artist_ids is None or track["track_id"] in exact_artist_ids)
                and self.service.display_value(track.get("artist_name")).casefold().find(artist_name.casefold()) >= 0
            ]
            if not results:
                return f"I couldn't find tracks by {artist_name} in the catalog."
            if "other" in masked_message.casefold() or "else" in masked_message.casefold():
                excluded = set(state.playlist_ids)
                if state.last_question_track_id:
                    excluded.add(state.last_question_track_id)
                results = [track for track in results if track["track_id"] not in excluded]
            if not results:
                return f"I couldn't find other tracks by {artist_name} beyond the ones already in context."
            state.last_question_artist = artist_name
            rendered = [self.service.display_track(track) for track in results[:5]]
            return f"Here are tracks by {artist_name} in the catalog: " + "; ".join(rendered) + "."

        if prediction.intent not in {"artist", "album", "release", "tags"}:
            return None
        action = prediction.intent
        title = title_mentions[0] if title_mentions else None
        artist = artist_mentions[0] if artist_mentions else None

        # Resolve explicit “that/it/this song” references against the last
        # track discussed by this user; never infer a silent playlist change.
        if title is None and re.search(r"\b(that|it|this song|this track|the song|the track)\b", message, re.I):
            track_id = state.last_question_track_id
            if track_id and self.service.get_track(track_id):
                state.pending_disambiguation_ids = []
                state.pending_disambiguation_action = None
                state.last_question_artist = self.service.display_value(
                    self.service.get_track(track_id).get("artist_name")
                )
                return self._answer_track_action(track_id, action)

        if title is None:
            return "Which song or track should I look up?"
        candidates = [self.service.get_track(track_id) for track_id in title.ids]
        candidates = [track for track in candidates if track]
        if artist:
            candidates = [track for track in candidates if track["track_id"] in artist.ids]
        elif state.last_question_track_id in title.ids:
            # A repeated title inherits its artist only when the immediately
            # preceding R5 lookup identified that exact same catalog title.
            previous = self.service.get_track(state.last_question_track_id)
            if previous and normalize(self.service.display_value(previous.get("track_name"))) == normalize(title.name):
                candidates = [previous]
        if not candidates:
            return f"I couldn't find a catalog track matching {title.name}."
        if len(candidates) > 1:
            choices = candidates[:5]
            state.pending_disambiguation_ids = [track["track_id"] for track in choices]
            state.pending_disambiguation_action = action
            state.pending_recommendation_ids.clear()
            lines = [f"{i}. {self.service.display_track(track)}" for i, track in enumerate(choices, 1)]
            return "Which track did you mean? Reply with its number.\n" + "\n".join(lines)
        track_id = candidates[0]["track_id"]
        state.last_question_track_id = track_id
        state.last_question_artist = self.service.display_value(candidates[0].get("artist_name"))
        return self._answer_track_action(track_id, action)

    def _resolve_track_question(
        self,
        entity: str,
        action: str,
        state: ConversationState,
    ) -> str:
        artist = None
        by_split = re.match(r"^(.+?)\s+by\s+(.+)$", entity, flags=re.I)
        if by_split:
            entity = _clean_entity(by_split.group(1))
            artist = _clean_entity(by_split.group(2))

        matches = self.service.search_tracks(title=entity, artist=artist, limit=10)
        matches = _exact_title_matches(entity, matches, self.service)
        if not matches:
            return f"I couldn't find ‘{entity}’ in the track catalog."
        if len(matches) > 1:
            # Exact same-title variants must be disambiguated; fuzzy variants
            # are already filtered where an exact title exists.
            choices = matches[:5]
            state.pending_disambiguation_ids = [track["track_id"] for track in choices]
            state.pending_disambiguation_action = action
            state.pending_recommendation_ids.clear()
            lines = [
                f"{index}. {self.service.display_track(track)}"
                for index, track in enumerate(choices, start=1)
            ]
            return "Which track did you mean? Reply with its number.\n" + "\n".join(lines)
        return self._answer_track_action(matches[0]["track_id"], action)

    def _answer_track_action(self, track_id: str, action: str | None) -> str:
        track = self.service.get_track(track_id)
        if track is None:
            return "That track is no longer available in the catalog."
        title = self.service.display_value(track.get("track_name")) or "That track"
        if action == "artist":
            value = self.service.display_value(track.get("artist_name"))
            return f"{title} is by {value}." if value else f"The artist for {title} is not listed in the catalog."
        if action == "album":
            value = self.service.display_value(track.get("album_name"))
            return f"{title} is on the album {value}." if value else f"The album for {title} is not listed in the catalog."
        if action == "release":
            value = self.service.display_value(track.get("release_date"))
            return f"{title} was released on {value}." if value else f"The release date for {title} is not listed in the catalog."
        if action == "tags":
            value = ", ".join(
                self.service.display_value(track.get("tag_list")).split(", ")[:6]
            )
            return f"The catalog tags for {title} are {value}." if value else f"There are no style tags listed for {title}."
        return "I couldn't identify what information you wanted about that track."

    @staticmethod
    def _is_recommendation_request(message: str) -> bool:
        lowered = message.casefold()
        return message.strip() == "/recommend" or bool(
            re.search(r"\b(recommend|suggest)\b", lowered)
            and re.search(r"\b(song|songs|track|tracks|music|something|playlist)\b", lowered)
        )

    @staticmethod
    def _is_playlist_creation(message: str) -> bool:
        lowered = message.casefold()
        return bool(
            re.search(r"\b(create|make|build|generate|put together)\b", lowered)
            and re.search(r"\bplaylist\b", lowered)
        )

    def _recommend(self, message: str, state: ConversationState) -> tuple[str, bool]:
        if not state.playlist_ids:
            return "Add at least one song to your playlist first, and I can suggest related tracks.", False
        recommendations = self.service.recommend_from_playlist(state, limit=5, request=message)
        if not recommendations:
            return "I couldn't find enough related tracks in the catalog for your playlist.", False
        state.pending_recommendation_ids = [track["track_id"] for track in recommendations]
        state.pending_disambiguation_ids.clear()
        state.pending_disambiguation_action = None
        rendered = []
        seed_tags = {
            tag.casefold()
            for track_id in state.playlist_ids
            if (seed := self.service.get_track(track_id)) is not None
            for tag in self.service.display_value(seed.get("tag_list")).split(", ")
            if tag
        }
        for index, track in enumerate(recommendations, start=1):
            track_tags = {
                tag.casefold()
                for tag in self.service.display_value(track.get("tag_list")).split(", ")
                if tag
            }
            common = sorted(seed_tags & track_tags)
            if common:
                reason = f"; shares tags: {', '.join(common[:3])}"
            else:
                seed_artists = {
                    artist.casefold()
                    for seed_id in state.playlist_ids
                    if (seed := self.service.get_track(seed_id)) is not None
                    for artist in self.service.display_value(seed.get("artist_name")).split(", ")
                    if artist
                }
                candidate_artists = {
                    artist.casefold()
                    for artist in self.service.display_value(track.get("artist_name")).split(", ")
                    if artist
                }
                if seed_artists & candidate_artists:
                    reason = "; by an artist already in your playlist"
                else:
                    reason = "; matched to your playlist's track and style metadata"
            rendered.append(f"{index}. {self.service.display_track(track)}{reason}")
        count_note = ""
        if len(recommendations) < 3:
            count_note = " I found fewer than three eligible matches."
        return (
            "Here are songs related to your playlist:\n"
            + "\n".join(rendered)
            + count_note
            + "\nYou can say ‘add all’, ‘add the first two’, or ‘add them except the one by [artist]’.",
            False,
        )

    def _create_playlist(self, message: str, state: ConversationState) -> tuple[str, bool]:
        query = re.sub(
            r"\b(create|make|build|generate|put together)\b|\b(a|me|my|new|playlist|for|with|of|song|songs|music)\b",
            " ",
            message,
            flags=re.I,
        )
        query = " ".join(query.split()).strip(" .!?,-") or message
        if query == message and not re.search(r"\b(mood|genre|style|activity|energy)\b", message, re.I):
            return "What mood, genre, or activity should I use for the playlist?", False
        candidates = self.service.recommend_tracks(
            query, exclude_ids=set(), limit=20, state=state, request=message, new_playlist=True,
        )
        unique_candidates = list(
            {track["track_id"]: track for track in candidates}.values()
        )
        lowered_query = query.casefold()
        if re.search(r"\b(party|road trip|long trip)\b", lowered_query):
            target_minutes = 75
        elif re.search(r"\b(short|quick|commute)\b", lowered_query):
            target_minutes = 30
        else:
            # A 45-minute listening session is a sensible default for mood,
            # study, dinner, and workout descriptions.
            target_minutes = 45

        unique_tracks = []
        total_duration_ms = 0
        minimum_tracks = min(8, len(unique_candidates))
        for track in unique_candidates:
            if len(unique_tracks) >= 20:
                break
            unique_tracks.append(track)
            try:
                duration = float(track.get("duration") or 0)
            except (TypeError, ValueError):
                duration = 0
            if duration <= 0:
                duration = 210_000
            elif duration < 10_000:
                duration *= 1000
            total_duration_ms += duration
            if (
                len(unique_tracks) >= minimum_tracks
                and total_duration_ms >= target_minutes * 60_000
            ):
                break

        if not unique_tracks:
            return (
                f"I couldn't find enough catalog tracks for ‘{query}’, so I left your playlist unchanged.",
                False,
            )
        ids = [track["track_id"] for track in unique_tracks]
        self.service.replace_playlist(state, ids)
        state.pending_recommendation_ids.clear()
        state.pending_disambiguation_ids.clear()
        state.pending_disambiguation_action = None
        rendered = "; ".join(self.service.display_track(track) for track in unique_tracks)
        estimated_minutes = round(total_duration_ms / 60_000)
        count_note = f" It should play for about {estimated_minutes} minutes."
        return (
            f"I created a playlist for ‘{query}’ with {len(ids)} songs.{count_note}\n{rendered}",
            True,
        )
