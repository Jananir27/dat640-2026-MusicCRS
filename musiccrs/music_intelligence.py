"""Natural-language handlers for track questions and music suggestions."""

from __future__ import annotations

from collections import Counter
import re

from musiccrs.music_features import classify_tags, extract_preferences, track_style
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
    "six": 6,
    "sixth": 6,
    "seven": 7,
    "seventh": 7,
    "eight": 8,
    "eighth": 8,
    "nine": 9,
    "ninth": 9,
    "ten": 10,
    "tenth": 10,
    "last": -1,
}

_R5_QUESTION_TERMS = set(
    "a an the who what which when where how is was are were do does did made by "
    "from on of in for to me my this that it they their same song songs track "
    "tracks music artist artists album release released date year genre style "
    "tags other else more out come came sings sang sung said performs performed recorded tell show list "
    "name please anything something band singer record kind tagged drop".split()
)


def _clean_entity(text: str) -> str:
    text = text.strip().strip(" \t\r\n\"'“”‘’.,!?;:")
    text = re.sub(r"^(?:the song|song|track)\s+", "", text, flags=re.I)
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

        # Keep conversational recommendation requests on the final G1 retriever.
        if self._is_recommendation_request(message):
            return self._recommend(message, state)

        # Existing R8: create a playlist from a description.
        if self._is_playlist_creation(message):
            return self._create_playlist(message, state)

        # R2: natural-language playlist management.
        playlist_action = self._handle_playlist_action(message, state)
        if playlist_action is not None:
            return playlist_action

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
            r"\b(show|view|display|see|list)\b.*\bplaylist\b|"
            r"\bwhat(?:'s| is)\s+(?:on|in)\s+(?:my|the)\s+playlist\b",
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
            r"^(?:(?:please|could you|can you|would you|will you|"
            r"i(?:'d| would) like you to|i want you to)\s+)*(?:add|put|include)\s+"
            r"(.+?)(?:\s+(?:to|in|into)\s+(?:my\s+|the\s+)?playlist)?[.!?]*$",
            message,
            flags=re.I,
        )
        if add_match:
            entity = _clean_entity(add_match.group(1))
            return self._add_track_from_text(entity, state)

        # REMOVE
        remove_match = re.match(
            r"^(?:(?:please|could you|can you|would you|will you)\s+)*(?:remove|delete|take out)\s+"
            r"(.+?)(?:\s+from\s+(?:my\s+|the\s+)?playlist)?[.!?]*$",
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

    @staticmethod
    def _canonicalize_r5_message(message: str, state: ConversationState) -> str:
        """Resolve a few clear paraphrases against the current conversation.

        This happens before catalog entity matching. In these constructions,
        ``drop`` is a release verb and ``them`` is an artist pronoun; neither
        should shadow an exact catalog title or album alias elsewhere.
        """
        text = message.strip()

        # Polite lead-ins should not be matched as catalog titles ("Do You Know"
        # is itself a track). Leave the actual question clause for NLU.
        text = re.sub(
            r"^(?:do you know|can you tell me|could you tell me|would you happen to know)\s+",
            "",
            text,
            flags=re.I,
        )
        if state.last_question_track_id:
            text = re.sub(
                r"^(who\s+(?:sang|sings?|sung|performs?|performed|recorded|said))\s+(?:it|this|that)([?.!]*)$",
                r"\1 this song\2", text, flags=re.I,
            )
            text = re.sub(
                r"\b(?:it|that|this)\b(?=\s+(?:was|is|came|come|got)\b)",
                "this song",
                text,
                flags=re.I,
            )

        release_drop = re.match(
            r"^(when\s+did\s+)(.+?)\s+drop\s*([?.!]*)$",
            text,
            flags=re.I,
        )
        if release_drop:
            subject = release_drop.group(2).strip()
            if re.fullmatch(
                r"(?:it|this|that|this song|that song|this track|that track)",
                subject,
                re.I,
            ):
                subject = "this song"
            text = f"When was {subject} released{release_drop.group(3)}"

        # In artist-list requests, "by them" and "from them" refer to the
        # remembered artist. Other uses of Them remain eligible as catalog names.
        if state.last_question_artist and re.search(
            r"\b(?:by|from)\s+them\b", text, re.I
        ):
            text = re.sub(r"\bthem\b", "this artist", text, flags=re.I)

        # The catalog stores performers, not songwriter/producer credits. In a
        # track question, users commonly use "made" to mean "performed".
        text = re.sub(r"^who\s+made\b", "Who performs", text, count=1, flags=re.I)
        return text

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
                if not selected:
                    return f"Please choose an option from 1 to {len(state.pending_disambiguation_ids)}. I haven't changed your playlist.", False
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
                if not positions:
                    return f"Please choose recommendations from 1 to {len(pending)}. I haven't added any songs; your suggestions are still available.", False
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
        """Return positions, [] for invalid bounds, or None for other requests.

        Match the complete selection so a valid first item cannot hide an
        invalid later item, or steal a title such as "The Second Coming".
        """
        lowered = message.casefold().strip().rstrip(".!?").strip()
        lead = (
            r"(?:(?:please|could you|can you|would you|will you)\s+)*"
            r"(?:(?:add|choose|select|take|i want|i'd like|i would like)\s+)?"
        )
        tail = r"(?:\s+(?:to|in|into)\s+(?:my\s+|the\s+)?playlist)?(?:\s+please)?"
        suffix = r"(?:\s+(?:ones?|songs?|tracks?|options?))?"
        amount = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        first_n = re.fullmatch(
            lead + r"(?:the\s+)?first\s+(?P<amount>" + amount + ")" + suffix + tail,
            lowered,
        )
        if first_n:
            value = first_n.group("amount")
            number = int(value) if value.isdigit() else _WORD_NUMBERS[value]
            return list(range(number)) if 0 < number <= count else []

        number = r"(?:\d+(?:st|nd|rd|th)?|" + "|".join(_WORD_NUMBERS) + ")"
        label = r"(?:(?:the|options?|numbers?|songs?|tracks?)\s+)*"
        item = label + number + suffix
        separator = r"(?:\s*,\s*(?:and\s+)?|\s+and\s+)"
        selection = re.fullmatch(
            lead + r"(?P<items>" + item + "(?:" + separator + item + ")*)" + tail,
            lowered,
        )
        if not selection:
            return None

        positions = []
        for part in re.split(separator, selection.group("items")):
            value = re.fullmatch(label + "(?P<number>" + number + ")" + suffix, part).group("number")
            if value == "last":
                position = count - 1
            elif value in _WORD_NUMBERS:
                position = _WORD_NUMBERS[value] - 1
            else:
                position = int(re.match(r"\d+", value).group()) - 1
            if not 0 <= position < count:
                return []
            if position not in positions:
                positions.append(position)
        return positions

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

    @staticmethod
    def _match_performer_album(entities, text: str, *, fuzzy: bool = True):
        """Resolve an album slot, allowing a surrounding descriptive 'album'.

        Exact catalog names win before removing words: 'Issa Album' is a
        name, while the last word in 'Issa Album album' is a type label.
        """
        raw = _clean_entity(text)
        without_prefix = re.sub(r"^(?:the\s+)?album\s+", "", raw, flags=re.I)
        without_article = re.sub(r"^the\s+", "", raw, flags=re.I)
        variants = list(dict.fromkeys([
            raw,
            without_prefix,
            without_article,
            re.sub(r"\s+album$", "", raw, flags=re.I),
            re.sub(r"\s+album$", "", without_prefix, flags=re.I),
            re.sub(r"\s+album$", "", without_article, flags=re.I),
        ]))
        for value in variants:
            matches = entities.match_name(value, "album", fuzzy=False)
            if matches:
                return matches
        return entities.match_name(without_prefix, "album") if fuzzy else []

    def _handle_question(self, message: str, state: ConversationState) -> str | None:
        from musiccrs.r5_nlu import IntentPrediction, get_r5_router, normalize

        # This short release follow-up is both common and highly ambiguous to
        # catalog entity matching ("Drop" is a real title). Resolve it from
        # conversation state before the intent/entity router sees the verb.
        release_followup = re.fullmatch(
            r"when\s+did\s+(?:it|this|that|this song|that song|this track|that track)\s+drop\s*[?.!]*",
            message.strip(),
            flags=re.I,
        )
        if release_followup and state.last_question_track_id:
            track = self.service.get_track(state.last_question_track_id)
            if track:
                state.pending_disambiguation_ids = []
                state.pending_disambiguation_action = None
                return self._answer_track_action(state.last_question_track_id, "release")

        message = self._canonicalize_r5_message(message, state)
        entities = self.service.get_r5_entities()
        performer_target = None
        explicit_artist = None
        explicit_album = None
        album_constraint_mentions = []
        # Extract the subject of performer questions before catalog matching.
        # This supplies typed entity slots to the NLU classifier: question verbs
        # cannot become artists, nor can a misspelled artist become a song title.
        subject = re.fullmatch(
            r"(who\s+(?:sang|sings?|sung|performs?|performed|recorded|said))\s+(.+?)[.!?]*",
            message, flags=re.I,
        )
        if subject and subject.group(2).casefold() not in {
            "it", "this", "that", "this song", "that song", "this track", "that track",
        }:
            performer_target = subject.group(2).strip()
            type_prefix = re.match(r"^(?:the\s+)?(song|track|album)\s+", performer_target, re.I)
            explicit_kind = type_prefix.group(1).casefold() if type_prefix else None
            if type_prefix:
                performer_target = performer_target[type_prefix.end():]
            # A real title such as "Stand By Me" wins over splitting on "by".
            title_mentions = entities.match_name(performer_target, "title", fuzzy=False)
            album_mentions = self._match_performer_album(entities, performer_target, fuzzy=False)
            if not title_mentions and not album_mentions:
                trailing_artist = re.fullmatch(r"(.+)\s+by\s+(.+)", performer_target, re.I)
                if trailing_artist and entities.match_name(trailing_artist.group(2), "artist", fuzzy=False):
                    performer_target, explicit_artist = trailing_artist.groups()
                # Split a track's album constraint only after checking the
                # complete catalog name. Prefer a split whose album is known,
                # so 'Carry On on Issa Album' retains the complete track name.
                album_clauses = list(re.finditer(r"(?<=\s)(?:on|from|in)\s+", performer_target, re.I))
                for clause in album_clauses:
                    album_text = performer_target[clause.end():]
                    matches = self._match_performer_album(entities, album_text, fuzzy=False)
                    if matches:
                        explicit_album = album_text
                        album_constraint_mentions = matches
                        performer_target = performer_target[:clause.start()].strip()
                        break
                else:
                    if album_clauses:
                        clause = album_clauses[-1]
                        explicit_album = performer_target[clause.end():]
                        album_constraint_mentions = self._match_performer_album(entities, explicit_album)
                        performer_target = performer_target[:clause.start()].strip()
            by_clause = re.fullmatch(r"(.+)\s+by\s+(.+)", performer_target, re.I)
            if not title_mentions and not album_mentions and by_clause and not explicit_artist:
                performer_target, explicit_artist = by_clause.groups()
            if explicit_kind == "album":
                title_mentions = []
                album_mentions = album_mentions or self._match_performer_album(entities, performer_target)
            elif explicit_kind in {"song", "track"}:
                album_mentions = []
                title_mentions = title_mentions or entities.match_name(performer_target, "title")
            elif not title_mentions and not album_mentions:
                title_mentions = entities.match_name(performer_target, "title")
                if not explicit_album:
                    album_mentions = self._match_performer_album(entities, performer_target)
            artist_mentions = entities.match_name(explicit_artist, "artist") if explicit_artist else []
            entity_kind = "album" if album_mentions and not title_mentions else "song"
            masked_message = f"{subject.group(1)} this {entity_kind}"
            if explicit_artist:
                masked_message += " by this artist"
        else:
            title_mentions = entities.mentions(message, "title")
            artist_mentions = entities.mentions(message, "artist")
            album_mentions = entities.mentions(message, "album")
            # A short title or album called "Park" must not override the
            # complete artist mention "Linkin Park" in the same words.
            title_mentions = [
                mention for mention in title_mentions
                if not any(artist.start <= mention.start and mention.end <= artist.end
                           for artist in artist_mentions)
            ]
            album_mentions = [
                mention for mention in album_mentions
                if not any(artist.start <= mention.start and mention.end <= artist.end
                           for artist in artist_mentions)
            ]
            masked_message = entities.mask(message, title_mentions, artist_mentions, album_mentions)
        router = self._r5_router or get_r5_router()
        prediction = router.predict(masked_message)
        if not prediction.accepted:
            return None

        # Spend fuzzy matching only on messages already classified as an R5
        # lookup. Ordinary chat and playlist actions use the fast exact index.
        if (
            prediction.intent == "artist_tracks"
            and not artist_mentions
            and not state.last_question_artist
        ):
            artist_mentions = entities.mentions(message, "artist", fuzzy=True)
        elif prediction.intent == "album_tracks" and not album_mentions:
            album_mentions = entities.mentions(message, "album", fuzzy=True)
        elif performer_target is None and prediction.intent in {"artist", "album", "release", "tags"}:
            if not title_mentions and not (artist_mentions and prediction.intent == "tags"):
                title_mentions = entities.mentions(message, "title", fuzzy=True)
            # Without an explicit artist slot, fuzzy matching the whole
            # question invents constraints (e.g. "sang" -> artist "Sango").
        if performer_target is None and (title_mentions or artist_mentions or album_mentions):
            masked_message = entities.mask(message, title_mentions, artist_mentions, album_mentions)
            prediction = router.predict(masked_message)
            if not prediction.accepted:
                return None

        if artist_mentions and re.search(
            r"\b(?:what|which|show|list)\b.*\b(?:songs|tracks)\b.*\b(?:does|by)\b",
            message, re.I,
        ) and not re.search(r"\b(album|record|tracklist)\b", message, re.I):
            prediction = IntentPrediction("artist_tracks", 1.0, 1.0, True)

        if performer_target is not None and prediction.intent == "artist":
            state.pending_disambiguation_ids.clear()
            state.pending_disambiguation_action = None
            if explicit_artist and not artist_mentions:
                return f"I couldn't find the artist ‘{explicit_artist}’ in the catalog. Please check the name."
            if explicit_album and not album_constraint_mentions:
                return f"I couldn't find the album ‘{explicit_album}’ in the catalog. Please check the name."
            if not title_mentions and album_mentions:
                album_ids = set().union(*(m.ids for m in album_mentions))
                if artist_mentions:
                    album_ids &= set().union(*(m.ids for m in artist_mentions))
                rows = [self.service.get_track(track_id) for track_id in sorted(album_ids)]
                rows = [row for row in rows if row]
                if not rows:
                    return f"I couldn't find catalog tracks from ‘{performer_target}’ by ‘{explicit_artist}’."
                matched_names = {m.name for m in album_mentions}
                albums = sorted({name for row in rows for name in self.service.display_value(row.get("album_name")).split(", ") if normalize(name) in matched_names})
                if len(albums) != 1:
                    return "Which album do you mean? Catalog matches: " + "; ".join(albums) + "."
                performers = sorted({name for row in rows for name in self.service.display_value(row.get("artist_name")).split(", ") if name})
                state.pending_disambiguation_ids.clear()
                state.pending_disambiguation_action = None
                state.last_question_track_id = None
                state.last_question_artist = None
                return f"‘{albums[0]}’ is an album. Performers credited on its catalog tracks include {', '.join(performers)}."
            if not title_mentions:
                return f"I couldn't find a track or album matching ‘{performer_target}’ in the catalog."

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
                exact_names = [name for name in album_names if name in {m.name for m in album_mentions}]
                if len(exact_names) == 1:
                    album_name = album_names[exact_names[0]]
                elif len(album_names) == 1:
                    album_name = next(iter(album_names.values()))
                else:
                    return "Which album do you mean? Please name the album or a track on it."
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

        if action == "tags" and title is None and artist is not None:
            tag_counts = Counter(
                str(tag).casefold()
                for track_id in artist.ids
                if (track := self.service.get_track(track_id)) is not None
                for tag in (track.get("tag_list") or [])
                if str(tag).strip()
            )
            if tag_counts:
                common = ", ".join(tag for tag, _ in tag_counts.most_common(6))
                state.last_question_artist = artist.name
                return f"Catalog tracks by {artist.name} are often tagged {common}."
            return f"The catalog lists no style tags for tracks by {artist.name}."

        # A short R5 follow-up with a clear information intent uses the last
        # discussed track even when the user omits a title ("what year?").
        if title is None:
            has_unmatched_term = any(
                token not in _R5_QUESTION_TERMS
                for token in re.findall(r"\w+", message.casefold())
            )
            track_id = state.last_question_track_id
            if (
                track_id
                and not has_unmatched_term
                and not artist
                and not album_mentions
                and self.service.get_track(track_id)
            ):
                state.pending_disambiguation_ids = []
                state.pending_disambiguation_action = None
                state.last_question_artist = self.service.display_value(
                    self.service.get_track(track_id).get("artist_name")
                )
                return self._answer_track_action(track_id, action)

        if title is None:
            return "Which song or track should I look up?"
        title_ids = set().union(*(m.ids for m in title_mentions)) if performer_target else title.ids
        candidates = [self.service.get_track(track_id) for track_id in sorted(title_ids)]
        candidates = [track for track in candidates if track]
        if artist:
            artist_ids = set().union(*(m.ids for m in artist_mentions))
            candidates = [track for track in candidates if track["track_id"] in artist_ids]
        if explicit_album:
            album_ids = set().union(*(m.ids for m in album_constraint_mentions))
            candidates = [track for track in candidates if track["track_id"] in album_ids]
        elif not artist and state.last_question_track_id in title.ids:
            # A repeated title inherits its artist only when the immediately
            # preceding R5 lookup identified that exact same catalog title.
            previous = self.service.get_track(state.last_question_track_id)
            if previous and normalize(self.service.display_value(previous.get("track_name"))) == normalize(title.name):
                candidates = [previous]
        if not candidates:
            if explicit_album:
                return f"I couldn't find a catalog track matching ‘{performer_target}’ on the album ‘{explicit_album}’."
            return f"I couldn't find a catalog track matching {title.name}."
        # Popularity orders ambiguous catalog versions; the ID only breaks ties.
        def candidate_rank(track):
            try:
                popularity = float(track.get("popularity") or 0)
            except (TypeError, ValueError):
                popularity = 0
            return (-popularity, track["track_id"])
        candidates.sort(key=candidate_rank)
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
            re.search(r"\b(recommend(?:ation)?s?|suggest(?:ion)?s?)\b", lowered)
            and re.search(r"\b(song|songs|track|tracks|music|something|playlist|few more|more like|like these)\b", lowered)
        ) or bool(
            re.search(r"\bwhat else\b.*\b(sound good|go(?:es)? well|fit)\b", lowered)
            and re.search(r"\b(song|songs|track|tracks|playlist|have)\b", lowered)
        )

    @staticmethod
    def _is_playlist_creation(message: str) -> bool:
        lowered = message.casefold()
        return bool(
            (re.search(r"\b(create|make|build|generate|put together)\b", lowered)
             and re.search(r"\b(playlist|songs|tracks|music)\b", lowered))
            or re.search(r"\bput together\b.*\b(?:some|a|an)\b", lowered)
            or re.search(r"\b(?:give me|i want|i need)\b.*\bplaylist\b", lowered)
        )

    def _recommend(self, message: str, state: ConversationState) -> tuple[str, bool]:
        if not state.playlist_ids:
            return "Add at least one song to your playlist first, and I can suggest related tracks.", False
        requested_genres, requested_moods = extract_preferences(message)
        retrieval_request = "Recommend songs based on my playlist"
        if requested_genres or requested_moods:
            details = []
            if requested_genres:
                details.append("genres: " + ", ".join(sorted(requested_genres)))
            if requested_moods:
                details.append("moods: " + ", ".join(sorted(requested_moods)))
            retrieval_request += "; preferences: " + "; ".join(details)
        recommendations = self.service.recommend_from_playlist(
            state, limit=5, request=retrieval_request
        )
        if not recommendations:
            return "I couldn't find enough related tracks in the catalog for your playlist.", False
        state.pending_recommendation_ids = [track["track_id"] for track in recommendations]
        state.pending_disambiguation_ids.clear()
        state.pending_disambiguation_action = None
        rendered = []
        seed_genres: set[str] = set()
        seed_moods: set[str] = set()
        seed_styles: set[str] = set()
        seed_artists = {
            artist.casefold()
            for track_id in state.playlist_ids
            if (seed := self.service.get_track(track_id)) is not None
            for artist in self.service.display_value(seed.get("artist_name")).split(", ")
            if artist
        }
        for track_id in state.playlist_ids:
            seed = self.service.get_track(track_id)
            if seed:
                genres, moods = classify_tags(
                    self.service.display_value(seed.get("tag_list")).split(", ")
                )
                seed_genres.update(genres)
                seed_moods.update(moods)
                seed_styles.update(style for style, weight in track_style(seed).items() if weight > 1)
        for index, track in enumerate(recommendations, start=1):
            genres, moods = classify_tags(
                self.service.display_value(track.get("tag_list")).split(", ")
            )
            common_genres = sorted(seed_genres & genres)
            common_moods = sorted(seed_moods & moods)
            common_styles = sorted(seed_styles & set(track_style(track)))
            candidate_artists = {
                artist.casefold()
                for artist in self.service.display_value(track.get("artist_name")).split(", ")
                if artist
            }
            if seed_artists & candidate_artists:
                reason = "; by an artist already in your playlist"
                if common_styles:
                    reason += f"; shares styles: {', '.join(common_styles[:3])}"
            elif common_styles:
                reason = f"; shares styles: {', '.join(common_styles[:3])}"
            elif common_genres:
                reason = f"; shares genres: {', '.join(common_genres[:3])}"
            elif common_moods:
                reason = f"; shares moods: {', '.join(common_moods[:3])}"
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
        query = self._playlist_description(message)
        if not query:
            return "What mood, genre, or activity should I use for the playlist?", False

        # Length cues determine how many tracks to select below; they should
        # not become retrieval terms (for example, "short commute" otherwise
        # favors tracks whose titles contain "short"). Keep the theme/activity
        # words and the original theme for the user-facing confirmation.
        retrieval_query = re.sub(r"\b(short|quick)\b|\blong\s+trip\b", " ", query, flags=re.I)
        # Activity words are useful for interpreting the request, but they
        # often collide with catalog titles (for example, Gym Class Heroes).
        # Replace them with the listening qualities the user likely wants.
        if re.search(r"\b(gym|workout|exercise)\b", retrieval_query, re.I):
            retrieval_query = re.sub(r"\b(gym|workout|exercise)\b", " ", retrieval_query, flags=re.I)
            retrieval_query += " energetic upbeat exercise music"
        if re.search(r"\bcommute\b", retrieval_query, re.I):
            retrieval_query = re.sub(r"\bcommute\b", " ", retrieval_query, flags=re.I)
            retrieval_query += " easy listening relaxed music"
        if re.search(r"\bsad\b", retrieval_query, re.I) and re.search(r"\blove\b", retrieval_query, re.I):
            retrieval_query = re.sub(r"\blove\b", "romantic love songs", retrieval_query, flags=re.I)
        retrieval_query = " ".join(retrieval_query.split()).strip(" .,!?:;-") or "music"
        candidates = self.service.recommend_for_playlist_description(
            retrieval_query, limit=20, state=state, request=retrieval_query,
        )
        unique_candidates = list(
            {track["track_id"]: track for track in candidates}.values()
        )
        lowered_query = query.casefold()
        if re.search(r"\b(party|road trip|long trip)\b", lowered_query):
            target_minutes = 75
        elif re.search(r"\b(short|quick|commute)\b", lowered_query):
            target_minutes = 20 if re.search(r"\b(short|quick)\b", lowered_query) else 30
        else:
            # A 45-minute listening session is a sensible default for mood,
            # study, dinner, and workout descriptions.
            target_minutes = 45

        unique_tracks = []
        total_duration_ms = 0
        short_request = bool(re.search(r"\b(short|quick)\b", lowered_query))
        minimum_tracks = min(4 if short_request else 8, len(unique_candidates))
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
        state.created_playlist_name = query[:100]
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

    @staticmethod
    def _playlist_description(message: str) -> str | None:
        """Extract theme words from a conversational playlist request."""
        text = message.strip().strip(" .!?,-")
        if re.search(r"\bgoing for a run\b", text, re.I):
            return "running workout"
        text = re.sub(r"[.!?]\s*(?:could|can|would) you (?:make|create|build|put together)\b.*$", "", text, flags=re.I)
        text = re.sub(
            r"^(?:(?:please|could you|can you|would you|will you)\s+)+", "", text, flags=re.I
        )
        text = re.sub(r"^(?:i(?:'d| would) like|i want)\s+", "", text, flags=re.I)
        text = re.sub(r"^(?:some|a|an)\s+", "", text, flags=re.I)
        text = re.sub(r"^(?:put together|create|make|build|generate)\s+", "", text, flags=re.I)
        text = re.sub(r"^(?:me\s+)?(?:a|an|the|some)\s+", "", text, flags=re.I)
        text = re.sub(r"^(?:me\s+)?(?:a|an|the|some)\s+", "", text, flags=re.I)
        text = re.sub(r"^(?:me\s+)", "", text, flags=re.I)

        # Retain meaningful modifiers while dropping the command scaffolding.
        text = re.sub(r"\bplaylist\b", " ", text, flags=re.I)
        text = re.sub(r"\b(?:please|thanks|thank you)\b", " ", text, flags=re.I)
        text = re.sub(r"\b(?:for|with|of)\b", " ", text, flags=re.I)
        text = re.sub(r"\b(?:my|the|a|an|song|music)\b", " ", text, flags=re.I)
        # Keep "love songs" as a useful romance cue, but drop generic trailing
        # "songs" so it does not become part of the theme/query.
        text = re.sub(r"(?<!love )\bsongs\b", " ", text, flags=re.I)
        text = re.sub(r"\b(?:something|that is|that's)\b", " ", text, flags=re.I)
        text = re.sub(r"\b(?:me|myself)\b", " ", text, flags=re.I)
        text = re.sub(r"[,;:]", " ", text)
        text = " ".join(text.split()).strip(" .,!?:;-")
        # Time-of-day wording is usually conversational filler here and causes
        # title matches (e.g. tracks named "Tonight") to dominate retrieval.
        text = re.sub(r"\b(?:tonight|today|this evening)\b", " ", text, flags=re.I)
        text = " ".join(text.split()).strip(" .,!?:;-")
        if not text or not re.search(
            r"\b(?:mood|genre|style|activity|energy|workout|gym|commute|party|"
            r"study|studying|love|sad|happy|upbeat|calm|focus|relax|relaxing|rock|pop|jazz|metal|"
            r"short|quick|road trip|dinner)\b",
            text,
            flags=re.I,
        ):
            return None
        return text
