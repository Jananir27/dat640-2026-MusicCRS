"""Conservative conversational feedback resolution; no implicit dislike on R7 rejection."""
from __future__ import annotations

from dataclasses import dataclass, field
import re

from .relevance_feedback import name_key, values


@dataclass
class FeedbackUpdate:
    likes: list[str] = field(default_factory=list)
    dislikes: list[str] = field(default_factory=list)
    liked_artists: list[str] = field(default_factory=list)
    disliked_artists: list[str] = field(default_factory=list)
    excluded_ids: list[str] = field(default_factory=list)
    excluded_artists: list[str] = field(default_factory=list)
    recommend: bool = False
    different: bool = False
    reset: bool = False
    error: str | None = None


def interpret_feedback(text, displayed_ids, catalog, select_positions, focus_id=None):
    """Resolve the entire utterance before applying any change (atomic feedback).

    Bare 'this' is resolved only for a single displayed track or an explicitly
    discussed track in that displayed set. Otherwise ask rather than guess.
    """
    text = text.strip().rstrip(".!?")
    if re.fullmatch(r"(?:please\s+)?(?:reset|forget|clear)\s+(?:my\s+)?(?:feedback|preferences)", text, re.I):
        return FeedbackUpdate(reset=True)
    if re.fullmatch(r"(?:please\s+)?(?:show|recommend|suggest|give)(?:\s+me)?\s+something\s+(?:different|else)", text, re.I):
        return FeedbackUpdate(recommend=True, different=True)

    clauses = re.split(
        r"\s*(?:,\s*)?(?:but|and)\s+(?=(?:i\s+)?(?:don['’]t|do not|dislike|like|love|hate|enjoy|exclude|avoid|recommend|show)\b)",
        text, flags=re.I,
    )
    update = FeedbackUpdate()
    recognized = False
    for clause in clauses:
        clause = clause.strip(" ,;")
        more = re.fullmatch(r"(?:please\s+)?(?:recommend|suggest)(?:\s+me)?\s+more(?:\s+(?:songs|tracks|music))?\s+like\s+(.+)", clause, re.I)
        vote = re.fullmatch(
            r"(?:i\s+)?(?:really\s+)?(don['’]t like|do not like|dislike|hate|didn['’]t like|like|love|enjoy)\s+(.+)",
            clause, re.I,
        )
        exclusion = re.fullmatch(r"(?:please\s+)?(?:exclude|avoid|excluding|without)\s+(.+)", clause, re.I)
        if more:
            target, positive, hard = more.group(1), True, False
            update.recommend = True
        elif vote:
            target = vote.group(2)
            positive = vote.group(1).casefold() in {"like", "love", "enjoy"}
            hard = False
        elif exclusion:
            target, positive, hard = exclusion.group(1), False, True
        else:
            if recognized:
                update.error = "Please give feedback by song number, title, or artist. I haven't changed your preferences."
                return update
            return None
        recognized = True
        target = re.sub(r"\s+please$", "", target, flags=re.I).strip(" \"“”'")
        target = re.sub(r"\b(tracks?|songs?)\s*(\d+)", r"\1 \2", target, flags=re.I)
        is_artist = bool(re.match(r"(?:(?:this|that|the)\s+)?artist\b", target, re.I)
                         or re.fullmatch(r"(?:this|that)\s+artist", target, re.I))
        reference = re.sub(r"^(?:the\s+)?artist\s+", "", target, flags=re.I)
        implicit = re.fullmatch(r"(?:this|that|it|this song|that song|this track|that track|this artist|that artist)", target, re.I)
        ids = []
        artist_names = []
        if implicit:
            if len(displayed_ids) == 1:
                ids = list(displayed_ids)
            elif focus_id and (focus_id in displayed_ids or not displayed_ids):
                ids = [focus_id]
            else:
                update.error = "Which recommendation do you mean? Please give its number, title, or artist."
                return update
            if is_artist:
                for track_id in ids:
                    artist_names.extend(values(catalog[track_id].get("artist_name")))
                if len(set(artist_names)) != 1:
                    update.error = "That track credits several artists. Please name the artist you mean."
                    return update
        else:
            positions = select_positions(target, len(displayed_ids))
            if positions is not None:
                if not displayed_ids:
                    update.error = "Please request recommendations first, or give the track's title and artist."
                    return update
                if not positions:
                    update.error = f"Please choose recommendation numbers from 1 to {len(displayed_ids)}. I haven't changed your preferences."
                    return update
                ids = [displayed_ids[position] for position in positions]
            else:
                by_artist = re.fullmatch(r"(?:the\s+)?(?:one|song|track)\s+by\s+(.+)", target, re.I)
                lookup = name_key(by_artist.group(1) if by_artist else reference)
                all_artists = {name_key(a): a for row in catalog.values() for a in values(row.get("artist_name"))}
                if name_key(target) in all_artists:
                    lookup = name_key(target)
                title_text = re.sub(r"^(?:the\s+)?(?:song|track)\s+", "", target, flags=re.I)
                parts = re.split(r"\s+by\s+", title_text, maxsplit=1, flags=re.I)
                title, named_artist = parts[0], parts[1] if len(parts) > 1 else ""
                candidates = [track_id for track_id, row in catalog.items()
                              if name_key(title) in {name_key(t) for t in values(row.get("track_name"))}
                              and (not named_artist or name_key(named_artist) in {name_key(a) for a in values(row.get("artist_name"))})]
                if by_artist:
                    ids = [track_id for track_id in displayed_ids
                           if lookup in {name_key(a) for a in values(catalog[track_id].get("artist_name"))}]
                    if len(ids) != 1:
                        update.error = "Which song by that artist do you mean? Please give its recommendation number."
                        return update
                elif lookup in all_artists and (is_artist or not candidates):
                    artist_names = [all_artists[lookup]]
                    ids = [track_id for track_id in displayed_ids
                           if lookup in {name_key(a) for a in values(catalog[track_id].get("artist_name"))}]
                else:
                    in_display = [track_id for track_id in candidates if track_id in displayed_ids]
                    ids = in_display or candidates
                    if len(ids) != 1:
                        update.error = "I couldn't identify one catalog track or artist. Please use a recommendation number or include its artist."
                        return update
        if any(track_id not in catalog for track_id in ids):
            update.error = "That track is no longer in the catalog."
            return update
        if artist_names:
            destination = update.excluded_artists if hard else (update.liked_artists if positive else update.disliked_artists)
            if not hard:
                opposite = update.disliked_artists if positive else update.liked_artists
                opposite[:] = [a for a in opposite if name_key(a) not in {name_key(n) for n in artist_names}]
            destination.extend(artist_names)
        if not artist_names or not hard:
            destination = update.excluded_ids if hard else (update.likes if positive else update.dislikes)
            if not hard:
                opposite = update.dislikes if positive else update.likes
                opposite[:] = [track_id for track_id in opposite if track_id not in ids]
            destination.extend(ids)
    return update if recognized else None


def split_exclusion_request(text):
    """Small bridge until R1 supplies structured constraints; never put negatives into the query."""
    match = re.search(r"\s+(?:but\s+)?(?:exclude|excluding|without|avoid)\s+(.+?)[.!?]*$", text, re.I)
    if not match:
        return text, None
    return text[:match.start()].strip(), "exclude " + match.group(1)
