"""Conservative conversational feedback resolution; no implicit dislike on R7 rejection."""
from __future__ import annotations

from dataclasses import dataclass, field
import re

from .relevance_feedback import artist_keys, artist_names as credit_names, name_key, values
from .request_constraints import ConstraintError, resolve_exclusions, split_request_exclusions


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
    recommendation_query: str | None = None
    error: str | None = None


def interpret_feedback(text, displayed_ids, catalog, select_positions, focus_id=None):
    """Resolve the entire utterance before applying any change (atomic feedback).

    Bare 'this' is resolved only for a single displayed track or an explicitly
    discussed track in that displayed set. Otherwise ask rather than guess.
    """
    polite = r"(?:(?:please|could you|can you|would you)\s+)*"
    text = re.sub(r"^" + polite, "", text.strip(), flags=re.I).rstrip(".!?")
    if re.fullmatch(r"(?:please\s+)?(?:reset|forget|clear)\s+(?:my\s+)?(?:feedback|preferences)", text, re.I):
        return FeedbackUpdate(reset=True)
    if re.fullmatch(r"(?:please\s+)?(?:show|recommend|suggest|give)(?:\s+me)?\s+something\s+(?:different|else)", text, re.I):
        return FeedbackUpdate(recommend=True, different=True)

    more_pattern = r"(?:please\s+)?(?:recommend|suggest)(?:\s+me)?\s+more(?:\s+(?:songs|tracks|music))?\s+like\s+(.+)"
    vote_pattern = r"(?:i\s+)?(?:really\s+)?(don['’]t like|do not like|dislike|hate|didn['’]t like|did not like|like|liked|love|loved|enjoy|enjoyed)\s+(.+)"
    clauses = re.split(
        r"\s*(?:,\s*)?(?:but|and)\s+(?=" + polite
        + r"(?:i\s+)?(?:really\s+)?(?:don['’]t|do not|didn['’]t|did not|dislike|like|liked|love|loved|hate|enjoy|enjoyed|exclude|avoid|recommend|suggest|show)\b)",
        text, flags=re.I,
    )
    whole = re.fullmatch(vote_pattern, text, re.I) or re.fullmatch(more_pattern, text, re.I)
    if whole and len(clauses) > 1:
        target = whole.groups()[-1]
        keys = {name_key(target), name_key(re.sub(r"^(?:the\s+)?(?:song|track|artist)\s+", "", target, flags=re.I))}
        literal = any(
            keys & {name_key(v) for v in values(row.get("track_name")) + values(row.get("artist_name"))}
            or any(name_key(f"{title} by {artist}") in keys for title in values(row.get("track_name"))
                   for artist in values(row.get("artist_name")))
            for row in catalog.values())
        if literal:
            clauses = [text]  # For example, a literal song titled 'Love and Hate'.
    update = FeedbackUpdate()
    recognized = False
    for clause in clauses:
        clause = re.sub(r"^" + polite, "", clause.strip(" ,;"), flags=re.I)
        more = re.fullmatch(more_pattern, clause, re.I)
        if more and re.fullmatch(r"(?:my|the|this)\s+playlist", more.group(1), re.I):
            return None  # A playlist request does not implicitly rate its songs.
        vote = re.fullmatch(vote_pattern, clause, re.I)
        style = re.fullmatch(r"(?:avoid|exclude|excluding|without)\s+(?:the\s+)?style\s+of\s+(.+)", clause, re.I)
        exclusion = re.fullmatch(r"(?:please\s+)?(?:exclude|avoid|excluding|without)\s+(.+)", clause, re.I)
        if more:
            target, positive, hard = more.group(1), True, False
            update.recommend = True
        elif vote:
            target = vote.group(2)
            positive = vote.group(1).casefold() in {"like", "liked", "love", "loved", "enjoy", "enjoyed"}
            hard = False
        elif style:
            target, positive, hard = style.group(1), False, False
        elif exclusion:
            target, positive, hard = exclusion.group(1), False, True
        else:
            # A fresh request may accompany feedback in either order. Keep it
            # separate from the reference being rated, and validate every vote
            # before applying any of the utterance.
            request = re.fullmatch(r"(?:recommend|suggest)\s+.+", clause, re.I)
            if request and len(clauses) > 1:
                if update.recommendation_query is not None:
                    update.error = "Please give one recommendation request at a time. I haven't changed your preferences."
                    return update
                update.recommendation_query = clause
                update.recommend = True
                continue
            if recognized:
                update.error = "Please give feedback by song number, title, or artist. I haven't changed your preferences."
                return update
            return None
        recognized = True
        target = re.sub(r"\s+please$", "", target, flags=re.I).strip(" \"“”'")
        target = re.sub(r"\b(tracks?|songs?)\s*(\d+)", r"\1 \2", target, flags=re.I)
        if hard:
            try:
                resolved = resolve_exclusions(target, catalog, displayed_ids, select_positions, focus_id)
            except ConstraintError as exc:
                update.error = str(exc)
                return update
            update.excluded_ids.extend(sorted(resolved.track_ids))
            update.excluded_artists.extend(sorted(resolved.artists))
            continue
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
                    artist_names.extend(credit_names(catalog[track_id]))
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
                all_artists = {name_key(a): a for row in catalog.values() for a in credit_names(row)}
                if name_key(target) in all_artists:
                    lookup = name_key(target)
                title_text = re.sub(r"^(?:the\s+)?(?:song|track)\s+", "", target, flags=re.I)
                literal_title = any(name_key(title_text) in {name_key(t) for t in values(row.get("track_name"))}
                                    for row in catalog.values())
                parts = [title_text] if literal_title else re.split(r"\s+by\s+", title_text, maxsplit=1, flags=re.I)
                title, named_artist = parts[0], parts[1] if len(parts) > 1 else ""
                candidates = [track_id for track_id, row in catalog.items()
                              if name_key(title) in {name_key(t) for t in values(row.get("track_name"))}
                              and (not named_artist or name_key(named_artist) in {name_key(a) for a in values(row.get("artist_name"))})]
                if by_artist:
                    ids = [track_id for track_id in displayed_ids
                           if lookup in artist_keys(catalog[track_id])]
                    if len(ids) != 1:
                        update.error = "Which song by that artist do you mean? Please give its recommendation number."
                        return update
                elif lookup in all_artists and (is_artist or not candidates):
                    artist_names = [all_artists[lookup]]
                    ids = [track_id for track_id in displayed_ids
                           if lookup in artist_keys(catalog[track_id])]

                else:
                    in_display = [
                        track_id for track_id in candidates
                        if track_id in displayed_ids
                    ]
                    ids = in_display or candidates

                    if len(ids) != 1:
                        # Prefer a single previously displayed recording when
                        # the user omits a common remaster suffix.
                        version_suffix = (
                            r"\s*(?:-\s*|\()\s*"
                            r"(?:\d{4}\s+)?remaster(?:ed)?"
                            r"(?:\s+\d{4})?\)?\s*$"
                        )

                        displayed_matches = [
                            track_id
                            for track_id in displayed_ids
                            if track_id in catalog
                            and any(
                                name_key(re.sub(version_suffix, "", track_title, flags=re.I))
                                == name_key(title)
                                for track_title in values(
                                    catalog[track_id].get("track_name")
                                )
                            )
                            and (
                                not named_artist
                                or name_key(named_artist) in {
                                    name_key(a)
                                    for a in values(
                                        catalog[track_id].get("artist_name")
                                    )
                                }
                            )
                        ]

                        if len(displayed_matches) == 1:
                            ids = displayed_matches
                        else:
                            # No unique displayed match: search the full catalog.
                            catalog_matches = [
                                track_id
                                for track_id, row in catalog.items()
                                if any(
                                    name_key(
                                        re.sub(version_suffix, "", track_title, flags=re.I)
                                    ) == name_key(title)
                                    for track_title in values(row.get("track_name"))
                                )
                                and (
                                    not named_artist
                                    or name_key(named_artist) in {
                                        name_key(a)
                                        for a in values(row.get("artist_name"))
                                    }
                                )
                            ]

                            if len(catalog_matches) == 1:
                                ids = catalog_matches
                            elif len(catalog_matches) > 1:
                                update.error = (
                                    "I found multiple versions of that song. "
                                    "Please give the complete track title "
                                    "or select a recommendation number."
                                )
                                return update
                            else:
                                update.error = (
                                    "I couldn't identify one catalog track or artist. "
                                    "Please use a recommendation number or include its artist."
                                )
                                return update


        if any(track_id not in catalog for track_id in ids):
            update.error = "That track is no longer in the catalog."
            return update
        if artist_names:
            if style:
                update.error = "Please name a track to use as the negative style reference. I haven't changed your preferences."
                return update
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
    if recognized and update.recommendation_query:
        positive, exclusion = split_exclusion_request(update.recommendation_query)
        if exclusion:
            try:
                resolved = resolve_exclusions(exclusion[len("exclude "):], catalog,
                                              displayed_ids, select_positions, focus_id)
            except ConstraintError as exc:
                update.error = str(exc)
                return update
            update.excluded_ids.extend(sorted(resolved.track_ids))
            update.excluded_artists.extend(sorted(resolved.artists))
            update.recommendation_query = positive
    return update if recognized else None


def split_exclusion_request(text):
    """Shared R1/R2 constraint syntax; negatives never become positive query terms."""
    return split_request_exclusions(text)
