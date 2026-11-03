"""Deterministic catalog resolution for constraints shared by R1 and R2."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
import re
import unicodedata

from .relevance_feedback import artist_keys, artist_names, name_key, values
from .music_features import extract_preferences


class ConstraintError(ValueError):
    """An explicit exclusion requires clarification; do not apply part of it."""


def _key(text):
    plain = unicodedata.normalize("NFKD", str(text))
    return name_key("".join(c for c in plain if not unicodedata.combining(c)))


def _unquoted(text):
    """Mask quoted names while retaining character offsets and apostrophes."""
    result, closing = list(text), None
    pairs = {'"': '"', "“": "”", "‘": "’", "'": "'"}
    for i, char in enumerate(text):
        if closing:
            result[i] = "_"
            if char == closing:
                closing = None
        elif char in pairs and (i == 0 or not text[i - 1].isalnum()):
            closing = pairs[char]
            result[i] = "_"
    return "".join(result)


def split_request_exclusions(text):
    """Separate explicit negative clauses; keep quoted/positive musical names."""
    masked = _unquoted(text)
    marker = r"\b(?:exclude|excluding|without|avoid|except|not\s+by|no|not)\s+"
    for match in re.finditer(marker, masked, re.I):
        before = text[:match.start()].rstrip()
        # 'Recommend No Doubt' and 'similar to Without Me' are positive names.
        if re.search(r"\b(?:recommend|suggest)(?:\s+me)?$|\b(?:like|similar to|by|called|titled|artist)$", before, re.I):
            continue
        if not before:
            continue
        positive = re.sub(r"(?:\s+but|\s+and)\s*$", "", before, flags=re.I).rstrip(" ,;")
        return positive, "exclude " + text[match.end():].strip().rstrip(".!?")
    return text, None


def explicit_request_preferences(text, named_artists=()):
    """Infer musical terms without treating a referenced artist as a genre."""
    positive, _ = split_request_exclusions(text)
    positive = re.split(r"\b(?:similar to|more like|like|by)\b", positive, maxsplit=1, flags=re.I)[0]
    for artist in sorted(named_artists, key=len, reverse=True):
        if len(artist.split()) > 1:
            positive = re.sub(r"(?<!\w)" + re.escape(artist) + r"(?!\w)", " ", positive, flags=re.I)
    return extract_preferences(positive)


@dataclass
class ResolvedExclusions:
    track_ids: set[str] = field(default_factory=set)
    artists: set[str] = field(default_factory=set)

    def joined(self, other):
        return ResolvedExclusions(self.track_ids | other.track_ids, self.artists | other.artists)


def resolve_exclusions(text, catalog, displayed_ids=(), select_positions=None, focus_id=None, *, kind=None,
                       allow_missing=False):
    """Resolve a scalar or list atomically, preserving literal catalog names.

    Explicit artist/track kinds come from the structured R1 interface. A bare
    ambiguous title asks for an artist. Hard track bans include matching remasters.
    Only candidate-only legacy providers may allow missing references: an absent
    target cannot be returned from that snapshot. Ambiguities still fail closed.
    """
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ConstraintError("Please give a nonempty exclusion of at most 2000 characters.")
    titles, artists = defaultdict(set), defaultdict(set)
    version = r"\s*(?:-\s*|\()\s*(?:\d{4}\s+)?remaster(?:ed)?(?:\s+\d{4})?\)?\s*$"
    for track_id, row in catalog.items():
        for title in values(row.get("track_name")):
            titles[_key(title)].add(track_id)
            titles[_key(re.sub(version, "", title, flags=re.I))].add(track_id)
        for artist in artist_names(row):
            artists[_key(artist)].add(name_key(artist))

    def songs(ids, reference, artist=None):
        ids = set(ids)
        if artist is not None:
            ids = {t for t in ids if artist in {_key(a) for a in artist_names(catalog[t])}}
        if not ids:
            return None
        identities = {tuple(sorted(_key(a) for a in values(catalog[t].get("artist_name")))) for t in ids}
        if artist is None and len(identities) > 1:
            raise ConstraintError(f'Which artist\'s "{reference}" should I exclude? Please give the title and artist.')
        return ResolvedExclusions(track_ids=ids)

    def one(reference):
        reference = reference.strip(' \"“”‘’\'').rstrip(".!?")
        local_kind = kind
        if select_positions is not None:
            positions = select_positions(reference, len(displayed_ids))
            if positions is not None:
                if not positions or not displayed_ids:
                    raise ConstraintError("Please choose an available recommendation number.")
                return ResolvedExclusions(track_ids={displayed_ids[i] for i in positions})
        typed = re.match(r"^(?:the\s+)?(artist|song|track)\s+(.+)$", reference, re.I)
        if typed and _key(reference) not in artists and _key(reference) not in titles:
            local_kind = "artist" if typed.group(1).lower() == "artist" else "track"
            reference = typed.group(2)
        key = _key(reference)
        if key in {"this song", "that song", "this track", "that track", "this artist", "that artist", "it"}:
            t = displayed_ids[0] if len(displayed_ids) == 1 else focus_id if focus_id in displayed_ids else None
            if t is None or t not in catalog:
                raise ConstraintError("Which recommendation do you mean? Please give its number, title, or artist.")
            if "artist" in key:
                names = artist_names(catalog[t])
                if len(names) != 1:
                    raise ConstraintError("That track credits several artists. Please name the artist to exclude.")
                return ResolvedExclusions(artists={name_key(names[0])})
            return ResolvedExclusions(track_ids={t})
        artist_hit, title_hit = artists.get(key), titles.get(key)
        if artist_hit and local_kind != "track":
            # Exact artist names have priority for a bare exclusion. A caller
            # can explicitly say 'track X' or 'X by Y' to select a song instead.
            return ResolvedExclusions(artists=set(artist_hit))
        if title_hit and local_kind != "artist":
            return songs(title_hit, reference)
        if local_kind != "artist":
            for match in reversed(list(re.finditer(r"\s+by\s+", reference, re.I))):
                found = songs(titles.get(_key(reference[:match.start()]), ()), reference,
                              _key(reference[match.end():]))
                if found:
                    return found
        return None

    @lru_cache(maxsize=512)
    def resolve(reference):
        reference = reference.strip(" ,;")
        found = one(reference)
        if found:
            return found
        separators = list(re.finditer(
            r"\s+(?:and|or)(?:\s+(?:exclude|excluding|without|avoid|except|not by|no|not))?\s+"
            r"|\s*[,;]\s*(?:(?:exclude|excluding|without|avoid|except|not by|no|not)\s+)?"
            r"|\s+(?:exclude|excluding|without|avoid|except|not by|no|not)\s+", _unquoted(reference), re.I))
        if len(separators) >= 20:
            raise ConstraintError("Please give at most 20 exclusions at a time.")
        error = None
        for match in reversed(separators):
            try:
                return resolve(reference[:match.start()]).joined(resolve(reference[match.end():]))
            except ConstraintError as exc:
                error = exc
        if error:
            raise error
        if allow_missing:
            return ResolvedExclusions()
        raise ConstraintError(f'I could not identify "{reference}". Please give an exact artist or title and artist.')

    return resolve(text)


def resolve_preference_exclusions(preferences, catalog, request="", playlist_ids=(), *, allow_missing=False):
    """Resolve structured R1 exclusions using the same identities as chat.

    A deterministically understood negative clause is authoritative. In
    particular, an LLM dropping or changing the artist in "Numb by Linkin Park"
    must not exclude a different artist's recording. Without such a clause,
    validated LLM references are resolved against metadata, with ambiguity checks.
    """
    _, negative = split_request_exclusions(request)
    if negative and not re.match(r"(?:the\s+)?style\s+of\b", negative[len("exclude "):], re.I):
        resolved = resolve_exclusions(negative[len("exclude "):], catalog, allow_missing=allow_missing)
    else:
        resolved = ResolvedExclusions()
        for kind, field in (("artist", "excluded_artists"), ("track", "excluded_tracks")):
            for reference in preferences.get(field, []):
                resolved = resolved.joined(resolve_exclusions(reference, catalog, kind=kind,
                                                               allow_missing=allow_missing))
    if preferences.get("avoid_playlist_artists"):
        resolved.artists.update(artist for tid in playlist_ids if tid in catalog
                                for artist in artist_keys(catalog[tid]))
    return resolved
