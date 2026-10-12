"""Small, deterministic genre and mood features derived from catalog tags.

The source ``tag_list`` mixes genres and moods with decades, ratings, and
personal labels. Keep the source data intact and map only known labels into
these two useful preference groups.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable


def _key(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


_GENRE_ALIASES = {
    "alternative": ("alternative", "alternative rock", "alt"),
    "ambient": ("ambient",),
    "blues": ("blues",),
    "classical": ("classical", "baroque", "orchestral"),
    "country": ("country",),
    "dance": ("dance", "dance pop"),
    "disco": ("disco",),
    "drum and bass": ("drum and bass", "drum n bass", "dnb"),
    "dub": ("dub",),
    "dubstep": ("dubstep",),
    "electronic": ("electronic", "electronica", "edm", "electro"),
    "emo": ("emo",),
    "experimental": ("experimental",),
    "folk": ("folk", "folktronica"),
    "funk": ("funk",),
    "gospel": ("gospel",),
    "hardcore": ("hardcore",),
    "hip hop": ("hip hop", "hip-hop", "hiphop", "rap", "hip hop rap"),
    "house": ("house", "deep house", "progressive house"),
    "indie": ("indie", "indie pop", "indie rock"),
    "jazz": ("jazz",),
    "latin": ("latin", "latino", "salsa", "reggaeton"),
    "metal": ("metal", "heavy metal", "metalcore", "death metal", "black metal"),
    "pop": ("pop", "pop rock", "synth pop", "synthpop", "k pop", "j pop"),
    "punk": ("punk", "punk rock", "post punk", "pop punk"),
    "reggae": ("reggae", "ska", "dancehall"),
    "r&b": ("r b", "r&b", "rnb", "rhythm and blues", "neo soul"),
    "rock": ("rock", "hard rock", "classic rock", "progressive rock", "psychedelic rock"),
    "singer songwriter": ("singer songwriter", "singer-songwriter"),
    "soul": ("soul",),
    "soundtrack": ("soundtrack", "soundtracks", "film score"),
    "techno": ("techno",),
    "trance": ("trance",),
    "world": ("world", "world music"),
}

_MOOD_ALIASES = {
    "aggressive": ("aggressive", "angry", "intense"),
    "calm": ("calm", "peaceful", "serene"),
    "chill": ("chill", "chillout", "laid back", "laid-back", "mellow"),
    "dark": ("dark", "gloomy"),
    "dreamy": ("dreamy",),
    "energetic": ("energetic", "high energy", "pumped up"),
    "happy": ("happy", "cheerful", "joyful"),
    "melancholy": ("melancholy", "melancholic", "melancholia"),
    "party": ("party", "club"),
    "relaxing": ("relax", "relaxing", "soothing", "sleep"),
    "romantic": ("romantic", "love songs", "sensual", "sexy"),
    "sad": ("sad", "sorrowful", "heartbroken"),
    "upbeat": ("upbeat", "uplifting", "feel good"),
}


def _alias_map(groups: dict[str, tuple[str, ...]]) -> dict[str, str]:
    return {_key(alias): canonical.strip() for canonical, aliases in groups.items()
            for alias in (canonical, *aliases)}


_GENRE_MAP = _alias_map(_GENRE_ALIASES)
_MOOD_MAP = _alias_map(_MOOD_ALIASES)


def classify_tags(tags: Iterable[str]) -> tuple[set[str], set[str]]:
    """Return recognized canonical genres and moods, ignoring other tags."""
    genres: set[str] = set()
    moods: set[str] = set()
    for tag in tags:
        key = _key(str(tag))
        if key in _GENRE_MAP:
            genres.add(_GENRE_MAP[key])
        if key in _MOOD_MAP:
            moods.add(_MOOD_MAP[key])
    return genres, moods


def extract_preferences(text: str) -> tuple[Counter[str], Counter[str]]:
    """Find known genre and mood phrases in a user's request."""
    key = f" {_key(text)} "
    found_genres: set[str] = set()
    found_moods: set[str] = set()
    for alias, canonical in _GENRE_MAP.items():
        if f" {alias} " in key:
            found_genres.add(canonical)
    for alias, canonical in _MOOD_MAP.items():
        if f" {alias} " in key:
            found_moods.add(canonical)
    return Counter({genre: 1 for genre in found_genres}), Counter(
        {mood: 1 for mood in found_moods}
    )


def profile_tracks(tracks: Iterable[dict]) -> tuple[Counter[str], Counter[str]]:
    """Count one vote per track for each recognized genre and mood."""
    genres: Counter[str] = Counter()
    moods: Counter[str] = Counter()
    for track in tracks:
        tags = track.get("tag_list") or []
        if isinstance(tags, str):
            tags = tags.split(",")
        track_genres, track_moods = classify_tags(tags)
        genres.update(track_genres)
        moods.update(track_moods)
    return genres, moods


def select_diverse(
    tracks: list[dict],
    limit: int,
    preferred_genres: Counter[str] | None = None,
    preferred_moods: Counter[str] | None = None,
) -> list[dict]:
    """Reorder a ranked candidate pool for preference fit and light diversity.

    Candidate order remains the relevance signal. Preference matches and
    genre/artist coverage influence greedy selection from that pool.
    """
    if limit <= 0:
        return []
    genres_pref = preferred_genres or Counter()
    moods_pref = preferred_moods or Counter()
    total_genres = max(1, sum(genres_pref.values()))
    total_moods = max(1, sum(moods_pref.values()))
    remaining = list(enumerate(tracks))
    selected: list[dict] = []
    covered_genres: set[str] = set()
    covered_artists: set[str] = set()
    pool_size = max(1, len(tracks))

    while remaining and len(selected) < min(limit, len(tracks)):
        best_index = 0
        best_score = float("-inf")
        best_features = (set(), set(), "")
        for position, (original_rank, track) in enumerate(remaining):
            tags = track.get("tag_list") or []
            if isinstance(tags, str):
                tags = tags.split(",")
            genres, moods = classify_tags(tags)
            artist_values = track.get("artist_name") or []
            if isinstance(artist_values, str):
                artist_values = artist_values.split(",")
            artist = _key(str(artist_values[0])) if artist_values else ""
            genre_fit = sum(count for genre, count in genres_pref.items() if genre in genres) / total_genres
            mood_fit = sum(count for mood, count in moods_pref.items() if mood in moods) / total_moods
            new_genres = sum(1 for genre in genres & set(genres_pref) if genre not in covered_genres)
            relevance = 1.0 - original_rank / pool_size
            score = (
                relevance
                + 0.22 * genre_fit
                + 0.12 * (mood_fit if moods_pref else 0.0)
                + 0.05 * new_genres
                + (0.10 if artist and artist not in covered_artists else 0.0)
                - (0.14 if artist and artist in covered_artists else 0.0)
            )
            if score > best_score:
                best_index = position
                best_score = score
                best_features = (genres, moods, artist)
        _, chosen = remaining.pop(best_index)
        selected.append(chosen)
        genres, _, artist = best_features
        covered_genres.update(genres & set(genres_pref))
        if artist:
            covered_artists.add(artist)
    return selected
