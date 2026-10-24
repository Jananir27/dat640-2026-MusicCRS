"""Part 4 R3: explanations backed by catalog metadata and recorded ranking evidence."""
from __future__ import annotations

from .music_features import classify_tags, extract_preferences, track_style
from .relevance_feedback import name_key, values


def explain_recommendation(track, query="", playlist=(), feedback=None, evidence=None, catalog=None):
    """Return a readable reason and machine-readable evidence; never invent attributes."""
    tags = track.get("tag_list") or []
    genres, moods = classify_tags(tags.split(",") if isinstance(tags, str) else tags)
    requested_genres, requested_moods = extract_preferences(query)
    matched_genres = sorted(genres & set(requested_genres))
    matched_moods = sorted(moods & set(requested_moods))
    artist_names = {name_key(a) for a in values(track.get("artist_name"))}
    seed_artists = {name_key(a) for seed in playlist for a in values(seed.get("artist_name"))}
    seed_styles = {style for seed in playlist for style, weight in track_style(seed).items() if weight > 1}
    shared_styles = sorted(set(track_style(track)) & seed_styles)
    seed_genres = {genre for seed in playlist for genre in classify_tags(values(seed.get("tag_list")))[0]}
    shared_genres = sorted(genres & seed_genres)
    reasons = []
    facts = {"requested_genres": matched_genres, "requested_moods": matched_moods,
             "shared_styles": shared_styles, "shared_genres": shared_genres,
             "shared_playlist_artist": bool(artist_names & seed_artists)}
    if matched_genres:
        reasons.append("matches your requested genres: " + ", ".join(matched_genres))
    if matched_moods:
        reasons.append("matches your requested moods: " + ", ".join(matched_moods))
    if facts["shared_playlist_artist"]:
        reasons.append("by an artist already in your playlist")
    if shared_styles:
        reasons.append("shares styles: " + ", ".join(shared_styles[:3]))
    elif shared_genres:
        reasons.append("shares genres: " + ", ".join(shared_genres[:3]))
    evidence = evidence or {}
    liked_id = evidence.get("liked_track_id")
    if (feedback and catalog and feedback.votes.get(liked_id) == 1 and liked_id in catalog
            and evidence.get("feedback_applied") and evidence.get("liked_similarity", 0) >= 0.5):
        liked = catalog[liked_id]
        label = ", ".join(values(liked.get("track_name"))) or liked_id
        artist = ", ".join(values(liked.get("artist_name")))
        reasons.append(f"has similar catalog metadata to a song you liked: {label} — {artist}")
        facts["liked_track_id"] = liked_id
        facts["liked_similarity"] = evidence["liked_similarity"]
    if feedback and artist_names & {name_key(a) for a in feedback.liked_artists}:
        reasons.append("by an artist you liked")
        facts["liked_artist"] = True
    if evidence.get("feedback_applied"):
        facts["feedback_delta"] = evidence.get("updated_cosine", 0) - evidence.get("original_cosine", 0)
    if not reasons:
        reasons.append("selected by the catalog retrieval ranking")
        facts["retrieval_fallback"] = True
    return "; ".join(reasons), facts
