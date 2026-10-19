"""Tests for deterministic playlist genre/mood preference features."""

import unittest
from collections import Counter

from musiccrs.music_features import (
    classify_tags, extract_preferences, profile_tracks, select_diverse,
    select_playlist_matches, track_style,
)


class MusicFeatureTests(unittest.TestCase):
    def test_tag_classifier_discards_non_genre_and_non_mood_tags(self):
        genres, moods = classify_tags([
            "hip-hop/rap", "Alternative Rock", "00s", "favorites", "seen live",
            "chillout", "melancholic", "8 of 10 stars",
        ])
        self.assertEqual(genres, {"hip hop", "alternative"})
        self.assertEqual(moods, {"chill", "melancholy"})

    def test_request_preferences_normalize_aliases(self):
        genres, moods = extract_preferences("Make a hip-hop and R&B playlist that feels mellow")
        self.assertEqual(genres, Counter({"hip hop": 1, "r&b": 1}))
        self.assertEqual(moods, Counter({"chill": 1}))

    def test_playlist_profile_counts_each_feature_once_per_track(self):
        genres, moods = profile_tracks([
            {"tag_list": ["rock", "hard rock", "sad", "2003"]},
            {"tag_list": ["rock", "chill", "favorites"]},
        ])
        self.assertEqual(genres, Counter({"rock": 2}))
        self.assertEqual(moods, Counter({"sad": 1, "chill": 1}))

    def test_selection_promotes_fit_and_spreads_artists(self):
        ranked = [
            {"track_id": "1", "artist_name": ["Same"], "tag_list": ["metal"]},
            {"track_id": "2", "artist_name": ["Same"], "tag_list": ["rock"]},
            {"track_id": "3", "artist_name": ["Other"], "tag_list": ["rock", "chill"]},
        ]
        selected = select_diverse(
            ranked, 2, Counter({"rock": 1}), Counter({"chill": 1}),
        )
        self.assertEqual([track["track_id"] for track in selected], ["1", "3"])
        self.assertNotEqual(selected[0]["artist_name"], selected[1]["artist_name"])
        self.assertIn("rock", classify_tags(selected[1]["tag_list"])[0])

    def test_playlist_style_preserves_subgenres_without_counting_aliases_twice(self):
        style = track_style({"tag_list": ["nu metal", "nu-metal", "new metal", "hard rock", "2003", "favorites"]})
        self.assertEqual(style, {"nu metal": 2.0, "hard rock": 2.0, "rock": 1.0})

    def test_playlist_matches_prioritize_style_over_rank_and_broad_tag_overlap(self):
        def track(track_id, artist, tags):
            return {"track_id": track_id, "artist_name": [artist], "tag_list": tags}

        seed = track("seed", "Seed Artist", ["nu metal", "alternative rock", "rock", "hip-hop", "electronic", "pop"])
        candidates = [
            track("country", "Country Artist", ["country", "pop"]),
            track("electronic", "Electronic Artist", ["electronic", "hip-hop"]),
            track("metal", "Metal Artist", ["nu-metal", "alternative rock", "rock"]),
            track("same", "Seed Artist", ["nu metal", "alternative rock", "rock"]),
            track("rock", "Rock Artist", ["hard rock", "alternative rock", "rock"]),
        ]
        selected = select_playlist_matches(candidates, [seed], 3)
        self.assertEqual({t["track_id"] for t in selected}, {"same", "metal", "rock"})

    def test_playlist_matches_do_not_fill_slots_with_unrelated_tracks(self):
        seed = {"artist_name": ["Seed"], "tag_list": ["metal"]}
        unrelated = {"track_id": "country", "artist_name": ["Other"], "tag_list": ["country"]}
        self.assertEqual(select_playlist_matches([unrelated], [seed], 5), [])

    def test_playlist_matches_support_tagless_seed_and_string_metadata(self):
        track = {"track_id": "other", "artist_name": "Artist", "tag_list": "rock, hard rock"}
        self.assertEqual(select_playlist_matches([track], [{"artist_name": ["Seed"]}], 5), [track])


if __name__ == "__main__":
    unittest.main()
