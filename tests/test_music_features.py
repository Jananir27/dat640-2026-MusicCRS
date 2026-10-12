"""Tests for deterministic playlist genre/mood preference features."""

import unittest
from collections import Counter

from musiccrs.music_features import classify_tags, extract_preferences, profile_tracks, select_diverse


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


if __name__ == "__main__":
    unittest.main()
