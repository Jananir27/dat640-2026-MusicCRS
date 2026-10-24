import unittest
from types import SimpleNamespace
from musiccrs.llm_recommendations import recommend_with_llm


class StubParser:
    def parse(self, request):
        return {
            "genre": ["rock"], "mood": ["energetic"],
            "preferred_artists": ["Green Day"],
            "excluded_artists": ["Linkin Park"],
            "excluded_tracks": ["No Surprises"],
            "use_playlist_context": True,
            "avoid_playlist_artists": False,
        }


class StubService:
    def __init__(self):
        self.tracks = [
            {"track_id": "a", "track_name": ["A Song"], "artist_name": ["Linkin Park"]},
            {"track_id": "b", "track_name": ["No Surprises"], "artist_name": ["Radiohead"]},
            {"track_id": "c", "track_name": ["Basket Case"], "artist_name": ["Green Day"]},
            {"track_id": "d", "track_name": ["X Song"], "artist_name": ["Billy Talent"]},
        ]
        self.last_query = None
        self.last_excluded = None

    def recommend_tracks(self, query, *, exclude_ids, limit, state, request, use_playlist_context):
        self.last_query = query
        self.last_excluded = exclude_ids
        assert use_playlist_context
        return self.tracks[:limit]

    def get_track(self, track_id):
        return next((t for t in self.tracks if t["track_id"] == track_id), None)


class TestIntegration(unittest.TestCase):
    def test_structured_query_and_exclusions(self):
        service = StubService()
        state = SimpleNamespace(playlist_ids=["d"])
        result = recommend_with_llm("example request", StubParser(), service, state,
                                    limit=3, candidate_limit=4)
        self.assertEqual([x["track_id"] for x in result["recommendations"]], ["c"])
        self.assertNotIn("Linkin Park", service.last_query)
        self.assertIn("Green Day", service.last_query)
        self.assertEqual(service.last_excluded, {"d"})


if __name__ == "__main__":
    unittest.main()

