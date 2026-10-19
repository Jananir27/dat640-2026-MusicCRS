"""Persistence and actual Socket.IO transport checks without model downloads."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from musiccrs.musiccrs import MusicCRS, MusicCRSPlatform, MusicChatNamespace
from musiccrs.music_service import MusicService
from musiccrs.playlist_store import PlaylistStore
from tests.test_music_intelligence import FakeCatalog, FakeRetriever


class SavedPlaylistTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "playlists.sqlite3"
        self.store = PlaylistStore(self.path)
        self.device = str(uuid4())

    def test_store_reopens_and_preserves_order_and_active_playlist(self):
        first = self.store.active(self.device)
        self.store.save(self.device, first["id"], ["3", "1", "3"])
        second = self.store.create(self.device, "Gym", ["2"])
        store = PlaylistStore(self.path)
        self.assertEqual(store.active(self.device)["id"], second)
        store.open(self.device, first["id"])
        self.assertEqual(store.active(self.device)["track_ids"], ["3", "1"])
        self.assertEqual(len(store.library(self.device)), 2)

    def test_other_device_cannot_open_edit_rename_or_delete_a_playlist(self):
        first = self.store.active(self.device)
        other = str(uuid4())
        self.store.active(other)
        for action in (
            lambda: self.store.open(other, first["id"]),
            lambda: self.store.save(other, first["id"], ["2"]),
            lambda: self.store.rename(other, first["id"], "Changed"),
            lambda: self.store.delete(other, first["id"]),
        ):
            with self.assertRaises(ValueError):
                action()
        self.assertEqual(self.store.active(self.device), first)
        self.assertEqual(len(self.store.library(other)), 1)

    def test_rename_and_delete_active_choose_remaining_playlist(self):
        first = self.store.active(self.device)
        second = self.store.create(self.device, "Gym", ["2"])
        self.store.rename(self.device, second, "Workout")
        self.assertEqual(self.store.active(self.device)["name"], "Workout")
        self.store.delete(self.device, second)
        self.assertEqual(self.store.active(self.device)["id"], first["id"])
        self.store.delete(self.device, first["id"])
        self.assertEqual(self.store.active(self.device)["track_ids"], [])
        self.assertEqual(len(self.store.library(self.device)), 1)

    def test_invalid_name_does_not_create_or_select_a_playlist(self):
        first = self.store.active(self.device)
        for name in (" ", "x" * 101):
            with self.assertRaises(ValueError):
                self.store.create(self.device, name)
        self.assertEqual(self.store.active(self.device), first)
        self.assertEqual(len(self.store.library(self.device)), 1)


class SavedPlaylistSocketTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        service = MusicService(FakeCatalog(), FakeRetriever())
        patcher = patch("musiccrs.musiccrs.get_shared_music_service", return_value=service)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.platform = MusicCRSPlatform(MusicCRS)
        self.platform._playlist_store = PlaylistStore(Path(directory.name) / "playlists.sqlite3")
        self.platform.socketio.on_namespace(MusicChatNamespace("/", self.platform))
        self.device = str(uuid4())

    def connect(self, device=None):
        client = self.platform.socketio.test_client(self.platform.app,
                                                  auth={"device_id": device or self.device})
        self.addCleanup(lambda: client.disconnect() if client.is_connected() else None)
        return client

    def response(self, client):
        events = [e for e in client.get_received() if e["name"] == "message"]
        self.assertTrue(events)
        data = events[-1]["args"]
        if isinstance(data, list):
            data = data[0]
        message = data["message"]
        annotations = {a["slot"]: a["value"] for act in message["dialogue_acts"]
                       for a in act["annotations"]}
        return message["text"], json.loads(annotations["playlist_json"]), json.loads(annotations["library_json"])

    def send(self, client, command):
        client.emit("message", {"message": command})
        return self.response(client)

    def test_add_remove_and_clear_survive_socket_reconnection(self):
        client = self.connect()
        _, tracks, library = self.response(client)
        self.assertEqual(tracks, [])
        original = library["active_id"]
        self.send(client, "/playlist add 1")
        self.send(client, "/playlist add 3")
        client.disconnect()
        client = self.connect()
        _, tracks, library = self.response(client)
        self.assertEqual([t["track_id"] for t in tracks], ["1", "3"])
        self.assertEqual(library["active_id"], original)
        self.send(client, "/playlist remove 1")
        client.disconnect()
        client = self.connect()
        self.assertEqual([t["track_id"] for t in self.response(client)[1]], ["3"])
        self.send(client, "Clear my playlist")
        client.disconnect()
        self.assertEqual(self.response(self.connect())[1], [])

    def test_generated_playlist_is_saved_separately_and_can_be_reopened(self):
        client = self.connect()
        first = self.response(client)[2]["active_id"]
        self.send(client, "/playlist add 1")
        _, generated, library = self.send(client, "Create a playlist for energetic gym workouts")
        second = library["active_id"]
        self.assertNotEqual(second, first)
        self.assertTrue(generated)
        self.assertEqual(len(library["playlists"]), 2)
        self.send(client, "/playlist rename Gym")
        _, tracks, library = self.send(client, f"/playlist open {first}")
        self.assertEqual([t["track_id"] for t in tracks], ["1"])
        self.send(client, "/playlist add 3")
        _, tracks, _ = self.send(client, f"/playlist open {second}")
        self.assertEqual(tracks, generated)
        self.assertEqual(next(p for p in self.response_after_reconnect(client)[2]["playlists"]
                              if p["id"] == second)["name"], "Gym")

    def response_after_reconnect(self, client):
        client.disconnect()
        return self.response(self.connect())

    def test_devices_are_isolated_and_foreign_playlist_command_is_rejected(self):
        first = self.connect()
        first_id = self.response(first)[2]["active_id"]
        self.send(first, "/playlist add 1")
        second = self.connect(str(uuid4()))
        self.assertEqual(self.response(second)[1], [])
        text, tracks, library = self.send(second, f"/playlist open {first_id}")
        self.assertIn("not available", text)
        self.assertEqual(tracks, [])
        self.assertEqual(len(library["playlists"]), 1)

    def test_switching_playlist_discards_old_recommendation_selection(self):
        client = self.connect()
        self.response(client)
        self.send(client, "/playlist add 1")
        self.send(client, "Recommend something based on my playlist")
        self.send(client, "/playlist new Other")
        _, tracks, _ = self.send(client, "Add all of them")
        self.assertEqual(tracks, [])

    def test_two_tabs_keep_edits_on_the_playlist_visible_in_each_tab(self):
        first = self.connect()
        original = self.response(first)[2]["active_id"]
        second = self.connect()
        self.response(second)
        _, _, library = self.send(second, "/playlist new Second tab")
        other_id = library["active_id"]
        _, tracks, library = self.send(first, "/playlist add 1")
        self.assertEqual(library["active_id"], original)
        self.assertEqual([t["track_id"] for t in tracks], ["1"])
        _, tracks, library = self.send(second, "/playlist show")
        self.assertEqual(library["active_id"], other_id)
        self.assertEqual(tracks, [])

    def test_invalid_device_id_is_rejected(self):
        client = self.platform.socketio.test_client(self.platform.app, auth={"device_id": "invalid"})
        self.assertFalse(client.is_connected())


if __name__ == "__main__":
    unittest.main()
