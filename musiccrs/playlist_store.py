"""Small SQLite playlist library, scoped to an anonymous browser ID."""

from contextlib import closing, contextmanager
import json
import os
from pathlib import Path
import sqlite3
from time import time_ns
from uuid import uuid4


class PlaylistStore:
    def __init__(self, path=None):
        catalog_path = Path(os.environ.get("MUSICCRS_CATALOG_DB", "cache/catalog.sqlite3"))
        self.path = Path(path or os.environ.get("MUSICCRS_PLAYLIST_DB") or
                         catalog_path.with_name("playlists.sqlite3"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS saved_playlists (
                    id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    track_ids TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS playlists_by_device
                    ON saved_playlists(device_id, updated_at);
                CREATE TABLE IF NOT EXISTS playlist_devices (
                    device_id TEXT PRIMARY KEY,
                    active_id TEXT NOT NULL
                );
            """)

    @contextmanager
    def _connection(self):
        with closing(sqlite3.connect(self.path, timeout=15)) as db:
            db.row_factory = sqlite3.Row
            with db:
                yield db

    @staticmethod
    def _name(name):
        name = name.strip()
        if not name or len(name) > 100:
            raise ValueError("Use a playlist name between 1 and 100 characters.")
        return name

    def _create(self, db, device_id, name, track_ids):
        playlist_id = str(uuid4())
        db.execute("INSERT INTO saved_playlists VALUES (?, ?, ?, ?, ?)",
                   (playlist_id, device_id, self._name(name),
                    json.dumps(list(dict.fromkeys(track_ids))), time_ns()))
        db.execute("INSERT INTO playlist_devices VALUES (?, ?) "
                   "ON CONFLICT(device_id) DO UPDATE SET active_id=excluded.active_id",
                   (device_id, playlist_id))
        return playlist_id

    def active(self, device_id):
        with self._connection() as db:
            # Serialize initial creation so two tabs share one default playlist.
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT p.* FROM saved_playlists p JOIN playlist_devices d "
                             "ON p.id=d.active_id AND p.device_id=d.device_id "
                             "WHERE d.device_id=?", (device_id,)).fetchone()
            if row is None:
                playlist_id = self._create(db, device_id, "My playlist", [])
                row = db.execute("SELECT * FROM saved_playlists WHERE id=?", (playlist_id,)).fetchone()
            return {"id": row["id"], "name": row["name"], "track_ids": json.loads(row["track_ids"])}

    def library(self, device_id):
        with self._connection() as db:
            rows = db.execute("SELECT * FROM saved_playlists WHERE device_id=? "
                              "ORDER BY updated_at DESC", (device_id,)).fetchall()
            return [{"id": r["id"], "name": r["name"],
                     "track_count": len(json.loads(r["track_ids"]))} for r in rows]

    def get(self, device_id, playlist_id):
        with self._connection() as db:
            row = db.execute("SELECT * FROM saved_playlists WHERE id=? AND device_id=?",
                             (playlist_id, device_id)).fetchone()
            if row is None:
                raise ValueError("That saved playlist is no longer available.")
            return {"id": row["id"], "name": row["name"], "track_ids": json.loads(row["track_ids"])}

    def create(self, device_id, name, track_ids=()):
        with self._connection() as db:
            return self._create(db, device_id, name, track_ids)

    def save(self, device_id, playlist_id, track_ids):
        with self._connection() as db:
            result = db.execute("UPDATE saved_playlists SET track_ids=?, updated_at=? "
                                "WHERE id=? AND device_id=?",
                                (json.dumps(list(dict.fromkeys(track_ids))), time_ns(), playlist_id, device_id))
            if not result.rowcount:
                raise ValueError("That saved playlist is no longer available.")

    def open(self, device_id, playlist_id):
        with self._connection() as db:
            if not db.execute("SELECT 1 FROM saved_playlists WHERE id=? AND device_id=?",
                              (playlist_id, device_id)).fetchone():
                raise ValueError("That saved playlist is not available on this device.")
            db.execute("UPDATE playlist_devices SET active_id=? WHERE device_id=?", (playlist_id, device_id))

    def rename(self, device_id, playlist_id, name):
        with self._connection() as db:
            result = db.execute("UPDATE saved_playlists SET name=?, updated_at=? WHERE id=? AND device_id=?",
                                (self._name(name), time_ns(), playlist_id, device_id))
            if not result.rowcount:
                raise ValueError("That saved playlist is no longer available.")

    def delete(self, device_id, playlist_id):
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            result = db.execute("DELETE FROM saved_playlists WHERE id=? AND device_id=?", (playlist_id, device_id))
            if not result.rowcount:
                raise ValueError("That saved playlist is not available on this device.")
            active = db.execute("SELECT active_id FROM playlist_devices WHERE device_id=?", (device_id,)).fetchone()
            if active and active["active_id"] == playlist_id:
                other = db.execute("SELECT id FROM saved_playlists WHERE device_id=? ORDER BY updated_at DESC LIMIT 1",
                                   (device_id,)).fetchone()
                if other:
                    db.execute("UPDATE playlist_devices SET active_id=? WHERE device_id=?", (other["id"], device_id))
                else:
                    self._create(db, device_id, "My playlist", [])
