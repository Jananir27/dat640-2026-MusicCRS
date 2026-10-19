"""Persistent SQLite catalog imported from the TalkPlay Challenge Dataset.

The database supplies track records and playlist validation. A read snapshot
retains the existing metadata_dict interface for retrieval and NLU indexes.
"""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from datasets import load_dataset

DEFAULT_DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
DEFAULT_SPLIT_TYPES = ["all_tracks"]
CATALOG_SCHEMA_VERSION = 1


class MusicCatalogLoader:
    """Import once, reuse the database offline, and look up tracks by ID."""

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET_NAME,
        split_types: list[str] = DEFAULT_SPLIT_TYPES,
        *,
        database_path: str | Path | None = None,
    ) -> None:
        """Load and combine metadata splits from the configured dataset.

        Args:
            dataset_name: Hugging Face dataset name containing track metadata.
            split_types: Dataset splits to load and concatenate.
            database_path: SQLite path; defaults to cache/catalog.sqlite3 for
                the standard catalog. MUSICCRS_CATALOG_DB can override it.
        """
        source = json.dumps([dataset_name, list(split_types)])
        filename = "catalog.sqlite3"
        if dataset_name != DEFAULT_DATASET_NAME or list(split_types) != DEFAULT_SPLIT_TYPES:
            filename = f"catalog-{hashlib.sha256(source.encode()).hexdigest()[:12]}.sqlite3"
        default_path = Path(__file__).resolve().parents[1] / "cache" / filename
        self.database_path = Path(
            database_path or os.environ.get("MUSICCRS_CATALOG_DB") or default_path
        ).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        with closing(sqlite3.connect(self.database_path, timeout=300)) as connection:
            # The import and its completion marker commit together. Concurrent
            # startup processes wait for this transaction instead of importing
            # partial catalogs or overwriting each other's rows.
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS tracks ("
                    "track_id TEXT PRIMARY KEY, metadata_json TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS catalog_metadata ("
                    "id INTEGER PRIMARY KEY CHECK (id = 1), source TEXT NOT NULL, "
                    "schema_version INTEGER NOT NULL, track_count INTEGER NOT NULL)"
                )
                saved = connection.execute(
                    "SELECT source, schema_version, track_count FROM catalog_metadata WHERE id = 1"
                ).fetchone()
                if saved is None:
                    dataset = load_dataset(dataset_name)
                    for split_type in split_types:
                        connection.executemany(
                            "INSERT INTO tracks(track_id, metadata_json) VALUES (?, ?) "
                            "ON CONFLICT(track_id) DO UPDATE SET metadata_json = excluded.metadata_json",
                            ((item["track_id"], json.dumps(item, ensure_ascii=False))
                             for item in dataset[split_type]),
                        )
                    count = connection.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
                    if not count:
                        raise ValueError("The TalkPlay catalog import produced no tracks.")
                    connection.execute(
                        "INSERT INTO catalog_metadata VALUES (1, ?, ?, ?)",
                        (source, CATALOG_SCHEMA_VERSION, count),
                    )
                elif saved[:2] != (source, CATALOG_SCHEMA_VERSION):
                    raise ValueError(
                        "Catalog database belongs to a different dataset, split, or schema. "
                        "Choose a separate MUSICCRS_CATALOG_DB path."
                    )
                elif saved[2] != connection.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]:
                    raise ValueError("Catalog database track count does not match its import metadata.")
            # Keep dataset order stable for the existing retrieval indexes.
            self.metadata_dict = {
                track_id: json.loads(metadata)
                for track_id, metadata in connection.execute(
                    "SELECT track_id, metadata_json FROM tracks ORDER BY rowid"
                )
            }

    def id_to_metadata(self, track_id: str) -> dict:
        """Read a catalog record using a parameterized, read-only SQL query.

        Connections belong to each call, so Socket.IO threads never share a
        SQLite connection. Unknown IDs retain the mapping's KeyError contract.
        """
        with closing(sqlite3.connect(self.database_path.as_uri() + "?mode=ro", uri=True)) as connection:
            row = connection.execute(
                "SELECT metadata_json FROM tracks WHERE track_id = ?", (track_id,)
            ).fetchone()
        if row is None:
            raise KeyError(track_id)
        return json.loads(row[0])

    def id_to_metadata_str(
        self, track_id: str, corpus_types: list[str]
    ) -> str:
        """Formats a track's metadata fields as a single string.

        Args:
            track_id: The track ID to look up.
            corpus_types: Metadata fields to include (e.g., `track_name`).

        Returns:
            A comma-separated "field: value" string, prefixed with the
            track ID.
        """
        metadata = self.metadata_dict[track_id]
        parts = [f"track_id: {track_id}"]
        for corpus_type in corpus_types:
            value = metadata[corpus_type]
            if isinstance(value, list):
                value = ", ".join(value)
            parts.append(f"{corpus_type}: {value}")
        return ", ".join(parts)
