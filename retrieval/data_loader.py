"""Minimal loader for the track metadata catalog.

Loads the same challenge dataset used in the RecSys 2026 MusicCRS
challenge (talkpl-ai/TalkPlayData-Challenge-Track-Metadata) from the
Hugging Face Hub and exposes it as a track_id -> metadata mapping.
"""
from datasets import load_dataset, concatenate_datasets

DEFAULT_DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
DEFAULT_SPLIT_TYPES = ["all_tracks"]


class MusicCatalogLoader:
    """Loads track metadata and provides lookup by track ID."""

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET_NAME,
        split_types: list[str] = DEFAULT_SPLIT_TYPES,
    ) -> None:
        """Load and combine metadata splits from the configured dataset.

        Args:
            dataset_name: Hugging Face dataset name containing track metadata.
            split_types: Dataset splits to load and concatenate.
        """
        dataset = load_dataset(dataset_name)
        concat_dataset = concatenate_datasets(
            [dataset[split_type] for split_type in split_types]
        )
        self.metadata_dict: dict[str, dict] = {
            item["track_id"]: item for item in concat_dataset
        }

    def id_to_metadata(self, track_id: str) -> dict:
        """Look up the metadata dict for a track ID."""
        return self.metadata_dict[track_id]

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
