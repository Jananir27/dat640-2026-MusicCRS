import React from "react";
import { PlaylistTrack } from "../types";

interface PlaylistPanelProps {
  tracks: PlaylistTrack[];
  onRemove: (trackId: string) => void;
  onClear: () => void;
}

export default function PlaylistPanel({
  tracks,
  onRemove,
  onClear,
}: PlaylistPanelProps) {
  return (
    <div className="playlist-panel">
      <div className="playlist-header">
        <div>
          <h5>My Playlist</h5>
          <small>
            {tracks.length} {tracks.length === 1 ? "song" : "songs"}
          </small>
        </div>

        {tracks.length > 0 && (
          <button
            className="btn btn-sm btn-outline-danger"
            onClick={onClear}
          >
            Clear all
          </button>
        )}
      </div>

      {tracks.length === 0 ? (
        <div className="playlist-empty">
          Your playlist is empty.
          <br />
          Try saying "Add Numb by Linkin Park".
        </div>
      ) : (
        <div className="playlist-tracks">
          {tracks.map((track) => (
            <div
              className="playlist-track"
              key={track.track_id}
            >
              <div className="playlist-track-info">
                <strong>
                  {track.track_name || track.track_id}
                </strong>

                {track.artist_name && (
                  <div>{track.artist_name}</div>
                )}

                {track.album_name && (
                  <small>{track.album_name}</small>
                )}
              </div>

              <button
                className="btn btn-sm btn-outline-danger"
                onClick={() => onRemove(track.track_id)}
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
