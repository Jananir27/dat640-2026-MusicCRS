import React, { useEffect, useState } from "react";
import { PlaylistTrack, PlaylistLibrary } from "../types";

interface PlaylistPanelProps {
  tracks: PlaylistTrack[];
  onRemove: (trackId: string) => void;
  onClear: () => void;
  library?: PlaylistLibrary | null;
  connected?: boolean;
  onLibraryAction?: (command: string) => void;
}

export default function PlaylistPanel({
  tracks,
  onRemove,
  onClear,
  library,
  connected = true,
  onLibraryAction,
}: PlaylistPanelProps) {
  const active = library?.playlists.find((item) => item.id === library.active_id);
  const [name, setName] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  useEffect(() => {
    setName(active?.name || "");
    setConfirmDelete(false);
  }, [active?.id, active?.name]);

  return (
    <div className="playlist-panel">
      {library && onLibraryAction && (
        <div className="playlist-library">
          <label htmlFor="saved-playlists">Saved playlists</label>
          <div className="playlist-library-row">
            <select id="saved-playlists" className="form-select" value={library.active_id}
              disabled={!connected}
              onChange={(event) => onLibraryAction(`/playlist open ${event.target.value}`)}>
              {library.playlists.map((item) => (
                <option key={item.id} value={item.id}>{item.name} ({item.track_count})</option>
              ))}
            </select>
            <button className="btn btn-sm btn-outline-info" disabled={!connected}
              onClick={() => onLibraryAction("/playlist new")}>New playlist</button>
          </div>
          <form className="playlist-library-row" onSubmit={(event) => {
            event.preventDefault();
            if (name.trim()) onLibraryAction(`/playlist rename ${name.trim()}`);
          }}>
            <input aria-label="Playlist name" className="form-control" value={name} maxLength={100}
              onChange={(event) => setName(event.target.value)} />
            <button className="btn btn-sm btn-outline-info" type="submit"
              disabled={!connected || !name.trim() || name.trim() === active?.name}>Rename</button>
          </form>
          <div className="playlist-library-footer">
            <small>{connected ? "Saves automatically" : "Reconnecting…"}</small>
            <button className="btn btn-sm btn-link text-danger" disabled={!connected}
              onClick={() => setConfirmDelete(true)}>Delete playlist</button>
          </div>
          {confirmDelete && (
            <div role="alert" className="playlist-delete-confirmation">
              <p>Delete “{active?.name}”?</p>
              <button className="btn btn-sm btn-danger" disabled={!connected} onClick={() => {
                onLibraryAction(`/playlist delete ${library.active_id}`);
                setConfirmDelete(false);
              }}>Delete</button>
              <button className="btn btn-sm btn-light" onClick={() => setConfirmDelete(false)}>Cancel</button>
            </div>
          )}
        </div>
      )}
      <div className="playlist-header">
        <div>
          <h5>{active?.name || "My Playlist"}</h5>
          <small>
            {tracks.length} {tracks.length === 1 ? "song" : "songs"}
          </small>
        </div>

        {tracks.length > 0 && (
          <button
            className="btn btn-sm btn-outline-danger"
            onClick={onClear}
            disabled={!connected}
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
                disabled={!connected}
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
