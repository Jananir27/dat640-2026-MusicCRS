import { fireEvent, render, screen } from "@testing-library/react";
import PlaylistPanel from "../components/PlaylistPanel";

const library = {
  active_id: "first",
  playlists: [
    { id: "first", name: "Gym", track_count: 1 },
    { id: "second", name: "Commute", track_count: 0 },
  ],
};

test("opens saved playlists and submits names without losing the selected ID", () => {
  const command = jest.fn();
  render(<PlaylistPanel tracks={[]} library={library} onLibraryAction={command}
    onClear={jest.fn()} onRemove={jest.fn()} />);
  fireEvent.change(screen.getByLabelText("Saved playlists"), { target: { value: "second" } });
  expect(command).toHaveBeenLastCalledWith("/playlist open second");
  fireEvent.change(screen.getByLabelText("Playlist name"), { target: { value: "My workouts" } });
  fireEvent.click(screen.getByRole("button", { name: "Rename" }));
  expect(command).toHaveBeenLastCalledWith("/playlist rename My workouts");
  fireEvent.click(screen.getByRole("button", { name: "New playlist" }));
  expect(command).toHaveBeenLastCalledWith("/playlist new");
});

test("deletion requires a second action and disconnected controls cannot edit", () => {
  const command = jest.fn();
  const props = { tracks: [], library, onLibraryAction: command,
    onClear: jest.fn(), onRemove: jest.fn() };
  const { rerender } = render(<PlaylistPanel {...props} />);
  fireEvent.click(screen.getByRole("button", { name: "Delete playlist" }));
  expect(command).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(command).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Delete playlist" }));
  fireEvent.click(screen.getByRole("button", { name: "Delete", exact: true }));
  expect(command).toHaveBeenCalledWith("/playlist delete first");
  rerender(<PlaylistPanel {...props} connected={false} />);
  expect(screen.getByLabelText("Saved playlists")).toBeDisabled();
  expect(screen.getByRole("button", { name: "New playlist" })).toBeDisabled();
});
