import { fireEvent, render, screen } from "@testing-library/react";
import RecommendationFeedback from "../components/RecommendationFeedback";

const batch = { batch_id: "original-batch", tracks: [
  { track_id: "track-a", track_name: "Numb", artist_name: "Linkin Park", explanation: "shares rock" },
  { track_id: "track-b", track_name: "Numb", artist_name: "21 Savage", explanation: "shares hip hop" },
] };

test("ratings preserve exact track and batch identity, including same-title tracks", () => {
  const rate = jest.fn();
  const { rerender } = render(<RecommendationFeedback batch={batch} onFeedback={rate} />);
  fireEvent.click(screen.getByRole("button", { name: "Like Numb by Linkin Park" }));
  expect(rate).toHaveBeenLastCalledWith("", "thumbs_up", "track-a", "original-batch");
  expect(screen.getByRole("button", { name: "Like Numb by Linkin Park" })).toHaveAttribute("aria-pressed", "true");
  rerender(<RecommendationFeedback batch={batch} onFeedback={rate} />);
  fireEvent.click(screen.getByRole("button", { name: "Dislike Numb by 21 Savage" }));
  expect(rate).toHaveBeenLastCalledWith("", "thumbs_down", "track-b", "original-batch");
  expect(rate).toHaveBeenCalledTimes(2);
});

test("changing a vote emits one event and updates the accessible state", () => {
  const rate = jest.fn();
  render(<RecommendationFeedback batch={batch} onFeedback={rate} />);
  fireEvent.click(screen.getByRole("button", { name: "Like Numb by Linkin Park" }));
  fireEvent.click(screen.getByRole("button", { name: "Dislike Numb by Linkin Park" }));
  expect(rate).toHaveBeenCalledTimes(2);
  expect(screen.getByRole("button", { name: "Like Numb by Linkin Park" })).toHaveAttribute("aria-pressed", "false");
  expect(screen.getByRole("button", { name: "Dislike Numb by Linkin Park" })).toHaveAttribute("aria-pressed", "true");
});
