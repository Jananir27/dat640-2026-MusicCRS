import { fireEvent, render, screen } from "@testing-library/react";
import RecommendationFeedback from "../components/RecommendationFeedback";
import { FeedbackStatus, FeedbackStatusContext, parseFeedbackStatus } from "../contexts/FeedbackStatusContext";

const batch = { batch_id: "batch-1", tracks: [
  { track_id: "track-a", track_name: "Numb", artist_name: "Linkin Park", explanation: "rock" },
] };

test("ratings follow server votes, chat corrections and resets", () => {
  const rate = jest.fn();
  const view = (status: FeedbackStatus) => <FeedbackStatusContext.Provider value={status}>
    <RecommendationFeedback batch={batch} onFeedback={rate} />
  </FeedbackStatusContext.Provider>;
  const { rerender } = render(view({ votes: {}, validBatchIds: ["batch-1"] }));
  const like = screen.getByRole("button", { name: "Like Numb by Linkin Park" });
  const dislike = screen.getByRole("button", { name: "Dislike Numb by Linkin Park" });
  fireEvent.click(like);
  expect(rate).toHaveBeenCalledWith("", "thumbs_up", "track-a", "batch-1");
  expect(like).toHaveAttribute("aria-pressed", "false"); // wait for server acceptance
  rerender(view({ votes: { "track-a": 1 }, validBatchIds: ["batch-1"] }));
  expect(like).toHaveAttribute("aria-pressed", "true");
  rerender(view({ votes: { "track-a": -1 }, validBatchIds: ["batch-1"] }));
  expect(like).toHaveAttribute("aria-pressed", "false");
  expect(dislike).toHaveAttribute("aria-pressed", "true");
  rerender(view({ votes: {}, validBatchIds: ["batch-1"] }));
  expect(dislike).toHaveAttribute("aria-pressed", "false");
});

test("expired or disconnected batches cannot submit a new vote", () => {
  const rate = jest.fn();
  render(<FeedbackStatusContext.Provider value={{ votes: {}, validBatchIds: [] }}>
    <RecommendationFeedback batch={batch} onFeedback={rate} />
  </FeedbackStatusContext.Provider>);
  const like = screen.getByRole("button", { name: "Like Numb by Linkin Park" });
  expect(like).toBeDisabled();
  fireEvent.click(like);
  expect(rate).not.toHaveBeenCalled();
});

test("malformed feedback snapshots are ignored", () => {
  for (const text of ["not json", "null", "{}", '{"votes":[],"valid_batch_ids":[]}',
    '{"votes":{"track-a":2},"valid_batch_ids":[]}', '{"votes":{},"valid_batch_ids":[42]}']) {
    expect(parseFeedbackStatus(text)).toBeUndefined();
  }
  expect(parseFeedbackStatus('{"votes":{"track-a":-1},"valid_batch_ids":["batch-1"]}'))
    .toEqual({ votes: { "track-a": -1 }, validBatchIds: ["batch-1"] });
});
