import { createContext } from "react";

export type FeedbackStatus = {
  votes: Record<string, 1 | -1>;
  validBatchIds: string[];
};

// Undefined preserves the standalone component's local behaviour. The chat
// supplies canonical server state, including resets and expired batches.
export const FeedbackStatusContext = createContext<FeedbackStatus | undefined>(undefined);

export function parseFeedbackStatus(text: string): FeedbackStatus | undefined {
  try {
    const value = JSON.parse(text);
    if (!value || !value.votes || typeof value.votes !== "object" || Array.isArray(value.votes)
        || !Object.values(value.votes).every(vote => vote === 1 || vote === -1)
        || !Array.isArray(value.valid_batch_ids)
        || !value.valid_batch_ids.every((id: unknown) => typeof id === "string")) {
      return undefined;
    }
    return { votes: value.votes, validBatchIds: value.valid_batch_ids };
  } catch {
    return undefined;
  }
}
