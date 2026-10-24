import Feedback from "./Feedback";
import { RecommendationBatch } from "../types";

export default function RecommendationFeedback({ batch, onFeedback }: {
  batch: RecommendationBatch;
  onFeedback: (message: string, event: string, trackId?: string, batchId?: string) => void;
}) {
  return <div className="mb-3 p-2 border rounded" aria-label="Rate recommended songs">
    <p className="small mb-2">Like or dislike a song to guide your next recommendations.</p>
    {batch.tracks.map((track, index) => <div className="d-flex justify-content-between align-items-center gap-2 mb-2" key={track.track_id}>
      <span className="small">{index + 1}. {track.track_name} — {track.artist_name}</span>
      <Feedback message={track.track_id} label={`${track.track_name} by ${track.artist_name}`}
        on_feedback={(_, event) => onFeedback("", event, track.track_id, batch.batch_id)} />
    </div>)}
  </div>;
}
