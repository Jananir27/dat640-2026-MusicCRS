import { useContext, useState } from "react";
import { MDBIcon } from "mdb-react-ui-kit";
import { FeedbackStatusContext } from "../contexts/FeedbackStatusContext";

export default function Feedback({
  message,
  on_feedback,
  label = "recommendation",
  batchId,
}: {
  message: string;
  label?: string;
  batchId?: string;
  on_feedback: (message: string, event: string) => void;
}): JSX.Element {
  const [localLiked, setLiked] = useState<boolean | null>(null);
  const status = useContext(FeedbackStatusContext);
  const tracked = status !== undefined && batchId !== undefined;
  const vote = status?.votes[message];
  const liked = tracked ? (vote === 1 ? true : vote === -1 ? false : null) : localLiked;
  const disabled = tracked && !status!.validBatchIds.includes(batchId!);
  const rate = (positive: boolean) => {
    if (disabled) return;
    if (!tracked) setLiked(positive);
    on_feedback(message, positive ? "thumbs_up" : "thumbs_down");
  };

  return (
    <span className="d-inline-flex gap-2">
      <button type="button" className="btn btn-sm btn-outline-success" disabled={disabled}
        aria-label={`Like ${label}`} aria-pressed={liked === true} onClick={() => rate(true)}>
        <MDBIcon className={`${liked ? "fas" : "far"}`} icon="thumbs-up" />
      </button>
      <button type="button" className="btn btn-sm btn-outline-danger" disabled={disabled}
        aria-label={`Dislike ${label}`} aria-pressed={liked === false} onClick={() => rate(false)}>
        <MDBIcon
          className={`${liked !== null && !liked ? "fas" : "far"}`}
          icon="thumbs-down"
        />
      </button>
    </span>
  );
}
