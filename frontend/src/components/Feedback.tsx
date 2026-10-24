import { useState } from "react";
import { MDBIcon } from "mdb-react-ui-kit";

export default function Feedback({
  message,
  on_feedback,
  label = "recommendation",
}: {
  message: string;
  label?: string;
  on_feedback: (message: string, event: string) => void;
}): JSX.Element {
  const [liked, setLiked] = useState<boolean | null>(null);
  const rate = (positive: boolean) => {
    setLiked(positive);
    on_feedback(message, positive ? "thumbs_up" : "thumbs_down");
  };

  return (
    <span className="d-inline-flex gap-2">
      <button type="button" className="btn btn-sm btn-outline-success"
        aria-label={`Like ${label}`} aria-pressed={liked === true} onClick={() => rate(true)}>
        <MDBIcon className={`${liked ? "fas" : "far"}`} icon="thumbs-up" />
      </button>
      <button type="button" className="btn btn-sm btn-outline-danger"
        aria-label={`Dislike ${label}`} aria-pressed={liked === false} onClick={() => rate(false)}>
        <MDBIcon
          className={`${liked !== null && !liked ? "fas" : "far"}`}
          icon="thumbs-down"
        />
      </button>
    </span>
  );
}
