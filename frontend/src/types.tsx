export type ChatMessageButton = {
  title: string;
  payload: string;
  button_type: string;
};

export type ChatMessageAttachment = {
  type: string;
  payload: {
    images?: string[];
    buttons?: ChatMessageButton[];
  };
};

// Dialogue-act structures sent by the MusicCRS backend
export type DialogueAnnotation = {
  slot: string;
  value: string;
};

export type DialogueAct = {
  intent: string;
  annotations: DialogueAnnotation[];
};

export type ChatMessage = {
  attachments?: ChatMessageAttachment[];
  text?: string;
  intent?: string;

  // Used for backend events such as PLAYLIST_UPDATED
  dialogue_acts?: DialogueAct[];
};

export type AgentMessage = {
  recipient: string;
  message: ChatMessage;
  info?: string;
};

export type UserMessage = {
  message: string;
};

// Track information displayed in the playlist panel
export type PlaylistTrack = {
  track_id: string;
  track_name?: string;
  artist_name?: string;
  album_name?: string;
};
