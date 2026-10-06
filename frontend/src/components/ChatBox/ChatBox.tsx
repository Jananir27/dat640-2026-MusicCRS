import "./ChatBox.css";

import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useContext,
} from "react";

import QuickReplyButton from "../QuickReply";
import PlaylistPanel from "../PlaylistPanel";

import { useSocket } from "../../contexts/SocketContext";
import { UserContext } from "../../contexts/UserContext";
import { ConfigContext } from "../../contexts/ConfigContext";

import {
  MDBCard,
  MDBCardHeader,
  MDBCardBody,
  MDBIcon,
  MDBCardFooter,
} from "mdb-react-ui-kit";

import {
  AgentChatMessage,
  UserChatMessage,
} from "../ChatMessage";

import {
  ChatMessage,
  PlaylistTrack,
} from "../../types";


export default function ChatBox() {
  const { config } = useContext(ConfigContext);
  const { user } = useContext(UserContext);

  const {
    startConversation,
    sendMessage,
    quickReply,
    onMessage,
    onRestart,
    giveFeedback,
  } = useSocket();

  // ---------------------------------------------------------
  // Chat state
  // ---------------------------------------------------------

  const [chatMessages, setChatMessages] =
    useState<JSX.Element[]>([]);

  const [chatButtons, setChatButtons] =
    useState<JSX.Element[]>([]);

  const [inputValue, setInputValue] =
    useState<string>("");

  // ---------------------------------------------------------
  // Playlist state
  // ---------------------------------------------------------

  const [playlist, setPlaylist] =
    useState<PlaylistTrack[]>([]);

  // ---------------------------------------------------------
  // Refs
  // ---------------------------------------------------------

  const chatMessagesRef = useRef(chatMessages);

  const inputRef =
    useRef<HTMLInputElement>(null);

  // ---------------------------------------------------------
  // Start conversation
  // ---------------------------------------------------------

  useEffect(() => {
    startConversation();
  }, [startConversation]);

  // ---------------------------------------------------------
  // Add message to chat
  // ---------------------------------------------------------

  const updateMessages = useCallback(
    (message: JSX.Element) => {
      chatMessagesRef.current = [
        ...chatMessagesRef.current,
        message,
      ];

      setChatMessages(chatMessagesRef.current);
    },
    []
  );

  // ---------------------------------------------------------
  // User sends normal chat message
  // ---------------------------------------------------------

  const handleInput = (
    e: React.FormEvent<HTMLFormElement>
  ) => {
    e.preventDefault();

    const message = inputValue.trim();

    if (message === "") {
      return;
    }

    updateMessages(
      <UserChatMessage
        key={chatMessagesRef.current.length}
        message={message}
      />
    );

    sendMessage({
      message: message,
    });

    setInputValue("");

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  // ---------------------------------------------------------
  // Quick replies
  // ---------------------------------------------------------

  const handleQuickReply = useCallback(
    (message: string) => {
      updateMessages(
        <UserChatMessage
          key={chatMessagesRef.current.length}
          message={message}
        />
      );

      quickReply({
        message: message,
      });
    },
    [quickReply, updateMessages]
  );

  // ---------------------------------------------------------
  // Handle normal agent text message
  // ---------------------------------------------------------

  const handelMessage = useCallback(
    (message: ChatMessage) => {
      if (message.text) {
        const image_url =
          message.attachments?.find(
            (attachment) =>
              attachment.type === "images"
          )?.payload.images?.[0];

        updateMessages(
          <AgentChatMessage
            key={chatMessagesRef.current.length}
            feedback={
              config.useFeedback
                ? giveFeedback
                : null
            }
            message={message.text}
            image_url={image_url}
          />
        );
      }
    },
    [
      giveFeedback,
      config,
      updateMessages,
    ]
  );

  // ---------------------------------------------------------
  // Handle quick-reply buttons from agent
  // ---------------------------------------------------------

  const handleButtons = useCallback(
    (message: ChatMessage) => {
      const buttons =
        message.attachments?.find(
          (attachment) =>
            attachment.type === "buttons"
        )?.payload.buttons;

      if (buttons && buttons.length > 0) {
        setChatButtons(
          buttons.map((button, index) => (
            <QuickReplyButton
              key={index}
              text={button.title}
              message={button.payload}
              click={handleQuickReply}
            />
          ))
        );
      } else {
        setChatButtons([]);
      }
    },
    [handleQuickReply]
  );

  // ---------------------------------------------------------
  // Handle PLAYLIST_UPDATED dialogue act
  // ---------------------------------------------------------

  const handlePlaylistUpdate = useCallback(
    (message: ChatMessage) => {
      const playlistAct =
        message.dialogue_acts?.find(
          (act) =>
            act.intent === "PLAYLIST_UPDATED"
        );

      // This message does not contain a playlist update.
      if (!playlistAct) {
        return;
      }

      const playlistAnnotation =
        playlistAct.annotations?.find(
          (annotation) =>
            annotation.slot === "playlist_json"
        );

      if (!playlistAnnotation) {
        return;
      }

      try {
        const tracks = JSON.parse(
          playlistAnnotation.value
        ) as PlaylistTrack[];

        setPlaylist(tracks);
      } catch (error) {
        console.error(
          "Could not parse playlist update:",
          error
        );
      }
    },
    []
  );

  // ---------------------------------------------------------
  // Listen for messages from backend
  // ---------------------------------------------------------

  useEffect(() => {
    onMessage((message: ChatMessage) => {
      handelMessage(message);

      handleButtons(message);

      handlePlaylistUpdate(message);
    });
  }, [
    onMessage,
    handelMessage,
    handleButtons,
    handlePlaylistUpdate,
  ]);

  // ---------------------------------------------------------
  // Remove track directly from playlist UI
  // ---------------------------------------------------------

  const handleRemoveTrack = (
    trackId: string
  ) => {
    sendMessage({
      message: `/playlist remove ${trackId}`,
    });
  };

  // ---------------------------------------------------------
  // Clear playlist directly from UI
  // ---------------------------------------------------------

  const handleClearPlaylist = () => {
    sendMessage({
      message: "/playlist clear",
    });
  };

  // ---------------------------------------------------------
  // Restart conversation
  // ---------------------------------------------------------

  useEffect(() => {
    onRestart(() => {
      setChatMessages([]);

      setChatButtons([]);

      setPlaylist([]);

      chatMessagesRef.current = [];
    });
  }, [onRestart]);

  // ---------------------------------------------------------
  // UI
  // ---------------------------------------------------------

  return (
    <div className="musiccrs-layout">

      {/* =====================================================
          CHAT
         ===================================================== */}

      <div className="chat-widget-content">

        <MDBCard
          id="chatBox"
          className="chat-widget-card"
          style={{
            borderRadius: "15px",
          }}
        >

          {/* Header */}

          <MDBCardHeader
            className="
              d-flex
              justify-content-between
              align-items-center
              p-3
              bg-info
              text-white
              border-bottom-0
            "
            style={{
              borderTopLeftRadius: "15px",
              borderTopRightRadius: "15px",
            }}
          >
            <p className="mb-0 fw-bold">
              {config.name}
            </p>

            <p className="mb-0 fw-bold">
              {user?.username}
            </p>
          </MDBCardHeader>

          {/* Messages */}

          <MDBCardBody>
            <div className="card-body-messages">

              {chatMessages}

              <div
                className="
                  d-flex
                  flex-wrap
                  justify-content-between
                "
              >
                {chatButtons}
              </div>

            </div>
          </MDBCardBody>

          {/* Input */}

          <MDBCardFooter
            className="
              text-muted
              d-flex
              justify-content-start
              align-items-center
              p-2
            "
          >

            <form
              className="d-flex flex-grow-1"
              onSubmit={handleInput}
            >

              <input
                type="text"
                className="
                  form-control
                  form-control-lg
                "
                id="ChatInput"
                value={inputValue}
                onChange={(e) =>
                  setInputValue(
                    e.target.value
                  )
                }
                placeholder="Type message"
                ref={inputRef}
              />

              <button
                type="submit"
                className="
                  btn
                  btn-link
                  text-muted
                "
              >
                <MDBIcon
                  fas
                  size="2x"
                  icon="paper-plane"
                />
              </button>

            </form>

          </MDBCardFooter>

        </MDBCard>

      </div>

      {/* =====================================================
          PLAYLIST
         ===================================================== */}

      <PlaylistPanel
        tracks={playlist}
        onRemove={handleRemoveTrack}
        onClear={handleClearPlaylist}
      />

    </div>
  );
}
