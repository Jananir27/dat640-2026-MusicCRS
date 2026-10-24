"""MusicCRS conversational agent."""

import json
import os
from uuid import UUID

from flask import g, has_request_context, request

import ollama
from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.intent import Intent
from dialoguekit.core.slot_value_annotation import SlotValueAnnotation
from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.agent import Agent
from dialoguekit.participant.participant import DialogueParticipant
from dialoguekit.platforms import FlaskSocketPlatform
from dialoguekit.platforms.flask_socket_platform import ChatNamespace

from .music_intelligence import MusicIntelligence
from .music_service import ConversationState, get_shared_music_service
from .playlist_store import PlaylistStore

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.ux.uis.no")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.3:70b")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")

_INTENT_OPTIONS = Intent("OPTIONS")


class MusicChatNamespace(ChatNamespace):
    def on_feedback(self, data):
        """Handle this app's track/batch feedback rather than DialogueKit's no-op."""
        if not isinstance(data, dict) or len(json.dumps(data)) > 5000:
            return {"ok": False, "error": "Invalid feedback payload"}
        if not all(key in data for key in ("batch_id", "track_id", "event")):
            return {"ok": False, "error": "Select a song's feedback button"}
        self._platform.message(request.sid, "/feedback " + json.dumps(data))
        return {"ok": True}

    def on_connect(self, auth=None):
        device_id = (auth or {}).get("device_id") if isinstance(auth, dict) else None
        if device_id is not None:
            try:
                g.musiccrs_device_id = str(UUID(device_id))
            except (ValueError, TypeError, AttributeError):
                return False
        super().on_connect()

    def on_disconnect(self, reason=None):
        super().on_disconnect()


class MusicCRSPlatform(FlaskSocketPlatform):
    """Run the assignment's Socket.IO server with its Werkzeug dev server."""

    def start(self, host: str = "127.0.0.1", port: str = "5000") -> None:
        self.socketio.on_namespace(MusicChatNamespace("/", self))
        self.socketio.run(
            self.app,
            host=host,
            port=port,
            allow_unsafe_werkzeug=True,
        )

    def get_new_agent(self):
        agent = super().get_new_agent()
        device_id = getattr(g, "musiccrs_device_id", None) if has_request_context() else None
        if device_id:
            if not hasattr(self, "_playlist_store"):
                self._playlist_store = PlaylistStore()
            agent.attach_device(device_id, self._playlist_store)
        return agent


class MusicCRS(Agent):
    def __init__(self, id: str = "MusicCRS", *, use_llm: bool | None = None):
        """Initialize MusicCRS agent."""
        # DialogueKit's Platform.get_new_agent() passes the class name first.
        super().__init__(id=id)

        if use_llm is None:
            setting = os.environ.get("MUSICCRS_USE_LLM")
            use_llm = bool(OLLAMA_API_KEY) if setting is None else setting.casefold() not in {
                "0", "false", "no", "off",
            }

        if use_llm:
            if not OLLAMA_API_KEY:
                raise RuntimeError(
                    "MUSICCRS_USE_LLM is enabled, but OLLAMA_API_KEY is not set."
                )
            self._llm = ollama.Client(
                host=OLLAMA_HOST,
                headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
            )
        else:
            self._llm = None

        # Catalog/retrieval data is shared by the server process; mutable
        # playlist and pending-choice state belongs to this connected user.
        self._music_service = get_shared_music_service()
        self._conversation_state = ConversationState()
        self._music_intelligence = MusicIntelligence(self._music_service)
        self._playlist_store = None
        self._device_id = None
        self._active_playlist_id = None

    def attach_device(self, device_id, store):
        self._device_id = device_id
        self._playlist_store = store
        self._restore_playlist()

    def _restore_playlist(self, keep_selection=False):
        saved = None
        if keep_selection and self._active_playlist_id:
            try:
                saved = self._playlist_store.get(self._device_id, self._active_playlist_id)
            except ValueError:
                pass  # Another tab may have deleted it; open the remaining playlist.
        if saved is None:
            saved = self._playlist_store.active(self._device_id)
        if saved["id"] != self._active_playlist_id:
            self._conversation_state = ConversationState(feedback=self._conversation_state.feedback)
        self._active_playlist_id = saved["id"]
        self._conversation_state.playlist_ids = [
            track_id for track_id in saved["track_ids"]
            if self._music_service.get_track(track_id) is not None
        ]

    def _library_dialogue_act(self):
        return DialogueAct(intent=Intent("SAVED_PLAYLISTS"), annotations=[
            SlotValueAnnotation("library_json", json.dumps({
                "active_id": self._active_playlist_id,
                "playlists": self._playlist_store.library(self._device_id),
            }, ensure_ascii=False))
        ])

    def welcome(self) -> None:
        """Sends the agent's welcome message."""
        utterance = AnnotatedUtterance(
            "Hello, I'm MusicCRS. What are you in the mood for?",
            participant=DialogueParticipant.AGENT,
            dialogue_acts=([self._playlist_updated_dialogue_act(), self._library_dialogue_act()]
                           if self._playlist_store else []),
        )
        self._dialogue_connector.register_agent_utterance(utterance)

    def goodbye(self) -> None:
        """Quits the conversation."""
        utterance = AnnotatedUtterance(
            "It was nice talking to you. Bye",
            dialogue_acts=[DialogueAct(intent=self.stop_intent)],
            participant=DialogueParticipant.AGENT,
        )
        self._dialogue_connector.register_agent_utterance(utterance)

    def receive_utterance(self, utterance: Utterance) -> None:
        """Gets called each time there is a new user utterance.

        For now the agent only understands specific command.

        Args:
            utterance: User utterance.
        """
        response = ""
        dialogue_acts = []
        if self._playlist_store:
            self._restore_playlist(keep_selection=True)
        self._conversation_state.created_playlist_name = None
        self._conversation_state.emit_recommendations = False
        if utterance.text.startswith("/info"):
            response = self._info()
        elif utterance.text.startswith("/ask_llm "):
            prompt = utterance.text[9:]
            response = self._ask_llm(prompt)
        elif utterance.text.startswith("/options"):
            options = [
                "Play some jazz music",
                "Recommend me some pop songs",
                "Create a workout playlist",
            ]
            response = self._options(options)
            dialogue_acts = [
                DialogueAct(
                    intent=_INTENT_OPTIONS,
                    annotations=[
                        SlotValueAnnotation("option", option) for option in options
                    ],
                )
            ]
        elif utterance.text == "/quit":
            self.goodbye()
            return
        elif utterance.text.startswith("/playlist"):
            response, include_playlist = self._handle_playlist_command(utterance.text)
            if include_playlist:
                dialogue_acts = [self._playlist_updated_dialogue_act()]
        else:
            handled = self._music_intelligence.handle(
                utterance.text,
                self._conversation_state,
            )
            if handled is None:
                response = "I'm sorry, I don't understand that request."
            else:
                response, playlist_updated = handled
                if playlist_updated:
                    dialogue_acts = [self._playlist_updated_dialogue_act()]

        if self._conversation_state.emit_recommendations:
            dialogue_acts.append(DialogueAct(intent=Intent("RECOMMENDATIONS"), annotations=[
                SlotValueAnnotation("recommendations_json", json.dumps(
                    self._conversation_state.recommendation_payload, ensure_ascii=False))
            ]))

        if self._playlist_store:
            if any(str(act.intent) == "PLAYLIST_UPDATED" for act in dialogue_acts):
                name = self._conversation_state.created_playlist_name
                if name:
                    self._active_playlist_id = self._playlist_store.create(
                        self._device_id, name, self._conversation_state.playlist_ids)
                    self._conversation_state.created_playlist_name = None
                else:
                    self._playlist_store.save(self._device_id, self._active_playlist_id,
                                              self._conversation_state.playlist_ids)
            # Sync any edits made to this playlist in another tab.
            if not any(str(act.intent) == "PLAYLIST_UPDATED" for act in dialogue_acts):
                dialogue_acts.append(self._playlist_updated_dialogue_act())
            dialogue_acts.append(self._library_dialogue_act())

        self._dialogue_connector.register_agent_utterance(
            AnnotatedUtterance(
                response,
                participant=DialogueParticipant.AGENT,
                dialogue_acts=dialogue_acts,
            )
        )

    # --- Response handlers ---

    def _handle_playlist_command(self, command: str) -> tuple[str, bool]:
        """Handle ID-based playlist actions sent by the web playlist panel."""
        parts = command.strip().split(maxsplit=2)
        if len(parts) < 2:
            return "Use /playlist add <track_id>, remove <track_id>, clear, or show.", False

        action = parts[1].casefold()
        try:
            if action in {"new", "open", "rename", "delete", "saved"}:
                if not self._playlist_store:
                    return "Saved playlists are available in the web app on your device.", False
                argument = parts[2].strip() if len(parts) == 3 else ""
                if action == "new":
                    self._playlist_store.create(self._device_id, argument or "Untitled playlist")
                    self._restore_playlist()
                    return "Created a new empty playlist. Changes are saved automatically.", True
                if action == "open" and argument:
                    self._playlist_store.open(self._device_id, argument)
                    self._restore_playlist()
                    return "Opened your saved playlist. You can add or remove songs.", True
                if action == "rename" and argument:
                    self._playlist_store.rename(self._device_id, self._active_playlist_id, argument)
                    return f"Renamed your playlist to ‘{argument}’.", True
                if action == "delete" and argument:
                    self._playlist_store.delete(self._device_id, argument)
                    self._restore_playlist()
                    return "Deleted the saved playlist.", True
                if action == "saved":
                    names = [p["name"] for p in self._playlist_store.library(self._device_id)]
                    return "Your saved playlists: " + "; ".join(names), True
                return "Choose a saved playlist or enter a playlist name.", False
            if action == "show":
                records = self._music_service.get_playlist_records(
                    self._conversation_state
                )
                if not records:
                    response = "Your playlist is empty."
                else:
                    response = "Your playlist:\n" + "\n".join(
                        f"{index}. {record['track_name']} — {record['artist_name']}"
                        for index, record in enumerate(records, start=1)
                    )
                return response, True
            if action == "clear":
                self._music_service.clear_playlist(self._conversation_state)
                return "Your playlist is now empty.", True
            if action in {"add", "remove"} and len(parts) == 3:
                track_id = parts[2].strip()
                track = self._music_service.get_track(track_id)
                if track is None:
                    return "That track ID is not in the catalog, so I didn't change your playlist.", False
                if action == "add":
                    self._music_service.add_tracks(
                        self._conversation_state,
                        [track_id],
                    )
                    return f"Added {self._music_service.display_track(track)} to your playlist.", True
                self._music_service.remove_tracks(
                    self._conversation_state,
                    [track_id],
                )
                return f"Removed {self._music_service.display_track(track)} from your playlist.", True
        except ValueError as error:
            return str(error), False

        return "Use /playlist add <track_id>, remove <track_id>, clear, or show.", False

    def _playlist_updated_dialogue_act(self) -> DialogueAct:
        records = self._music_service.get_playlist_records(self._conversation_state)
        return DialogueAct(
            intent=Intent("PLAYLIST_UPDATED"),
            annotations=[
                SlotValueAnnotation(
                    "playlist_json",
                    json.dumps(records, ensure_ascii=False),
                )
            ],
        )

    def _info(self) -> str:
        """Gives information about the agent."""
        return "I am MusicCRS, a conversational recommender system for music."

    def _ask_llm(self, prompt: str) -> str:
        """Calls a large language model (LLM) with the given prompt.

        Args:
            prompt: Prompt to send to the LLM.

        Returns:
            Response from the LLM.
        """
        if not self._llm:
            return "The agent is not configured to use an LLM"

        llm_response = self._llm.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={
                "stream": False,
                "temperature": 0.7,  # optional: controls randomness
                "max_tokens": 100,  # optional: limits the length of the response
            },
        )

        return f"LLM response: {llm_response['response']}"

    def _options(self, options: list[str]) -> str:
        """Presents options to the user."""
        return (
            "Here are some options:\n<ol>\n"
            + "\n".join([f"<li>{option}</li>" for option in options])
            + "</ol>\n"
        )


if __name__ == "__main__":
    from retrieval.final import load_artifact

    load_artifact()
    music_service = get_shared_music_service()
    music_service._get_retriever()
    music_service.warm_r5()
    platform = MusicCRSPlatform(MusicCRS)
    platform.start(
        host=os.environ.get("MUSICCRS_HOST", "127.0.0.1"),
        port=os.environ.get("MUSICCRS_PORT", "5000"),
    )
