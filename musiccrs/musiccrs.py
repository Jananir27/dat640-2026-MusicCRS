"""MusicCRS conversational agent."""

import json

import ollama
from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.intent import Intent
from dialoguekit.core.slot_value_annotation import SlotValueAnnotation
from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.agent import Agent
from dialoguekit.participant.participant import DialogueParticipant
from dialoguekit.platforms import FlaskSocketPlatform

from .music_intelligence import MusicIntelligence
from .music_service import ConversationState, get_shared_music_service

OLLAMA_HOST = "https://ollama.ux.uis.no"
OLLAMA_MODEL = "llama3.3:70b"
OLLAMA_API_KEY = "SET YOUR API KEY HERE"

_INTENT_OPTIONS = Intent("OPTIONS")


class MusicCRS(Agent):
    def __init__(self, use_llm: bool = True):
        """Initialize MusicCRS agent."""
        super().__init__(id="MusicCRS")

        if use_llm:
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

    def welcome(self) -> None:
        """Sends the agent's welcome message."""
        utterance = AnnotatedUtterance(
            "Hello, I'm MusicCRS. What are you in the mood for?",
            participant=DialogueParticipant.AGENT,
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
    platform = FlaskSocketPlatform(MusicCRS)
    platform.start()
    platform.start()
