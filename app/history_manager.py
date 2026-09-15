from .logger import logger 
from typing import List


class HistoryManager:
    """
    Maintains conversation history as:
      - a rolling summary (for turns that fell out of the window)
      - a verbatim sliding window of the last `window_size` turns

    Call `.add_turn()` after every exchange; it handles rollover automatically.
    """

    def __init__(self, model, window_size: int = 6):
        self.model = model
        self.window_size = window_size
        self.summary: str = ""
        self.recent_turns: List["ConversationTurn"] = []

    def add_turn(self, turn: "ConversationTurn") -> None:
        self.recent_turns.append(turn)

        if len(self.recent_turns) > self.window_size:
            overflow = self.recent_turns.pop(0)
            self._fold_into_summary(overflow)

    def _fold_into_summary(self, turn: "ConversationTurn") -> None:
        """Collapse a turn that's aging out of the window into the rolling summary."""
        prompt = f"""
        You maintain a running summary of a conversation for context purposes.

        Existing summary:
        {self.summary or "(empty, this is the first turn to summarize)"}

        New turn to fold in:
        User: {turn.query}
        Assistant: {turn.response}

        Update the summary to incorporate this turn. Keep it concise (2-5 sentences),
        preserve key facts, preferences, and unresolved questions. Do not include
        any preamble, just return the updated summary text.
        """
        try:
            updated = self.model.invoke_model(full_prompt=prompt)
            self.summary = updated.strip()
            logger.info("Rolled up one turn into rolling summary")
        except Exception as e:
            # Don't lose the turn's content entirely if summarization fails —
            # fall back to a crude append rather than silently dropping it.
            logger.error(f"Summary rollover failed: {e}")
            self.summary = (self.summary + f"\nUser previously asked: {turn.query}").strip()

    def get_context(self) -> str:
        """Formatted block to drop into a prompt."""
        parts = []
        if self.summary:
            parts.append(f"Summary of earlier conversation:\n{self.summary}")
        if self.recent_turns:
            recent = "\n".join(
                f"User: {t.query}\nAssistant: {t.response}" for t in self.recent_turns
            )
            parts.append(f"Recent conversation:\n{recent}")
        return "\n\n".join(parts) if parts else "No previous conversation."

    def get_retrieval_query(self, current_query: str) -> str:
        """
        Only recent turns feed retrieval — not the summary — so the
        embedding stays focused on what's actually relevant right now.
        """
        recent = "\n".join(
            f"User: {t.query}\nAssistant: {t.response}" for t in self.recent_turns
        )
        return f"{recent}\nUser: {current_query}" if recent else current_query