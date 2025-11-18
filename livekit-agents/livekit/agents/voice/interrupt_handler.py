from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Iterable, Literal

Decision = Literal["ignore", "interrupt", "speech"]


class InterruptionFilter:
    """Extension-layer interruption filter for LiveKit voice agents."""

    DEFAULT_IGNORED_WORDS: tuple[str, ...] = ("uh", "umm", "hmm", "haan")
    CONFIDENCE_THRESHOLD = 0.6

    _WORD_RE = re.compile(r"[\w']+")

    def __init__(self, *, ignored_words: Iterable[str] | None = None) -> None:
        env_words = self._parse_env_words(os.getenv("IGNORED_WORDS"))
        initial_words = (
            self._normalize_words(ignored_words)
            if ignored_words is not None
            else env_words
        )

        if not initial_words:
            initial_words = list(self.DEFAULT_IGNORED_WORDS)

        self.ignored_words = initial_words
        self._ignored_set = set(initial_words)
        self._agent_speaking = False

        # Create the asyncio.Lock up front to avoid lazy-race conditions.
        # Creating the lock in __init__ is fine — it will bind to the current loop when used.
        self._lock: asyncio.Lock = asyncio.Lock()

        self._logger = logging.getLogger(__name__)

    async def set_agent_speaking(self, speaking: bool) -> None:
        """Async-safe setter for whether agent TTS is currently playing."""
        async with self._lock:
            self._agent_speaking = speaking

    async def update_ignored_words(self, words: Iterable[str]) -> None:
        """Allow runtime reconfiguration of filler vocabulary in an async-safe way."""
        normalized = self._normalize_words(words)
        if not normalized:
            return

        async with self._lock:
            self.ignored_words = normalized
            self._ignored_set = frozenset(normalized)

        self._logger.info(
            "Updated interruption filter ignored words",
            extra={"ignored_words": normalized},
        )

    def filter_transcript(self, text: str, confidence: float) -> Decision:
        """
        Sync convenience wrapper. Prefer the async API `handle_transcription_event`
        when running inside the event loop or when concurrency safety is required.

        This wrapper will call the async API using asyncio.get_event_loop().run_until_complete
        only if there is no running loop. This keeps unit tests and simple uses working.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — run the async API synchronously (useful in tests).
            return asyncio.run(self.handle_transcription_event(text, confidence, None))

        # If there *is* a running loop, we do NOT block it; the caller should use async API.
        raise RuntimeError(
            "filter_transcript() cannot be called from a running event loop; "
            "use await handle_transcription_event(...) instead."
        )

    async def handle_transcription_event(
        self,
        text: str,
        confidence: float,
        speaker_id: str | None = None,
    ) -> Decision:
        """
        Canonical async, thread-safe API. Acquires the lock and makes a decision.
        """
        async with self._lock:
            decision = self._decide(text, confidence)
        self._log_decision(decision, text, confidence, speaker_id)
        return decision

    def _decide(self, text: str, confidence: float) -> Decision:
        # Fast path: if agent is not speaking, always treat as speech
        if not self._agent_speaking:
            return "speech"

        tokens = self._tokenize(text)
        if not tokens:
            return "ignore"

        # If confidence is low, treat as noise (ignore) even if tokens look real.
        if confidence < self.CONFIDENCE_THRESHOLD:
            return "ignore"

        has_real_word = any(token not in self._ignored_set for token in tokens)
        if has_real_word:
            return "interrupt"

        return "ignore"

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        if not text:
            return []
        return [match.group(0).lower() for match in InterruptionFilter._WORD_RE.finditer(text)]

    def _log_decision(
        self,
        decision: Decision,
        transcript: str,
        confidence: float,
        speaker_id: str | None,
    ) -> None:
        extra = {"transcript": transcript, "confidence": confidence}
        if speaker_id:
            extra["speaker_id"] = speaker_id

        # Log ignored noise at debug level — high-frequency events should not spam INFO.
        if decision == "ignore":
            extra["agent_speaking"] = self._agent_speaking
            self._logger.debug("Ignoring filler/noise while agent speaking", extra=extra)
        elif decision == "interrupt":
            extra["agent_speaking"] = self._agent_speaking
            self._logger.info("Interrupting agent due to user speech", extra=extra)
        else:
            self._logger.debug("User speech while agent idle", extra=extra)

    @staticmethod
    def _normalize_words(words: Iterable[str] | None) -> list[str]:
        if not words:
            return []
        normalized: list[str] = []
        for word in words:
            stripped = word.strip().lower()
            if stripped:
                normalized.append(stripped)
        return normalized

    @staticmethod
    def _parse_env_words(raw: str | None) -> list[str]:
        if not raw:
            return []
        return InterruptionFilter._normalize_words(raw.split(","))