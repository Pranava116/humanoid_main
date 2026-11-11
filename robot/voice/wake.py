"""Wake-word detection: optionally require a trigger phrase before processing commands.

When ``wake.enabled`` is true, the STT stream is filtered: utterances that start
with the wake phrase are stripped and passed through; others are dropped.
When disabled, every utterance is passed through.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

log = logging.getLogger(__name__)


class WakeDetector:
    """Simple prefix-based wake-word filter."""

    def __init__(self, enabled: bool = False, phrase: str = "hey robot") -> None:
        self.enabled = enabled
        self.phrase = phrase.lower().strip()
        self._awake = not enabled  # if disabled, always awake

    def check(self, text: str) -> str:
        """Return the command portion if the wake word was detected (or if disabled).

        Returns empty string if the wake word was not found and is required.
        """
        if not self.enabled:
            return text

        lower = text.lower().strip()
        if lower.startswith(self.phrase):
            # Strip the wake phrase and return the rest
            remainder = text[len(self.phrase):].strip()
            if remainder:
                log.debug("wake: detected, command='%s'", remainder)
                return remainder
            else:
                log.debug("wake: detected but no command after it")
                return ""
        log.debug("wake: no wake word in '%s'", text[:50])
        return ""

    @classmethod
    def from_config(cls, cfg_wake: Dict[str, Any]) -> "WakeDetector":
        return cls(
            enabled=bool(cfg_wake.get("enabled", False)),
            phrase=str(cfg_wake.get("phrase", "hey robot")),
        )
