"""LLM-based intent parsing (optional, off by default).

When ``llm.enabled`` is true in voice.yaml and the rule-based parser produces no
match, the transcribed text is sent to Claude Haiku which returns a JSON command.

Requires: ``pip install anthropic`` and ``ANTHROPIC_API_KEY`` in the environment.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from robot.behavior.commands import Command, KINDS

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a robot command parser. The user says something to a humanoid robot.
Return ONLY a JSON object with "kind" and "args" fields.

Valid kinds: {kinds}

Kind args:
- walk: {{"vx": float m/s (positive=forward, negative=back), "vy": 0.0, "yaw": 0.0}}
- turn: {{"direction": "L" or "R"}}
- side_step: {{"direction": "L" or "R"}}
- gesture: {{"name": "wave"|"wave_left"|"wave_right"|"bow"|"raise_arms"|"squat"}}
- say: {{"text": "response text"}}
- stand, sit, stop, follow, come_here, estop, reset: {{}}

If the utterance is not a robot command, return {{"kind": "say", "args": {{"text": "a friendly short reply"}}}}.
Return ONLY the JSON, no markdown, no explanation.
"""


class LlmIntentParser:
    """Calls Claude Haiku to parse free-form text into a Command."""

    def __init__(self, cfg_llm: Dict[str, Any]) -> None:
        self.model = str(cfg_llm.get("model", "claude-3-haiku-20240307"))
        self.max_tokens = int(cfg_llm.get("max_tokens", 100))
        self.temperature = float(cfg_llm.get("temperature", 0.0))
        self.timeout = float(cfg_llm.get("timeout_s", 5.0))
        self._client = None  # type: Any
        self._system = SYSTEM_PROMPT.format(kinds=", ".join(KINDS))

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise RuntimeError("pip install anthropic  (needed for llm intents)")
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("set ANTHROPIC_API_KEY env var for LLM intents")
            self._client = anthropic.Anthropic(api_key=key, timeout=self.timeout)
        return self._client

    def parse(self, text: str, source: str = "voice") -> Optional[Command]:
        """Send *text* to Claude and return a Command, or None on error."""
        try:
            client = self._get_client()
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self._system,
                messages=[{"role": "user", "content": text}],
            )
            raw = resp.content[0].text.strip()
            log.debug("llm response: %s", raw)
            data = json.loads(raw)
            kind = str(data.get("kind", ""))
            if kind not in KINDS:
                log.warning("llm returned unknown kind '%s'", kind)
                return None
            return Command(kind, dict(data.get("args", {})), source)
        except json.JSONDecodeError as e:
            log.warning("llm: bad JSON: %s", e)
            return None
        except Exception as e:
            log.warning("llm intent failed: %s", e)
            return None
