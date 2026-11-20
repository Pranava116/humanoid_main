"""LLM-based intent parsing (optional, off by default).

When ``llm.enabled`` is true in voice.yaml (or ``--llm``) and the rule-based parser
produces no match, the transcribed text is sent to Claude, which answers with a
JSON command (structured output, so the reply always matches the schema).

The LLM can never send ``reset``: standing up after a fall or e-stop needs the
explicit spoken / typed command, which the rule parser handles.

Requires: ``pip install anthropic`` (a recent release: structured outputs and
``effort``) and ``ANTHROPIC_API_KEY`` in the environment. Any error falls back to
"not understood", never to a guess.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from robot.behavior.commands import KINDS, Command

log = logging.getLogger(__name__)

LLM_KINDS = [k for k in KINDS if k != "reset"]
GESTURES = ["wave", "wave_left", "wave_right", "bow", "raise_arms", "squat"]

SYSTEM_PROMPT = """\
You turn what a person says to a small humanoid robot into one robot command.

kind: one of {kinds}
- walk: walk forward (vx > 0) or backward (vx < 0); vx in m/s, normal speed 0.06, slow 0.03, fast 0.1
- turn / side_step: direction "L" or "R"
- gesture: gesture is one of {gestures}
- say: a short friendly spoken reply in text (use this when the sentence is not a robot command)
- estop: the person sounds alarmed or asks the robot to freeze immediately
Fields that do not apply to the chosen kind: vx 0, direction "none", gesture "none", text "".
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": LLM_KINDS},
        "vx": {"type": "number"},
        "direction": {"type": "string", "enum": ["L", "R", "none"]},
        "gesture": {"type": "string", "enum": GESTURES + ["none"]},
        "text": {"type": "string"},
    },
    "required": ["kind", "vx", "direction", "gesture", "text"],
    "additionalProperties": False,
}


def to_command(data: Dict[str, Any], max_speed: float, source: str = "voice") -> Optional[Command]:
    """Schema fields -> Command with only the args that kind uses (None if invalid)."""
    kind = str(data.get("kind", ""))
    if kind not in LLM_KINDS:
        return None
    args: Dict[str, Any] = {}
    if kind == "walk":
        vx = float(data.get("vx", 0.0))
        if vx == 0.0:
            return None
        args = {"vx": max(-max_speed, min(max_speed, vx))}
    elif kind in ("turn", "side_step"):
        d = data.get("direction")
        if d not in ("L", "R"):
            return None
        args = {"direction": d}
    elif kind == "gesture":
        g = data.get("gesture")
        if g not in GESTURES:
            return None
        args = {"name": g}
    elif kind == "say":
        text = str(data.get("text", "")).strip()
        if not text:
            return None
        args = {"text": text[:200]}
    return Command(kind, args, source)


class LlmIntentParser:
    """Calls Claude to parse free-form text into a Command."""

    def __init__(self, cfg_llm: Dict[str, Any]) -> None:
        self.model = str(cfg_llm.get("model", "claude-haiku-5-5"))
        self.max_tokens = int(cfg_llm.get("max_tokens", 1024))
        self.effort = str(cfg_llm.get("effort", "low"))
        self.timeout = float(cfg_llm.get("timeout_s", 5.0))
        self.max_speed = float(cfg_llm.get("max_speed", 0.04))
        self._client = None  # type: Any
        self._system = SYSTEM_PROMPT.format(kinds=", ".join(LLM_KINDS), gestures=", ".join(GESTURES))

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise RuntimeError("pip install anthropic  (needed for llm intents)")
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("set ANTHROPIC_API_KEY env var for LLM intents")
            # one quick retry only: a spoken command is stale after a few seconds
            self._client = anthropic.Anthropic(timeout=self.timeout, max_retries=1)
        return self._client

    def parse(self, text: str, source: str = "voice") -> Optional[Command]:
        """Send *text* to Claude and return a Command, or None on any error."""
        try:
            resp = self._get_client().messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self._system,
                messages=[{"role": "user", "content": text}],
                output_config={"effort": self.effort,
                               "format": {"type": "json_schema", "schema": SCHEMA}},
            )
            if resp.stop_reason != "end_turn":
                log.warning("llm: no answer (stop_reason %s)", resp.stop_reason)
                return None
            raw = next(b.text for b in resp.content if b.type == "text")
            log.debug("llm response: %s", raw)
            cmd = to_command(json.loads(raw), self.max_speed, source)
            if cmd is None:
                log.warning("llm: unusable answer %s", raw[:200])
            return cmd
        except Exception as e:     # network, auth, rate limit, bad JSON: fall back to "not understood"
            log.warning("llm intent failed: %s", e)
            return None
