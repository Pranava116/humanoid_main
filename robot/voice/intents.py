"""Intent parser: text sentence -> Command.

Rule-based by default. Maps spoken phrases to ``Command`` objects that the FSM
understands. Falls back to ``llm_intents.py`` when ``llm.enabled`` is true and the
rules produce no match.

The grammar is intentionally loose (substring matching + aliases) so minor STT
errors or paraphrases still work, e.g. "could you walk forward" and "go forward"
both match ``walk`` with ``vx > 0``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from robot.behavior.commands import Command, KINDS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- aliases
# Each entry is (pattern_regex, command_kind, args_dict).
# Patterns are tried in order; first match wins. Every pattern is wrapped in word
# boundaries, so "please stop" is a stop (not "e stop") and "understand" is not "stand".

_RULES: List[Tuple[str, str, Dict[str, Any]]] = [
    # emergency
    (r"(?:emergency\s*stop|e[\s-]*stop)", "estop", {}),
    (r"reset", "reset", {}),

    # walking
    (r"(?:walk|go|move|step)\s+(?:forward|ahead|straight)", "walk", {"vx": 0.02}),
    (r"(?:walk|go|move|step)\s+(?:backward|back|backwards)", "walk", {"vx": -0.02}),
    (r"turn\s+(?:left|l\b)", "turn", {"direction": "L"}),
    (r"turn\s+(?:right|r\b)", "turn", {"direction": "R"}),
    (r"side\s*step\s+(?:left|l\b)", "side_step", {"direction": "L"}),
    (r"side\s*step\s+(?:right|r\b)", "side_step", {"direction": "R"}),
    (r"(?:strafe|shuffle)\s+(?:left|l\b)", "side_step", {"direction": "L"}),
    (r"(?:strafe|shuffle)\s+(?:right|r\b)", "side_step", {"direction": "R"}),

    # posture
    (r"stop", "stop", {}),
    (r"(?:stand\s*up|stand|get\s*up)", "stand", {}),
    (r"(?:sit\s*down|sit)", "sit", {}),

    # gestures
    (r"wave\s+(?:left|l\b)", "gesture", {"name": "wave_left"}),
    (r"wave\s+(?:right|r\b)", "gesture", {"name": "wave_right"}),
    (r"wave", "gesture", {"name": "wave"}),
    (r"bow", "gesture", {"name": "bow"}),
    (r"(?:raise|lift)\s+(?:arms|hands)", "gesture", {"name": "raise_arms"}),
    (r"squat", "gesture", {"name": "squat"}),
    (r"dance", "gesture", {"name": "wave"}),  # fun alias

    # follow / come
    (r"(?:come\s*here|come\s*to\s*me)", "come_here", {}),
    (r"follow\s*(?:me)?", "follow", {}),

    # conversation (mapped to 'say' so the robot speaks back)
    (r"(?:hello|hi|hey)", "say", {"text": "Hello! I am ready for your commands."}),
    (r"how\s+are\s+you", "say", {"text": "I am doing great, thank you for asking!"}),
    (r"what\s+is\s+your\s+name", "say", {"text": "I am a humanoid robot. Nice to meet you!"}),
]

# Compile once
_COMPILED = [(re.compile(rf"\b(?:{pat})\b", re.IGNORECASE), kind, args) for pat, kind, args in _RULES]


def parse(text: str, source: str = "voice") -> Optional[Command]:
    """Parse a text sentence into a Command, or None if no rule matches."""
    text = text.strip()
    if not text:
        return None
    for regex, kind, args in _COMPILED:
        if regex.search(text):
            log.debug("intent: '%s' -> %s %s", text, kind, args)
            return Command(kind, dict(args), source)
    log.info("intent: no match for '%s'", text)
    return None


def all_patterns() -> List[str]:
    """Human-readable list of supported phrases (for help text)."""
    examples = [
        "walk forward / backward",
        "turn left / right",
        "side step left / right",
        "stop",
        "stand up / sit down",
        "wave / wave left / wave right",
        "bow",
        "raise arms",
        "squat",
        "come here / follow me",
        "emergency stop / e stop",
        "reset",
        "hello / hi",
    ]
    return examples
