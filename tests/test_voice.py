"""Tests for the voice command pipeline (intents, wake, tts, client). No microphone needed."""

from __future__ import annotations

import pytest

from robot.behavior.commands import Command
from robot.voice.intents import parse, all_patterns
from robot.voice.wake import WakeDetector
from robot.voice.tts import TTS, make_tts
from robot.voice.stt import TextSTT


# ---------------------------------------------------------------- intents
class TestIntents:
    """Rule-based intent parser."""

    @pytest.mark.parametrize("text,kind,args", [
        ("walk forward", "walk", {"vx": 0.02}),
        ("go forward", "walk", {"vx": 0.02}),
        ("move forward", "walk", {"vx": 0.02}),
        ("step forward", "walk", {"vx": 0.02}),
        ("walk ahead", "walk", {"vx": 0.02}),
        ("walk backward", "walk", {"vx": -0.02}),
        ("go back", "walk", {"vx": -0.02}),
        ("walk backwards", "walk", {"vx": -0.02}),
        ("turn left", "turn", {"direction": "L"}),
        ("turn right", "turn", {"direction": "R"}),
        ("side step left", "side_step", {"direction": "L"}),
        ("sidestep right", "side_step", {"direction": "R"}),
        ("stop", "stop", {}),
        ("stand up", "stand", {}),
        ("stand", "stand", {}),
        ("sit down", "sit", {}),
        ("sit", "sit", {}),
        ("wave", "gesture", {"name": "wave"}),
        ("wave left", "gesture", {"name": "wave_left"}),
        ("wave right", "gesture", {"name": "wave_right"}),
        ("bow", "gesture", {"name": "bow"}),
        ("raise arms", "gesture", {"name": "raise_arms"}),
        ("lift hands", "gesture", {"name": "raise_arms"}),
        ("squat", "gesture", {"name": "squat"}),
        ("come here", "come_here", {}),
        ("follow me", "follow", {}),
        ("follow", "follow", {}),
        ("emergency stop", "estop", {}),
        ("e stop", "estop", {}),
        ("reset", "reset", {}),
        ("hello", "say", {"text": "Hello! I am ready for your commands."}),
        ("hi", "say", {"text": "Hello! I am ready for your commands."}),
        ("how are you", "say", {"text": "I am doing great, thank you for asking!"}),
        ("dance", "gesture", {"name": "wave"}),
    ])
    def test_parse_known(self, text, kind, args):
        cmd = parse(text)
        assert cmd is not None, f"'{text}' should parse"
        assert cmd.kind == kind
        assert cmd.args == args
        assert cmd.source == "voice"

    def test_parse_unknown_returns_none(self):
        assert parse("fly to the moon") is None
        assert parse("") is None
        assert parse("   ") is None

    def test_parse_case_insensitive(self):
        cmd = parse("Walk Forward")
        assert cmd is not None and cmd.kind == "walk"

    def test_parse_with_extra_words(self):
        """Regex .search() means extra words around the pattern still match."""
        cmd = parse("could you walk forward please")
        assert cmd is not None and cmd.kind == "walk"

    def test_all_patterns_returns_strings(self):
        pats = all_patterns()
        assert len(pats) > 5
        assert all(isinstance(p, str) for p in pats)

    def test_source_override(self):
        cmd = parse("walk forward", source="dashboard")
        assert cmd is not None and cmd.source == "dashboard"


# ---------------------------------------------------------------- wake
class TestWake:
    def test_disabled_passes_everything(self):
        w = WakeDetector(enabled=False)
        assert w.check("walk forward") == "walk forward"
        assert w.check("random stuff") == "random stuff"

    def test_enabled_requires_wake_phrase(self):
        w = WakeDetector(enabled=True, phrase="hey robot")
        assert w.check("walk forward") == ""
        assert w.check("hey robot walk forward") == "walk forward"
        assert w.check("Hey Robot turn left") == "turn left"

    def test_wake_phrase_alone_returns_empty(self):
        w = WakeDetector(enabled=True, phrase="hey robot")
        assert w.check("hey robot") == ""

    def test_from_config(self):
        w = WakeDetector.from_config({"enabled": True, "phrase": "ok bot"})
        assert w.enabled
        assert w.phrase == "ok bot"
        assert w.check("ok bot wave") == "wave"


# ---------------------------------------------------------------- tts
class TestTTS:
    def test_silent_tts(self):
        t = TTS()
        t.say("hello")  # should not raise
        t.stop()

    def test_make_tts_none(self):
        t = make_tts({"tts": {"engine": "none"}})
        assert isinstance(t, TTS)

    def test_make_tts_unknown_falls_back(self):
        t = make_tts({"tts": {"engine": "unknown_engine"}})
        assert isinstance(t, TTS)


# ---------------------------------------------------------------- stt
class TestTextSTT:
    def test_text_stt_init(self):
        stt = TextSTT(prompt="test> ")
        assert stt.prompt == "test> "


# ---------------------------------------------------------------- client integration
class TestVoiceClient:
    """Integration test: VoiceClient with a mock link."""

    class MockLink:
        def __init__(self):
            self.sent = []  # type: list

        def send(self, cmd):
            self.sent.append(cmd)

        def status(self):
            return {}

        def close(self):
            pass

        @property
        def connected(self):
            return True

    def test_process_sends_command(self):
        from robot.voice.client import VoiceClient
        link = self.MockLink()
        vc = VoiceClient(link, {}, tts=TTS())
        cmd = vc.process("walk forward")
        assert cmd is not None
        assert cmd.kind == "walk"
        assert len(link.sent) == 1
        assert link.sent[0].kind == "walk"

    def test_process_unknown_returns_none(self):
        from robot.voice.client import VoiceClient
        link = self.MockLink()
        vc = VoiceClient(link, {}, tts=TTS())
        cmd = vc.process("fly to mars")
        assert cmd is None
        assert len(link.sent) == 0

    def test_process_with_wake_word(self):
        from robot.voice.client import VoiceClient
        link = self.MockLink()
        voice_cfg = {"wake": {"enabled": True, "phrase": "hey robot"}}
        vc = VoiceClient(link, voice_cfg, tts=TTS())
        assert vc.process("walk forward") is None  # no wake word
        assert len(link.sent) == 0
        cmd = vc.process("hey robot walk forward")
        assert cmd is not None and cmd.kind == "walk"
        assert len(link.sent) == 1

    def test_process_say_command(self):
        from robot.voice.client import VoiceClient
        link = self.MockLink()
        vc = VoiceClient(link, {}, tts=TTS())
        cmd = vc.process("hello")
        assert cmd is not None
        assert cmd.kind == "say"
        assert "Hello" in cmd.args["text"]

    def test_all_motion_commands(self):
        """Every motion command parses and sends correctly."""
        from robot.voice.client import VoiceClient
        link = self.MockLink()
        vc = VoiceClient(link, {}, tts=TTS())
        commands = [
            ("walk forward", "walk"),
            ("walk backward", "walk"),
            ("turn left", "turn"),
            ("turn right", "turn"),
            ("side step left", "side_step"),
            ("side step right", "side_step"),
            ("stop", "stop"),
            ("stand up", "stand"),
            ("sit down", "sit"),
            ("wave", "gesture"),
            ("bow", "gesture"),
            ("emergency stop", "estop"),
            ("reset", "reset"),
        ]
        for text, expected_kind in commands:
            cmd = vc.process(text)
            assert cmd is not None, f"'{text}' should produce a command"
            assert cmd.kind == expected_kind, f"'{text}' should be kind '{expected_kind}', got '{cmd.kind}'"


# ---------------------------------------------------------------- FSM integration
class TestVoiceFSM:
    """Voice commands processed by the real FSM (dry backend)."""

    def test_voice_walk_through_fsm(self, cfg, dry):
        from robot.behavior.runtime import RobotRuntime
        from robot.voice.intents import parse
        rt = RobotRuntime(cfg, dry, realtime=False)
        # Wait for IDLE
        for _ in range(int(10 * rt.app.rate_hz)):
            rt.tick()
            if rt.fsm.state == "IDLE":
                break
        assert rt.fsm.state == "IDLE"
        # Send a voice command
        cmd = parse("walk forward", source="voice")
        assert cmd is not None
        rt.send(cmd)
        rt.tick()
        assert rt.fsm.state == "WALKING"
        # Stop
        cmd = parse("stop", source="voice")
        assert cmd is not None
        rt.send(cmd)
        for _ in range(int(30 * rt.app.rate_hz)):
            rt.tick()
            if rt.fsm.state == "IDLE":
                break
        assert rt.fsm.state == "IDLE"

    def test_voice_gesture(self, cfg, dry):
        from robot.behavior.runtime import RobotRuntime
        from robot.voice.intents import parse
        rt = RobotRuntime(cfg, dry, realtime=False)
        for _ in range(int(10 * rt.app.rate_hz)):
            rt.tick()
            if rt.fsm.state == "IDLE":
                break
        cmd = parse("wave")
        assert cmd is not None
        rt.send(cmd)
        rt.tick()
        assert rt.fsm.state == "GESTURE"

    def test_voice_estop(self, cfg, dry):
        from robot.behavior.runtime import RobotRuntime
        from robot.voice.intents import parse
        rt = RobotRuntime(cfg, dry, realtime=False)
        for _ in range(int(10 * rt.app.rate_hz)):
            rt.tick()
            if rt.fsm.state == "IDLE":
                break
        cmd = parse("emergency stop")
        assert cmd is not None
        rt.send(cmd)
        rt.tick()
        assert rt.fsm.state == "ESTOP"
