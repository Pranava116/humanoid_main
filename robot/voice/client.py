"""Voice command client: STT -> wake -> intents -> Command -> Link -> robot.

This is both a command-line tool and a library used by ``robot.run``.

Text mode (no microphone needed — for testing):
  python -m robot.voice.client --text

Vosk mode (offline, needs microphone):
  python -m robot.voice.client --vosk

Whisper mode (better accuracy):
  python -m robot.voice.client --whisper

The client can connect to a running robot process (``--connect HOST``) or
run everything locally with ``--local`` (requires ``--backend``).

Examples:
  python -m robot.voice.client --text --local --backend sim         # robot + voice in one process
  mjpython -m robot.voice.client --text --local --backend sim --viewer --dashboard
  python -m robot.voice.client --text                               # robot.run already running here
  python -m robot.voice.client --vosk --connect 192.168.1.42        # microphone, robot on the Pi
  python -m robot.voice.client --text --llm                         # free-form sentences via Claude
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import threading
from typing import Any, Dict, List, Optional

from robot.behavior.commands import Command
from robot.cli import add_common_args, backend_from_args, realtime_from_args, setup_logging
from robot.config import Config, load_config
from robot.voice.intents import all_patterns, parse
from robot.voice.stt import make_stt
from robot.voice.tts import TTS, make_tts
from robot.voice.wake import WakeDetector

log = logging.getLogger(__name__)


class VoiceClient:
    """Wires STT -> wake -> intents -> Link.

    Call ``run()`` to block on the STT stream, or ``process(text)`` to handle
    one utterance programmatically.
    """

    def __init__(self, link: Any, cfg_voice: Dict[str, Any],
                 stt: Any = None, tts: Optional[TTS] = None,
                 llm: Any = None) -> None:
        self.link = link
        self.stt = stt or make_stt(cfg_voice)
        self.tts = tts or make_tts(cfg_voice)
        self.wake = WakeDetector.from_config(cfg_voice.get("wake", {}) or {})
        self.llm = llm
        self._stop = threading.Event()

    def process(self, text: str) -> Optional[Command]:
        """Process one utterance: wake -> intents -> send.

        Returns the Command sent, or None if no match.
        """
        # Wake word filter
        text = self.wake.check(text)
        if not text:
            return None

        # Rule-based intent parsing
        cmd = parse(text, source="voice")

        # Fall back to LLM if enabled and no rule matched
        if cmd is None and self.llm is not None:
            log.info("trying LLM for: '%s'", text)
            cmd = self.llm.parse(text, source="voice")

        if cmd is None:
            self.tts.say("Sorry, I did not understand that command.")
            return None

        # Handle TTS for 'say' commands locally (the robot also logs them)
        if cmd.kind == "say":
            self.tts.say(str(cmd.args.get("text", "")))

        # Send to the robot
        self.link.send(cmd)
        log.info("sent: %s", cmd)

        # Announce the action
        if cmd.kind != "say":
            self._announce(cmd)

        return cmd

    def _announce(self, cmd: Command) -> None:
        """Brief TTS confirmation of the command."""
        phrases = {
            "walk": "Walking.",
            "turn": "Turning.",
            "side_step": "Side stepping.",
            "stop": "Stopping.",
            "stand": "Standing up.",
            "sit": "Sitting down.",
            "gesture": "Okay.",
            "follow": "Following.",
            "come_here": "Coming to you.",
            "estop": "Emergency stop!",
            "reset": "Resetting.",
        }
        text = phrases.get(cmd.kind, "")
        if text:
            self.tts.say(text)

    def run(self) -> None:
        """Block on the STT stream until stopped or EOF."""
        log.info("voice client started (engine: %s)", type(self.stt).__name__)
        log.info("supported commands: %s", ", ".join(all_patterns()))
        try:
            for text in self.stt.listen():
                if self._stop.is_set():
                    break
                self.process(text)
        except KeyboardInterrupt:
            pass
        log.info("voice client stopped")

    def stop(self) -> None:
        self._stop.set()
        self.tts.stop()


# ---------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m robot.voice.client", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    stt_group = p.add_mutually_exclusive_group()
    stt_group.add_argument("--text", action="store_true", help="type commands (no microphone)")
    stt_group.add_argument("--vosk", action="store_true", help="Vosk offline STT (needs microphone)")
    stt_group.add_argument("--whisper", action="store_true", help="faster-whisper STT (needs microphone)")
    p.add_argument("--connect", metavar="HOST", help="connect to robot at HOST (default: from config)")
    p.add_argument("--local", action="store_true", help="run the robot in this process (uses --backend)")
    p.add_argument("--dashboard", action="store_true", help="also serve the web dashboard (with --local)")
    p.add_argument("--llm", action="store_true", help="enable LLM intent fallback (needs ANTHROPIC_API_KEY)")
    p.add_argument("--quiet", action="store_true", help="no speech output (log replies only)")
    add_common_args(p)
    return p


def voice_config(cfg: Config, args: argparse.Namespace) -> Dict[str, Any]:
    """voice.yaml section with the command-line overrides applied (cfg itself is not changed)."""
    voice_cfg = copy.deepcopy(cfg.section("voice"))
    engine = "text" if args.text else "vosk" if args.vosk else "whisper" if args.whisper else None
    if engine:
        voice_cfg.setdefault("stt", {})["engine"] = engine
    if args.quiet:
        voice_cfg.setdefault("tts", {})["engine"] = "none"
    if args.llm:
        voice_cfg.setdefault("llm", {})["enabled"] = True
    return voice_cfg


def make_client(link: Any, voice_cfg: Dict[str, Any]) -> VoiceClient:
    llm = None
    llm_cfg = voice_cfg.get("llm", {}) or {}
    if llm_cfg.get("enabled"):
        from robot.voice.llm_intents import LlmIntentParser
        llm = LlmIntentParser(llm_cfg)
        log.info("LLM intent fallback enabled (%s)", llm.model)
    return VoiceClient(link, voice_cfg, llm=llm)


def print_banner() -> None:
    print("\n=== Humanoid Voice Commands ===")
    print("Supported commands:")
    for p in all_patterns():
        print(f"  • {p}")
    print("(Ctrl-D or Ctrl-C to quit)\n")


def run_local(cfg: Config, args: argparse.Namespace, voice_cfg: Dict[str, Any]) -> int:
    """Robot loop on the main thread (MuJoCo viewer on macOS needs it), voice on a worker thread."""
    from robot.behavior.runtime import RobotRuntime
    from robot.interface.bus import LocalLink
    from robot.interface.logtail import LogTail
    from robot.run import start_dashboard

    cfg, backend = backend_from_args(args, cfg)
    runtime = RobotRuntime(cfg, backend, realtime=realtime_from_args(args))
    tail = LogTail(int(cfg.get("network.log_lines", 200))).install()
    dash = start_dashboard(cfg, runtime, tail, None) if args.dashboard else None
    client = make_client(LocalLink(runtime, tail), voice_cfg)

    def voice() -> None:
        try:
            client.run()
        finally:
            runtime.stop()      # end of input (Ctrl-D) stops the robot loop too

    threading.Thread(target=voice, name="voice", daemon=True).start()
    try:
        runtime.run()
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()
        if dash is not None:
            dash.stop()
        backend.close()
        tail.remove()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    cfg = load_config(args.config_dir)
    voice_cfg = voice_config(cfg, args)
    print_banner()
    if args.local:
        return run_local(cfg, args, voice_cfg)
    from robot.interface.bus import ZmqLink
    link = ZmqLink(cfg, host=args.connect, source="voice")
    if not link.wait_connected(timeout=5.0):
        log.warning("could not reach robot at %s (is robot.run running?)", link.tel_addr)
    client = make_client(link, voice_cfg)
    try:
        client.run()
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()
        link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
