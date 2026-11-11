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

Usage examples:
  # text mode, local sim
  python -m robot.voice.client --text --local --backend sim --fast

  # text mode, connect to robot on 127.0.0.1
  python -m robot.voice.client --text

  # vosk with microphone, connect to Pi
  python -m robot.voice.client --vosk --connect 192.168.1.42
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from typing import Any, Dict, List, Optional

from robot.behavior.commands import Command
from robot.config import Config, load_config
from robot.voice.intents import all_patterns, parse
from robot.voice.stt import TextSTT, make_stt
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
    p = argparse.ArgumentParser(
        prog="python -m robot.voice.client",
        description="Voice command client for the humanoid robot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m robot.voice.client --text                    # type commands, send to robot at 127.0.0.1
  python -m robot.voice.client --text --local --backend sim --fast   # all in one process
  python -m robot.voice.client --vosk                    # microphone, Vosk STT
  python -m robot.voice.client --vosk --connect 192.168.1.42
""",
    )
    stt_group = p.add_mutually_exclusive_group()
    stt_group.add_argument("--text", action="store_true", help="type commands (no microphone)")
    stt_group.add_argument("--vosk", action="store_true", help="Vosk offline STT (needs microphone)")
    stt_group.add_argument("--whisper", action="store_true", help="faster-whisper STT (needs microphone)")

    p.add_argument("--connect", metavar="HOST", help="connect to robot at HOST (default: from config)")
    p.add_argument("--local", action="store_true",
                   help="run robot locally in this process (needs --backend)")
    p.add_argument("--backend", choices=("sim", "dry"), default="sim",
                   help="backend when using --local (default: sim)")
    p.add_argument("--fast", action="store_true", help="sim/dry: faster than realtime (with --local)")
    p.add_argument("--viewer", action="store_true", help="sim: open MuJoCo viewer (with --local)")
    p.add_argument("--dashboard", action="store_true", help="also start the web dashboard (with --local)")
    p.add_argument("--config-dir", help="config directory (default: config/stack)")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--llm", action="store_true", help="enable LLM intent fallback (needs ANTHROPIC_API_KEY)")
    return p


def _make_local_link(cfg: Config, args: argparse.Namespace) -> "tuple[Any, Any]":
    """Build a local RobotRuntime + LocalLink for --local mode."""
    from robot.behavior.runtime import RobotRuntime
    from robot.hal import make_backend
    from robot.interface.bus import LocalLink
    from robot.interface.logtail import LogTail

    backend = make_backend(args.backend, cfg, viewer=args.viewer)
    runtime = RobotRuntime(cfg, backend, realtime=not args.fast)
    tail = LogTail(200).install()
    link = LocalLink(runtime, tail)

    # Start the robot loop in a background thread
    robot_thread = threading.Thread(target=runtime.run, name="robot", daemon=True)
    robot_thread.start()

    # Optionally start the dashboard
    if args.dashboard:
        try:
            from robot.interface.dashboard import DashboardServer, create_app
            dash_link = LocalLink(runtime, tail)
            port = int(cfg.get("network.dashboard.port", 8000))
            DashboardServer(create_app(dash_link, cfg), "0.0.0.0", port).start()
            log.info("dashboard at http://localhost:%d", port)
        except ImportError:
            log.warning("dashboard needs: pip install fastapi uvicorn")

    return link, runtime


def _make_remote_link(cfg: Config, host: Optional[str]) -> Any:
    """Connect to a running robot process over ZeroMQ."""
    from robot.interface.bus import ZmqLink
    link = ZmqLink(cfg, host=host, source="voice")
    if not link.wait_connected(timeout=5.0):
        log.warning("could not reach robot (is robot.run running?)")
    return link


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    cfg = load_config(args.config_dir)
    voice_cfg = cfg.section("voice")

    # Override STT engine from CLI flags
    if args.text:
        voice_cfg.setdefault("stt", {})["engine"] = "text"
    elif args.vosk:
        voice_cfg.setdefault("stt", {})["engine"] = "vosk"
    elif args.whisper:
        voice_cfg.setdefault("stt", {})["engine"] = "whisper"

    # Build STT
    stt = make_stt(voice_cfg)

    # Build TTS
    tts = make_tts(voice_cfg)

    # Build optional LLM parser
    llm = None
    llm_cfg = voice_cfg.get("llm", {}) or {}
    if args.llm or llm_cfg.get("enabled"):
        from robot.voice.llm_intents import LlmIntentParser
        llm = LlmIntentParser(llm_cfg)
        log.info("LLM intent fallback enabled")

    # Build link
    if args.local:
        link, runtime = _make_local_link(cfg, args)
    else:
        link = _make_remote_link(cfg, args.connect)

    # Print help banner
    print("\n=== Humanoid Voice Commands ===")
    print("Supported commands:")
    for p in all_patterns():
        print(f"  • {p}")
    print()

    client = VoiceClient(link, voice_cfg, stt=stt, tts=tts, llm=llm)
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
