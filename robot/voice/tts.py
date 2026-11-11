"""Text-to-speech: the robot talks back.

Engines:
  ``espeak``  — espeak-ng (``sudo apt install espeak-ng`` on Pi, ``brew install espeak`` on Mac)
  ``piper``   — neural TTS via piper-tts (``pip install piper-tts``)
  ``none``    — silent (just logs)

TTS runs in a background thread so it never blocks the control loop.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class TTS:
    """Base TTS: logs the text but produces no audio."""

    def say(self, text: str) -> None:
        log.info("tts: %s", text)

    def stop(self) -> None:
        pass


class EspeakTTS(TTS):
    """espeak-ng subprocess."""

    def __init__(self, voice: str = "en", speed: int = 150, pitch: int = 50) -> None:
        self.voice = voice
        self.speed = speed
        self.pitch = pitch
        self._lock = threading.Lock()
        self._proc = None  # type: Optional[subprocess.Popen]

    def say(self, text: str) -> None:
        if not text:
            return

        def _speak() -> None:
            try:
                with self._lock:
                    self._proc = subprocess.Popen(
                        ["espeak-ng", "-v", self.voice, "-s", str(self.speed),
                         "-p", str(self.pitch), text],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                if self._proc is not None:
                    self._proc.wait(timeout=10)
            except FileNotFoundError:
                log.warning("espeak-ng not found (apt install espeak-ng / brew install espeak)")
            except Exception as e:
                log.debug("espeak: %s", e)
            finally:
                with self._lock:
                    self._proc = None

        threading.Thread(target=_speak, name="tts", daemon=True).start()

    def stop(self) -> None:
        with self._lock:
            if self._proc is not None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass


class PiperTTS(TTS):
    """Neural TTS via piper-tts."""

    def __init__(self, model: str = "en_US-lessac-medium") -> None:
        self.model = model
        self._piper = None  # type: Any

    def _get_piper(self) -> Any:
        if self._piper is None:
            try:
                from piper import PiperVoice  # type: ignore[import-untyped]
                self._piper = PiperVoice.load(self.model)
            except ImportError:
                raise RuntimeError("pip install piper-tts")
        return self._piper

    def say(self, text: str) -> None:
        if not text:
            return

        def _speak() -> None:
            try:
                import io
                import wave

                voice = self._get_piper()
                buf = io.BytesIO()
                with wave.open(buf, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(voice.config.sample_rate)
                    voice.synthesize(text, w)
                # Play with aplay / afplay / sounddevice
                self._play(buf.getvalue(), voice.config.sample_rate)
            except Exception as e:
                log.warning("piper tts failed: %s", e)

        threading.Thread(target=_speak, name="tts", daemon=True).start()

    @staticmethod
    def _play(wav_bytes: bytes, sample_rate: int) -> None:
        """Try to play WAV bytes using available tools."""
        import tempfile
        import os
        import sys

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp = f.name
        try:
            if sys.platform == "darwin":
                subprocess.run(["afplay", tmp], check=True, timeout=10)
            else:
                subprocess.run(["aplay", "-q", tmp], check=True, timeout=10)
        except FileNotFoundError:
            log.debug("no audio player found")
        except Exception as e:
            log.debug("play: %s", e)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def make_tts(cfg_voice: Dict[str, Any]) -> TTS:
    """Factory: returns the TTS engine selected in voice.yaml."""
    tts_cfg = cfg_voice.get("tts", {}) or {}
    engine = str(tts_cfg.get("engine", "espeak")).lower()
    if engine == "none":
        return TTS()
    elif engine == "espeak":
        e = tts_cfg.get("espeak", {}) or {}
        return EspeakTTS(
            voice=str(e.get("voice", "en")),
            speed=int(e.get("speed", 150)),
            pitch=int(e.get("pitch", 50)),
        )
    elif engine == "piper":
        p = tts_cfg.get("piper", {}) or {}
        return PiperTTS(model=str(p.get("model", "en_US-lessac-medium")))
    else:
        log.warning("unknown TTS engine '%s', using silent", engine)
        return TTS()
