"""Text-to-speech: the robot talks back.

Engines:
  ``espeak``  — espeak-ng (``sudo apt install espeak-ng`` on Pi, ``brew install espeak-ng`` on Mac)
  ``piper``   — neural TTS via piper-tts (``pip install piper-tts``)
  ``none``    — silent (just logs)

Speech runs on one background thread with a queue: it never blocks the caller,
and utterances play one after another instead of on top of each other.
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PIPER_DIR = os.path.join(REPO_ROOT, "models", "weights", "piper")


class TTS:
    """Base TTS: logs the text but produces no audio."""

    def say(self, text: str) -> None:
        log.info("tts: %s", text)

    def stop(self) -> None:
        pass


class QueuedTTS(TTS):
    """Speaks queued texts in order on one worker thread. Subclasses implement ``_speak``."""

    MAX_QUEUE = 5    # drop old phrases rather than lag behind the robot

    def __init__(self) -> None:
        self._q: "queue.Queue[Optional[str]]" = queue.Queue(self.MAX_QUEUE)
        self._proc: Optional[subprocess.Popen] = None
        self._worker: Optional[threading.Thread] = None

    def say(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        log.info("tts: %s", text)
        if self._worker is None:
            self._worker = threading.Thread(target=self._run, name="tts", daemon=True)
            self._worker.start()
        try:
            self._q.put_nowait(text)
        except queue.Full:
            log.debug("tts queue full, dropping '%s'", text)

    def _run(self) -> None:
        while True:
            text = self._q.get()
            if text is None:
                return
            try:
                self._speak(text)
            except Exception as e:
                log.warning("tts failed: %s", e)

    def _speak(self, text: str) -> None:
        raise NotImplementedError

    def _run_player(self, cmd: "list[str]", timeout: float = 15.0) -> None:
        """Run an audio command so ``stop()`` can interrupt it."""
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        finally:
            self._proc = None

    def stop(self) -> None:
        """Drop queued phrases and cut the current one."""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        if self._worker is not None:
            self._q.put(None)
            self._worker = None
        proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
            except OSError:
                pass


class EspeakTTS(QueuedTTS):
    """espeak-ng (or the older espeak) subprocess."""

    def __init__(self, voice: str = "en", speed: int = 150, pitch: int = 50) -> None:
        super().__init__()
        self.voice = voice
        self.speed = speed
        self.pitch = pitch
        self.binary = shutil.which("espeak-ng") or shutil.which("espeak")
        if self.binary is None:
            log.warning("espeak-ng not found (apt install espeak-ng / brew install espeak-ng); replies are logged only")

    def _speak(self, text: str) -> None:
        if self.binary is None:
            return
        self._run_player([self.binary, "-v", self.voice, "-s", str(self.speed), "-p", str(self.pitch), text])


def piper_model_path(model: str) -> str:
    """A path to an .onnx file, or a voice name looked up in models/weights/piper/."""
    if model.endswith(".onnx"):
        return model
    return os.path.join(PIPER_DIR, model + ".onnx")


class PiperTTS(QueuedTTS):
    """Neural TTS via piper-tts. Voice files (.onnx + .onnx.json) go in models/weights/piper/."""

    def __init__(self, model: str = "en_US-lessac-medium") -> None:
        super().__init__()
        self.model = piper_model_path(model)
        self._piper = None  # type: Any

    def _get_piper(self) -> Any:
        if self._piper is None:
            try:
                from piper import PiperVoice  # type: ignore[import-untyped]
            except ImportError:
                raise RuntimeError("pip install piper-tts")
            if not os.path.isfile(self.model):
                raise RuntimeError(f"piper voice not found: {self.model}")
            self._piper = PiperVoice.load(self.model)
        return self._piper

    def _speak(self, text: str) -> None:
        import tempfile
        import wave

        voice = self._get_piper()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            with wave.open(tmp, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(voice.config.sample_rate)
                voice.synthesize(text, w)
            player = ["afplay", tmp] if sys.platform == "darwin" else ["aplay", "-q", tmp]
            try:
                self._run_player(player)
            except FileNotFoundError:
                log.warning("no audio player found (%s)", player[0])
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
    if engine == "espeak":
        e = tts_cfg.get("espeak", {}) or {}
        return EspeakTTS(voice=str(e.get("voice", "en")), speed=int(e.get("speed", 150)),
                         pitch=int(e.get("pitch", 50)))
    if engine == "piper":
        p = tts_cfg.get("piper", {}) or {}
        return PiperTTS(model=str(p.get("model", "en_US-lessac-medium")))
    log.warning("unknown TTS engine '%s', using silent", engine)
    return TTS()
