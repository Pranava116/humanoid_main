"""Speech-to-text backends: text (stdin), vosk (offline), whisper (faster-whisper).

All backends are generators: ``for text in stt: ...`` yields transcribed strings.
``TextSTT`` reads lines from stdin (for testing without a microphone).
``VoskSTT`` uses the Vosk library with optional closed grammar.
``WhisperSTT`` uses faster-whisper for better accuracy at higher cost.

Each generator yields stripped, non-empty lines / utterances and blocks while
listening. The caller (``client.py``) runs it in a thread.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Generator, Optional

log = logging.getLogger(__name__)


class TextSTT:
    """Read commands from stdin (text mode). No microphone needed."""

    def __init__(self, prompt: str = "voice> ") -> None:
        self.prompt = prompt

    def listen(self) -> Generator[str, None, None]:
        """Yield lines typed by the user. Stops on EOF or Ctrl-C."""
        try:
            while True:
                try:
                    line = input(self.prompt)
                except EOFError:
                    return
                line = line.strip()
                if line:
                    yield line
        except KeyboardInterrupt:
            return


class VoskSTT:
    """Offline speech recognition using Vosk.

    Requires: ``pip install vosk sounddevice``
    Model: downloads automatically or uses a path from config.
    """

    SMALL_MODEL = "vosk-model-small-en-us-0.15"

    def __init__(self, cfg_vosk: Dict[str, Any]) -> None:
        self.model_name = str(cfg_vosk.get("model", "small"))
        self.sample_rate = int(cfg_vosk.get("sample_rate", 16000))
        self.grammar = cfg_vosk.get("grammar")  # type: Optional[list]

    def _get_model_path(self) -> str:
        """Resolve model name to a path, downloading the small model if needed."""
        import os
        if os.path.isabs(self.model_name) or os.path.isdir(self.model_name):
            return self.model_name
        # Vosk looks for models in the current dir or ~/.cache/vosk
        from vosk import Model  # type: ignore[import-untyped]
        cache = os.path.join(os.path.expanduser("~"), ".cache", "vosk")
        path = os.path.join(cache, self.SMALL_MODEL)
        if os.path.isdir(path):
            return path
        # Let vosk download it
        log.info("downloading Vosk model '%s' (first run only)...", self.SMALL_MODEL)
        return self.SMALL_MODEL

    def listen(self) -> Generator[str, None, None]:
        import json as _json
        import queue

        import sounddevice as sd  # type: ignore[import-untyped]
        from vosk import KaldiRecognizer, Model  # type: ignore[import-untyped]

        model = Model(self._get_model_path())
        if self.grammar:
            rec = KaldiRecognizer(model, self.sample_rate, _json.dumps(self.grammar))
        else:
            rec = KaldiRecognizer(model, self.sample_rate)

        audio_q = queue.Queue()  # type: queue.Queue[bytes]

        def callback(indata, frames, time_info, status):
            if status:
                log.debug("audio status: %s", status)
            audio_q.put(bytes(indata))

        log.info("listening (vosk, %d Hz)...", self.sample_rate)
        with sd.RawInputStream(samplerate=self.sample_rate, blocksize=4000,
                                dtype="int16", channels=1, callback=callback):
            while True:
                data = audio_q.get()
                if rec.AcceptWaveform(data):
                    result = _json.loads(rec.Result())
                    text = result.get("text", "").strip()
                    if text:
                        log.debug("vosk: '%s'", text)
                        yield text


class WhisperSTT:
    """Speech recognition using faster-whisper (CTranslate2).

    Requires: ``pip install faster-whisper sounddevice``
    """

    def __init__(self, cfg_whisper: Dict[str, Any]) -> None:
        self.model_name = str(cfg_whisper.get("model", "tiny.en"))
        self.device = str(cfg_whisper.get("device", "cpu"))
        self.compute_type = str(cfg_whisper.get("compute_type", "int8"))
        self.beam_size = int(cfg_whisper.get("beam_size", 1))

    def listen(self) -> Generator[str, None, None]:
        import io
        import queue
        import wave

        import numpy as np
        import sounddevice as sd  # type: ignore[import-untyped]
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        model = WhisperModel(self.model_name, device=self.device,
                             compute_type=self.compute_type)
        sample_rate = 16000
        chunk_s = 3.0  # process audio in chunks
        chunk_samples = int(sample_rate * chunk_s)
        silence_threshold = 500  # RMS below this = silence
        audio_q = queue.Queue()  # type: queue.Queue[bytes]

        def callback(indata, frames, time_info, status):
            audio_q.put(bytes(indata))

        log.info("listening (whisper %s)...", self.model_name)
        with sd.RawInputStream(samplerate=sample_rate, blocksize=4000,
                                dtype="int16", channels=1, callback=callback):
            buffer = b""
            silence_chunks = 0
            while True:
                data = audio_q.get()
                # Check if there is speech (simple energy check)
                arr = np.frombuffer(data, dtype=np.int16)
                rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
                if rms > silence_threshold:
                    buffer += data
                    silence_chunks = 0
                elif buffer:
                    silence_chunks += 1
                    buffer += data
                    if silence_chunks >= 3:
                        # End of utterance; transcribe
                        audio_array = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
                        segments, _ = model.transcribe(audio_array, beam_size=self.beam_size)
                        text = " ".join(s.text for s in segments).strip()
                        buffer = b""
                        silence_chunks = 0
                        if text:
                            log.debug("whisper: '%s'", text)
                            yield text


def make_stt(cfg_voice: Dict[str, Any]) -> Any:
    """Factory: returns the STT backend selected in voice.yaml."""
    stt_cfg = cfg_voice.get("stt", {}) or {}
    engine = str(stt_cfg.get("engine", "text")).lower()
    if engine == "text":
        return TextSTT()
    elif engine == "vosk":
        return VoskSTT(stt_cfg.get("vosk", {}) or {})
    elif engine == "whisper":
        return WhisperSTT(stt_cfg.get("whisper", {}) or {})
    else:
        log.warning("unknown STT engine '%s', falling back to text", engine)
        return TextSTT()
