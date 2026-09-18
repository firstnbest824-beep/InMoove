"""Replaceable local speech-to-text boundary with a Vosk baseline."""

from collections.abc import Callable
import json
from pathlib import Path
from typing import Protocol

from .recorder import AudioBuffer


class TranscriptionError(RuntimeError):
    """Raised when an STT engine cannot provide a usable transcript."""


class EmptyTranscriptionError(TranscriptionError):
    """Raised when an STT engine reports no spoken text."""


class Transcriber(Protocol):
    """An STT engine that can be replaced without changing the application flow."""

    def transcribe(self, audio: AudioBuffer) -> str:
        """Return one non-empty transcript for a validated audio buffer."""


ModelFactory = Callable[[str], object]
RecognizerFactory = Callable[[object, int], object]


class VoskTranscriber:
    """One-shot local Vosk transcription using a pre-downloaded model directory."""

    def __init__(
        self,
        model_path: str | Path,
        model_factory: ModelFactory | None = None,
        recognizer_factory: RecognizerFactory | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self._model_factory = model_factory
        self._recognizer_factory = recognizer_factory
        self._model: object | None = None

    def transcribe(self, audio: AudioBuffer) -> str:
        """Transcribe one validated buffer, rejecting malformed or empty results."""
        if not isinstance(audio, AudioBuffer):
            raise TypeError("audio must be an AudioBuffer")

        try:
            recognizer = self._get_recognizer(audio.sample_rate)
            recognizer.AcceptWaveform(audio.samples)
            raw_result = recognizer.FinalResult()
        except TranscriptionError:
            raise
        except Exception as error:
            raise TranscriptionError(f"Vosk transcription failed: {error}") from error

        try:
            result = json.loads(raw_result)
        except (TypeError, json.JSONDecodeError) as error:
            raise TranscriptionError("Vosk result must be valid JSON") from error

        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise TranscriptionError("Vosk result must contain a text string")

        text = result["text"].strip()
        if not text:
            raise EmptyTranscriptionError("STT result is empty")

        return text

    def _get_recognizer(self, sample_rate: int) -> object:
        recognizer_factory = self._recognizer_factory or self._default_recognizer_factory
        return recognizer_factory(self._get_model(), sample_rate)

    def _get_model(self) -> object:
        if self._model is None:
            factory = self._model_factory or self._default_model_factory
            try:
                self._model = factory(str(self.model_path))
            except TranscriptionError:
                raise
            except Exception as error:
                raise TranscriptionError(
                    f"could not load Vosk model from {self.model_path}: {error}"
                ) from error
        return self._model

    @staticmethod
    def _default_model_factory(model_path: str) -> object:
        path = Path(model_path)
        if not path.is_dir():
            raise TranscriptionError(f"Vosk model directory does not exist: {path}")

        try:
            from vosk import Model
        except ModuleNotFoundError as error:
            raise TranscriptionError(
                "vosk is required for local STT; install requirements-audio.txt"
            ) from error

        return Model(model_path)

    @staticmethod
    def _default_recognizer_factory(model: object, sample_rate: int) -> object:
        try:
            from vosk import KaldiRecognizer
        except ModuleNotFoundError as error:
            raise TranscriptionError(
                "vosk is required for local STT; install requirements-audio.txt"
            ) from error

        return KaldiRecognizer(model, sample_rate)
