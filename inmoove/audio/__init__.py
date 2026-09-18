"""Optional local audio recording and speech-to-text boundaries."""

from .recorder import AudioBuffer, AudioRecorder, RecordingError, SoundDeviceRecorder
from .stt import (
    EmptyTranscriptionError,
    Transcriber,
    TranscriptionError,
    VoskTranscriber,
)

__all__ = [
    "AudioBuffer",
    "AudioRecorder",
    "EmptyTranscriptionError",
    "RecordingError",
    "SoundDeviceRecorder",
    "Transcriber",
    "TranscriptionError",
    "VoskTranscriber",
]
