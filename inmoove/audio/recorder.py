"""Validated fixed-format microphone recording without import-time device access."""

from dataclasses import dataclass
from numbers import Real
from typing import Protocol


SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2


class RecordingError(RuntimeError):
    """Raised when a microphone cannot produce a valid audio buffer."""


@dataclass(frozen=True)
class AudioBuffer:
    """Mono 16 kHz signed-16-bit PCM bytes accepted by the Vosk boundary."""

    samples: bytes
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS
    sample_width_bytes: int = SAMPLE_WIDTH_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.samples, bytes):
            raise TypeError("AudioBuffer samples must be bytes")
        if self.sample_rate != SAMPLE_RATE:
            raise ValueError(f"AudioBuffer sample rate must be {SAMPLE_RATE} Hz")
        if self.channels != CHANNELS:
            raise ValueError("AudioBuffer must be mono")
        if self.sample_width_bytes != SAMPLE_WIDTH_BYTES:
            raise ValueError("AudioBuffer must use 16-bit samples")
        if len(self.samples) % self.sample_width_bytes:
            raise ValueError("AudioBuffer samples must be byte-aligned")


class AudioRecorder(Protocol):
    """Records one fixed-duration audio buffer."""

    def record(self, duration_seconds: float) -> AudioBuffer:
        """Record and return validated mono PCM audio."""


class SoundDeviceRecorder:
    """A lazy sounddevice adapter with an optional numeric or named input device."""

    def __init__(
        self,
        device: int | str | None = None,
        sounddevice_module: object | None = None,
    ) -> None:
        self.device = device
        self._sounddevice_module = sounddevice_module

    def record(self, duration_seconds: float) -> AudioBuffer:
        """Record mono 16-bit PCM, normalizing backend failures to RecordingError."""
        if (
            isinstance(duration_seconds, bool)
            or not isinstance(duration_seconds, Real)
            or duration_seconds <= 0
        ):
            raise ValueError("recording duration must be a positive number")

        frame_count = round(float(duration_seconds) * SAMPLE_RATE)
        if frame_count <= 0:
            raise ValueError("recording duration is too short")

        try:
            sounddevice = self._sounddevice_module or self._import_sounddevice()
            recording = sounddevice.rec(
                frame_count,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                device=self.device,
            )
            sounddevice.wait()
            return AudioBuffer(recording.tobytes())
        except RecordingError:
            raise
        except Exception as error:
            raise RecordingError(f"could not record microphone input: {error}") from error

    @staticmethod
    def _import_sounddevice() -> object:
        try:
            import sounddevice
        except ModuleNotFoundError as error:
            raise RecordingError(
                "sounddevice is required for microphone recording; "
                "install requirements-audio.txt"
            ) from error
        return sounddevice
