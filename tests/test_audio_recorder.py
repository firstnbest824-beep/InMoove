"""Audio recording boundary tests that require no microphone hardware."""

import importlib

import pytest


def recorder_module():
    """Load the new boundary at test time so missing code is a test failure."""
    return importlib.import_module("inmoove.audio.recorder")


def stt_module():
    """Load the optional STT implementation only for its focused tests."""
    return importlib.import_module("inmoove.audio.stt")


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"sample_rate": 8000}, "16000"),
        ({"channels": 2}, "mono"),
        ({"sample_width_bytes": 1}, "16-bit"),
        ({"samples": b"\x00"}, "aligned"),
    ],
)
def test_audio_buffer_rejects_format_that_vosk_cannot_consume(overrides, error):
    """Invalid PCM cannot cross the recorder-to-STT boundary."""
    recorder = recorder_module()
    values = {
        "samples": b"\x00\x00",
        "sample_rate": 16000,
        "channels": 1,
        "sample_width_bytes": 2,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=error):
        recorder.AudioBuffer(**values)


def test_sounddevice_recorder_converts_backend_failure_to_recording_error():
    """A missing device/backend is exposed to the app as RecordingError."""
    recorder = recorder_module()

    class FailingSoundDevice:
        @staticmethod
        def rec(**_):
            raise RuntimeError("no input device")

    microphone = recorder.SoundDeviceRecorder(
        device="USB Microphone",
        sounddevice_module=FailingSoundDevice,
    )

    with pytest.raises(recorder.RecordingError, match="record microphone"):
        microphone.record(1.0)


class FakeRecognizer:
    """Minimal Vosk-shaped recognizer for local-only tests."""

    def __init__(self, final_result):
        self.final_result = final_result
        self.received_samples = None

    def AcceptWaveform(self, samples):
        self.received_samples = samples
        return True

    def FinalResult(self):
        return self.final_result


def test_vosk_transcriber_rejects_empty_text_from_recognizer():
    """An empty Vosk result must not be forwarded to MiniCPM."""
    recorder = recorder_module()
    stt = stt_module()
    transcriber = stt.VoskTranscriber(
        "unused-model",
        model_factory=lambda _: object(),
        recognizer_factory=lambda *_: FakeRecognizer('{"text": "  "}'),
    )

    with pytest.raises(stt.EmptyTranscriptionError, match="empty"):
        transcriber.transcribe(recorder.AudioBuffer(b"\x00\x00"))


def test_vosk_transcriber_wraps_malformed_recognizer_json_as_stt_error():
    """A malformed local-engine response becomes an actionable STT error."""
    recorder = recorder_module()
    stt = stt_module()
    transcriber = stt.VoskTranscriber(
        "unused-model",
        model_factory=lambda _: object(),
        recognizer_factory=lambda *_: FakeRecognizer("not json"),
    )

    with pytest.raises(stt.TranscriptionError, match="valid JSON"):
        transcriber.transcribe(recorder.AudioBuffer(b"\x00\x00"))
