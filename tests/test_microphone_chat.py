"""Microphone → STT → MiniCPM composition tests without audio hardware."""

import importlib

import pytest

from inmoove.audio.recorder import AudioBuffer, RecordingError
from inmoove.audio.stt import TranscriptionError


def microphone_chat_module():
    """Import the example lazily so its missing implementation fails tests."""
    return importlib.import_module("examples.microphone_chat")


class FakeRecorder:
    def __init__(self, audio=None, error=None):
        self.audio = audio
        self.error = error
        self.durations = []

    def record(self, duration_seconds):
        self.durations.append(duration_seconds)
        if self.error is not None:
            raise self.error
        return self.audio


class FakeTranscriber:
    def __init__(self, text=None, error=None):
        self.text = text
        self.error = error
        self.audio = None

    def transcribe(self, audio):
        self.audio = audio
        if self.error is not None:
            raise self.error
        return self.text


class FakeChat:
    def __init__(self, reply):
        self.reply = reply
        self.messages = []

    def reply_json(self, text):
        self.messages.append(text)
        return self.reply


def test_run_once_passes_korean_transcript_to_reply_json_and_prints_reply():
    """The complete software-only voice path preserves the transcript."""
    microphone_chat = microphone_chat_module()
    audio = AudioBuffer(b"\x00\x00")
    recorder = FakeRecorder(audio=audio)
    transcriber = FakeTranscriber(text="오늘 기분이 좋아")
    chat = FakeChat(
        {"response": "정말 잘됐네요!", "expression": "happy", "intensity": 2}
    )
    outputs = []

    result = microphone_chat.run_once(
        recorder,
        transcriber,
        chat,
        duration_seconds=3.0,
        output_fn=outputs.append,
    )

    assert result == chat.reply
    assert recorder.durations == [3.0]
    assert transcriber.audio is audio
    assert chat.messages == ["오늘 기분이 좋아"]
    assert outputs == [
        "You said: 오늘 기분이 좋아",
        "AI response: 정말 잘됐네요!",
        "expression: happy",
        "intensity: 2",
    ]


def test_run_once_rejects_empty_stt_result_before_calling_minicpm():
    """Whitespace-only fake STT output cannot reach reply_json."""
    microphone_chat = microphone_chat_module()
    outputs = []
    chat = FakeChat({"response": "must not be used"})

    result = microphone_chat.run_once(
        FakeRecorder(audio=AudioBuffer(b"\x00\x00")),
        FakeTranscriber(text="  "),
        chat,
        duration_seconds=3.0,
        output_fn=outputs.append,
    )

    assert result is None
    assert chat.messages == []
    assert outputs == ["STT error: STT result is empty"]


def test_run_once_reports_recording_error_without_calling_stt_or_minicpm():
    """Missing microphones are distinguishable from STT/model failures."""
    microphone_chat = microphone_chat_module()
    outputs = []
    transcriber = FakeTranscriber(text="must not be used")
    chat = FakeChat({"response": "must not be used"})

    result = microphone_chat.run_once(
        FakeRecorder(error=RecordingError("no input device")),
        transcriber,
        chat,
        duration_seconds=3.0,
        output_fn=outputs.append,
    )

    assert result is None
    assert transcriber.audio is None
    assert chat.messages == []
    assert outputs == ["Recording error: no input device"]


def test_run_once_reports_stt_error_without_calling_minicpm():
    """Transcription failures are explicit and stop before inference."""
    microphone_chat = microphone_chat_module()
    outputs = []
    chat = FakeChat({"response": "must not be used"})

    result = microphone_chat.run_once(
        FakeRecorder(audio=AudioBuffer(b"\x00\x00")),
        FakeTranscriber(error=TranscriptionError("model unavailable")),
        chat,
        duration_seconds=3.0,
        output_fn=outputs.append,
    )

    assert result is None
    assert chat.messages == []
    assert outputs == ["STT error: model unavailable"]


@pytest.mark.parametrize(
    ("device_argument", "expected_device"),
    [
        ("3", 3),
        ("USB Microphone", "USB Microphone"),
    ],
)
def test_main_parses_device_and_passes_it_to_recorder_factory(
    device_argument, expected_device
):
    """Numeric device IDs and named devices remain unambiguous at the CLI seam."""
    microphone_chat = microphone_chat_module()
    devices = []
    outputs = []

    class RecorderFactory:
        def __init__(self, device):
            devices.append(device)

        def record(self, _):
            return AudioBuffer(b"\x00\x00")

    result = microphone_chat.main(
        ["--model-path", "unused-model", "--device", device_argument],
        recorder_factory=RecorderFactory,
        transcriber_factory=lambda _: FakeTranscriber(text="안녕"),
        chat_factory=lambda: FakeChat(
            {"response": "안녕하세요!", "expression": "neutral", "intensity": 1}
        ),
        output_fn=outputs.append,
    )

    assert result == 0
    assert devices == [expected_device]
