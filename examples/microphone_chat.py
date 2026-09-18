"""Record one short microphone clip, transcribe it locally, then call MiniCPM."""

import argparse
from pathlib import Path
import sys
from typing import Callable, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inmoove.audio import (
    AudioRecorder,
    EmptyTranscriptionError,
    RecordingError,
    SoundDeviceRecorder,
    Transcriber,
    TranscriptionError,
    VoskTranscriber,
)
from inmoove.inference import MiniCPMTextChat


class JsonChat(Protocol):
    """The existing MiniCPM structured-response boundary used by this example."""

    def reply_json(self, text: str) -> dict:
        """Return response, expression, and intensity for one text input."""


def parse_device(value: str | None) -> int | str | None:
    """Parse decimal device indices while preserving device-name strings."""
    if value is None:
        return None
    return int(value) if value.isdecimal() else value


def run_once(
    recorder: AudioRecorder,
    transcriber: Transcriber,
    chat: JsonChat,
    duration_seconds: float,
    output_fn: Callable[[str], None] = print,
) -> dict | None:
    """Run the fixed recording → STT → existing MiniCPM JSON flow once."""
    try:
        audio = recorder.record(duration_seconds)
    except RecordingError as error:
        output_fn(f"Recording error: {error}")
        return None

    try:
        text = transcriber.transcribe(audio)
        if not text.strip():
            raise EmptyTranscriptionError("STT result is empty")
    except TranscriptionError as error:
        output_fn(f"STT error: {error}")
        return None

    output_fn(f"You said: {text}")
    reply = chat.reply_json(text)
    output_fn(f"AI response: {reply['response']}")
    output_fn(f"expression: {reply['expression']}")
    output_fn(f"intensity: {reply['intensity']}")
    return reply


def main(
    argv: list[str] | None = None,
    recorder_factory: Callable[[int | str | None], AudioRecorder] = SoundDeviceRecorder,
    transcriber_factory: Callable[[str], Transcriber] = VoskTranscriber,
    chat_factory: Callable[[], JsonChat] = MiniCPMTextChat,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Parse one microphone request and run it without any serial hardware."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        required=True,
        help="path to a pre-downloaded local Vosk model directory",
    )
    parser.add_argument(
        "--device",
        help="sounddevice input index (for example 3) or input device name",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=4.0,
        help="fixed recording duration in seconds (default: 4.0)",
    )
    args = parser.parse_args(argv)

    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    run_once(
        recorder_factory(parse_device(args.device)),
        transcriber_factory(args.model_path),
        chat_factory(),
        args.seconds,
        output_fn,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
