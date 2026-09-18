# Microphone STT Baseline Design

## Goal

Add one local, non-streaming microphone path that records a short fixed
duration, transcribes it to Korean text, and sends that text to the existing
`MiniCPMTextChat.reply_json()` boundary. This work does not control hardware,
capture camera input, or produce TTS output.

## Scope

The command-line flow is:

```text
selected microphone device
  -> fixed-duration recording
  -> local Vosk transcription
  -> non-empty text validation
  -> MiniCPMTextChat.reply_json(text)
  -> response / expression / intensity printed
```

The baseline uses Vosk with a user-supplied local model directory. It never
downloads a model at runtime. Vosk is an implementation detail behind a small
`Transcriber` protocol, so a future `faster-whisper` transcriber can replace
it without changing the recorder, example flow, or MiniCPM boundary.

## Components

### `inmoove.audio.recorder`

- `AudioBuffer` represents mono, 16 kHz, 16-bit PCM audio and validates its
  sample rate, channel count, sample width, and byte alignment at creation.
  Invalid audio is rejected before it can reach Vosk.
- `AudioRecorder` is a protocol with `record(duration_seconds)`.
- `SoundDeviceRecorder` accepts an optional numeric or string `device` value.
  When omitted, it deliberately delegates device choice to `sounddevice`.
  Its dependency is imported only while recording, so tests do not require a
  microphone or audio driver.
- Missing input devices, audio-backend failures, and invalid recorded data are
  normalized to `RecordingError` with actionable context.

### `inmoove.audio.stt`

- `Transcriber` is a protocol with `transcribe(audio)`.
- `VoskTranscriber` lazily imports Vosk and receives a model directory at
  construction time. It converts a complete `AudioBuffer` in one operation.
- Empty or whitespace-only transcripts raise `EmptyTranscriptionError`.
- Dependency, model, decoding, and malformed-result failures surface as
  `TranscriptionError` with actionable context.

### `examples.microphone_chat`

- `run_once()` receives recorder, transcriber, and chat dependencies. This is
  the test seam and has no direct microphone or model dependency.
- It prints `You said: <text>`, calls `reply_json(text)`, then prints the
  response, expression, and intensity fields.
- It catches `TranscriptionError`, prints `STT error: <reason>`, and does not
  invoke MiniCPM in that case.
- The direct CLI requires `--model-path`; `--device` is optional and passes a
  numeric device index or a device name through to `sounddevice`. A value made
  only of decimal digits, such as `"3"`, becomes integer `3`; every other
  value, such as `"USB Microphone"`, remains a device-name string.
- It catches `RecordingError` separately as `Recording error: <reason>` and
  `TranscriptionError` as `STT error: <reason>`; neither case calls MiniCPM.

## Dependencies and setup

`requirements-audio.txt` keeps `numpy`, `sounddevice`, and `vosk` separate
from existing inference and hardware requirements. The user installs the
dependencies and downloads a compatible Vosk Korean model into a local
directory before running the example. The example does not access serial
ports, Arduino, PCA9685, servos, cameras, or speakers.

## Tests

`tests/test_microphone_chat.py` uses fake recorder, transcriber, and chat
objects to verify:

1. Korean transcript reaches `reply_json()` and the structured response is
   printed.
2. Invalid sample rate, channel count, sample width, and unaligned PCM bytes
   are rejected before transcription.
3. Empty transcripts are rejected before MiniCPM is called.
4. Recording and STT failures are printed distinctly and MiniCPM is not
   called.
5. CLI parsing converts `--device 3` to integer `3` and preserves
   `--device "USB Microphone"` as a string passed to the recorder factory.

Existing pytest regression remains part of completion verification.
