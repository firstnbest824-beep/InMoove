# Microphone STT Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record a short microphone clip, transcribe it locally, and pass the resulting text to existing `MiniCPMTextChat.reply_json()`.

**Architecture:** `inmoove.audio` owns valid fixed-format PCM recording and a replaceable `Transcriber` protocol. The example composes injected recorder, transcriber, and MiniCPM dependencies, enabling fake-only tests and keeping all microphone/STT errors outside the existing inference module.

**Tech Stack:** Python 3.10, numpy, sounddevice, Vosk, pytest, existing MiniCPM pipeline adapter.

**Spec:** `docs/superpowers/specs/2026-09-19-microphone-stt-design.md`

## Global Constraints

- Audio buffers must be mono, 16,000 Hz, 16-bit PCM with byte-aligned samples.
- `--device` decimal digits become an integer index; every other value remains a device-name string.
- Recording failures print `Recording error: <reason>`; transcription failures print `STT error: <reason>`.
- No serial, Arduino, PCA9685, servo, camera, TTS, or speaker operation.
- No runtime model download; `--model-path` names an already downloaded Vosk model directory.
- Do not commit or push.

---

### Task 1: Audio buffer and microphone recorder boundary

**Files:**
- Create: `inmoove/audio/__init__.py`
- Create: `inmoove/audio/recorder.py`
- Create: `tests/test_audio_recorder.py`

**Interfaces:**
- Produces: `AudioBuffer(samples: bytes, sample_rate: int = 16000, channels: int = 1, sample_width_bytes: int = 2)`.
- Produces: `RecordingError(RuntimeError)`, `AudioRecorder` protocol, and `SoundDeviceRecorder(device: int | str | None = None)` with `record(duration_seconds: float) -> AudioBuffer`.
- Consumed later by: `Transcriber.transcribe(audio: AudioBuffer)` and `examples.microphone_chat.run_once()`.

- [ ] **Step 1: Write failing audio-boundary tests**

```python
import pytest

from inmoove.audio.recorder import AudioBuffer


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"sample_rate": 8000}, "16000"),
        ({"channels": 2}, "mono"),
        ({"sample_width_bytes": 1}, "16-bit"),
        ({"samples": b"\x00"}, "aligned"),
    ],
)
def test_audio_buffer_rejects_format_that_vosk_cannot_consume(kwargs, error):
    defaults = {"samples": b"\x00\x00", "sample_rate": 16000,
                "channels": 1, "sample_width_bytes": 2}
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=error):
        AudioBuffer(**defaults)
```

- [ ] **Step 2: Run the tests to verify the missing module fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_audio_recorder.py`

Expected: failure because `inmoove.audio` does not exist.

- [ ] **Step 3: Implement the minimal recorder boundary**

```python
@dataclass(frozen=True)
class AudioBuffer:
    samples: bytes
    sample_rate: int = 16000
    channels: int = 1
    sample_width_bytes: int = 2

    def __post_init__(self) -> None:
        if self.sample_rate != 16000:
            raise ValueError("AudioBuffer sample rate must be 16000 Hz")
        if self.channels != 1:
            raise ValueError("AudioBuffer must be mono")
        if self.sample_width_bytes != 2:
            raise ValueError("AudioBuffer must use 16-bit samples")
        if len(self.samples) % self.sample_width_bytes:
            raise ValueError("AudioBuffer samples must be byte-aligned")
```

`SoundDeviceRecorder.record()` validates a positive duration, lazily imports
`sounddevice`, records `round(duration_seconds * 16000)` mono `int16` frames,
waits for completion, then wraps `recording.tobytes()` in `AudioBuffer`.
Translate audio-library exceptions to `RecordingError`.

- [ ] **Step 4: Run the recorder tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_audio_recorder.py`

Expected: all audio format-validation and fake-recording tests pass without a
physical microphone.

### Task 2: Replaceable local Vosk transcriber

**Files:**
- Create: `inmoove/audio/stt.py`
- Modify: `inmoove/audio/__init__.py`
- Modify: `tests/test_audio_recorder.py`

**Interfaces:**
- Consumes: `AudioBuffer` from `inmoove.audio.recorder`.
- Produces: `Transcriber` protocol with `transcribe(audio: AudioBuffer) -> str`, `TranscriptionError`, `EmptyTranscriptionError`, and `VoskTranscriber(model_path, model_factory=None, recognizer_factory=None)`.
- Consumed later by: `examples.microphone_chat.run_once()`.

- [ ] **Step 1: Write failing Vosk-boundary tests**

```python
def test_vosk_transcriber_rejects_empty_text_from_recognizer():
    transcriber = VoskTranscriber(
        "unused-model",
        model_factory=lambda _: object(),
        recognizer_factory=lambda *_: FakeRecognizer('{"text": "  "}'),
    )
    with pytest.raises(EmptyTranscriptionError, match="empty"):
        transcriber.transcribe(AudioBuffer(b"\x00\x00"))
```

Use a fake recognizer whose `AcceptWaveform()` returns `True` and whose
`FinalResult()` returns the declared JSON string. Add a malformed JSON case
that raises `TranscriptionError`.

- [ ] **Step 2: Run the tests to verify the class is absent**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_audio_recorder.py`

Expected: failure because `VoskTranscriber` is not defined.

- [ ] **Step 3: Implement Vosk behind the protocol**

```python
class Transcriber(Protocol):
    def transcribe(self, audio: AudioBuffer) -> str: ...

class VoskTranscriber:
    def transcribe(self, audio: AudioBuffer) -> str:
        recognizer = self._recognizer_factory(self._model(), audio.sample_rate)
        recognizer.AcceptWaveform(audio.samples)
        result = json.loads(recognizer.FinalResult())
        text = result.get("text")
        if not isinstance(text, str):
            raise TranscriptionError("Vosk result does not contain text")
        if not text.strip():
            raise EmptyTranscriptionError("STT result is empty")
        return text.strip()
```

The default factories lazy-import `vosk`, validate `model_path.is_dir()`, and
translate missing dependencies, unavailable models, Vosk exceptions, and bad
JSON to `TranscriptionError`.

- [ ] **Step 4: Run the transcriber tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_audio_recorder.py`

Expected: fake Vosk success, empty output, and malformed result cases pass.

### Task 3: Injectable microphone-to-MiniCPM example and regression coverage

**Files:**
- Create: `examples/microphone_chat.py`
- Create: `tests/test_microphone_chat.py`
- Create: `requirements-audio.txt`

**Interfaces:**
- Consumes: `AudioRecorder`, `RecordingError`, `Transcriber`, `TranscriptionError`, and any object with `reply_json(text: str) -> dict`.
- Produces: `run_once(recorder, transcriber, chat, duration_seconds, output_fn=print) -> dict | None`, `parse_device(value: str | None) -> int | str | None`, and `main(argv=None, ..., output_fn=print) -> int`.

- [ ] **Step 1: Write failing application-flow tests**

```python
def test_run_once_passes_korean_transcript_to_reply_json_and_prints_reply():
    outputs = []
    result = run_once(
        FakeRecorder(AudioBuffer(b"\x00\x00")),
        FakeTranscriber("오늘 기분이 좋아"),
        FakeChat({"response": "정말 잘됐네요!", "expression": "happy", "intensity": 2}),
        duration_seconds=3,
        output_fn=outputs.append,
    )
    assert result["expression"] == "happy"
    assert outputs == ["You said: 오늘 기분이 좋아", "AI response: 정말 잘됐네요!",
                       "expression: happy", "intensity: 2"]
```

Add separate tests that assert `RecordingError` prints `Recording error:` and
`TranscriptionError` or an empty fake transcript prints `STT error:`, with a
fake chat that raises if called. Add CLI tests that pass `--device 3` and
`--device USB Microphone` through fake recorder factories and assert the
received values are `3` and `"USB Microphone"` respectively.

- [ ] **Step 2: Run the tests to verify the example is absent**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_microphone_chat.py`

Expected: failure because `examples.microphone_chat` does not exist.

- [ ] **Step 3: Implement the thin composition layer and optional dependencies**

```python
def parse_device(value: str | None) -> int | str | None:
    if value is None:
        return None
    return int(value) if value.isdecimal() else value

def run_once(recorder, transcriber, chat, duration_seconds, output_fn=print):
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
    reply = chat.reply_json(text)
    output_fn(f"You said: {text}")
    output_fn(f"AI response: {reply['response']}")
    output_fn(f"expression: {reply['expression']}")
    output_fn(f"intensity: {reply['intensity']}")
    return reply
```

Add `numpy`, `sounddevice`, and `vosk` to `requirements-audio.txt`. The CLI
requires `--model-path`, accepts `--seconds` with default `4.0`, and forwards
the parsed device value to `SoundDeviceRecorder`.

- [ ] **Step 4: Run focused and full regression tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_microphone_chat.py && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`

Expected: microphone fake-flow tests and all existing tests pass with no
microphone, Vosk model, serial port, or hardware required.
