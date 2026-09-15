# MiniCPM Step 1 Design

## Goal

Run a pre-trained MiniCPM model locally and provide a keyboard-only text chat that replies to `안녕`.

## Scope

- Use `openbmb/MiniCPM-V-4.6`, the official 1B MiniCPM-V checkpoint.
- Load the model lazily through the Hugging Face Transformers `image-text-to-text` pipeline.
- Provide a CLI that reads one text prompt at a time and prints the assistant response.
- Keep model loading, camera input, STT, JSON output, and robot control outside this step.

## Constraints

- The development PC has an RTX 5080 with 16GB VRAM; MiniCPM-o 4.5 is excluded because its official demo requires more than 28GB VRAM.
- The model download must occur only when the CLI runs, never when importing `inmoove` or executing unit tests.
- Unit tests must not download a model or require CUDA.
- The current system Python command is `python3`.

## Design

`inmoove.inference.MiniCPMTextChat` owns the small boundary to Transformers. Its constructor receives a model identifier and, optionally, a pipeline factory for tests. Its `reply(message)` method validates non-empty input, loads the pipeline on first use, passes a single user text message, and returns the final assistant text from the pipeline result.

`examples/minicpm_text_chat.py` remains a thin CLI adapter. It constructs `MiniCPMTextChat`, repeatedly accepts keyboard input, and exits on `/quit` or end-of-input. It has no robot, image, microphone, or conversation-memory dependency.

## Error Handling

- Empty prompts raise `ValueError` before model inference.
- Missing optional inference dependencies raise an actionable `RuntimeError` naming `requirements-minicpm.txt`.
- Unexpected pipeline result shapes raise an actionable `RuntimeError` rather than printing a non-answer.

## Verification

- Unit test: a fake pipeline receives `안녕` and its assistant reply is returned.
- Unit test: the CLI prints the reply and exits cleanly.
- Live check: after installing the optional dependencies, pipe `안녕` then `/quit` into the example and confirm a non-empty model response.
