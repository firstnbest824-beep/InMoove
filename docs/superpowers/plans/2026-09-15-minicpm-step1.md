# MiniCPM Step 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a keyboard-only local MiniCPM-V chat that responds to a greeting without camera, STT, JSON, or robot-control integration.

**Architecture:** `MiniCPMTextChat` is a small lazy-loading adapter around the official Transformers pipeline. A standalone example owns stdin/stdout only, so CUDA/model download is isolated from imports and unit tests.

**Tech Stack:** Python 3.10, PyTorch CUDA, Hugging Face Transformers, Accelerate, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-minicpm-step1-design.md`

## Global Constraints

- Use `openbmb/MiniCPM-V-4.6`; do not attempt MiniCPM-o 4.5 on the 16GB RTX 5080.
- Load the model only at first `reply()` call.
- Do not require CUDA, network access, or the Transformers package for unit tests.
- Run tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest` because the globally installed ROS `launch_testing` plugin is incompatible with pytest 9.
- Use `python3`, not `python`.

---

### Task 1: Lazy MiniCPM text-chat adapter

**Files:**
- Create: `inmoove/inference/__init__.py`
- Create: `inmoove/inference/minicpm.py`
- Create: `tests/test_minicpm_text_chat.py`

**Interfaces:**
- Produces: `class MiniCPMTextChat(model_id: str = "openbmb/MiniCPM-V-4.6", pipeline_factory: Callable[..., Callable[..., object]] | None = None)`.
- Produces: `MiniCPMTextChat.reply(message: str) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
def test_reply_sends_one_text_message_and_returns_assistant_content():
    calls = []
    def fake_pipeline_factory(*args, **kwargs):
        calls.append((args, kwargs))
        return lambda *, text, **_: [{"generated_text": text + [{"role": "assistant", "content": "안녕하세요!"}]}]

    chat = MiniCPMTextChat(pipeline_factory=fake_pipeline_factory)

    assert chat.reply("안녕") == "안녕하세요!"
    assert calls[0][0] == ("image-text-to-text",)

def test_reply_rejects_blank_message_before_loading_model():
    chat = MiniCPMTextChat(pipeline_factory=lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="message must not be blank"):
        chat.reply("   ")
```

- [ ] **Step 2: Run the tests to verify the expected failure**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_minicpm_text_chat.py -v`

Expected: FAIL during collection because `inmoove.inference` does not exist.

- [ ] **Step 3: Write the minimal adapter**

```python
messages = [{"role": "user", "content": [{"type": "text", "text": message}]}]
result = self._pipeline(text=messages, max_new_tokens=128, do_sample=False)
return _assistant_content(result)
```

Use a private lazy importer for `transformers.pipeline`, and make `_assistant_content` return the last assistant `content` field from the pipeline's generated conversation.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_minicpm_text_chat.py -v`

Expected: PASS with two tests.

- [ ] **Step 5: Commit the adapter**

```bash
git add inmoove/inference tests/test_minicpm_text_chat.py
git commit -m "feat: add lazy MiniCPM text chat adapter"
```

### Task 2: Keyboard chat executable and optional inference dependencies

**Files:**
- Create: `examples/minicpm_text_chat.py`
- Create: `requirements-minicpm.txt`
- Modify: `README.md`
- Modify: `tests/test_minicpm_text_chat.py`

**Interfaces:**
- Consumes: `MiniCPMTextChat.reply(message: str) -> str`.
- Produces: `run_cli(chat: MiniCPMTextChat, input_fn: Callable[[str], str] = input, output_fn: Callable[[str], None] = print) -> None`.

- [ ] **Step 1: Write the failing CLI test**

```python
def test_run_cli_prints_reply_for_greeting_and_stops_on_quit():
    prompts = iter(["안녕", "/quit"])
    output = []

    run_cli(FakeChat("안녕하세요!"), input_fn=lambda _: next(prompts), output_fn=output.append)

    assert output == ["AI: 안녕하세요!"]
```

- [ ] **Step 2: Run the test to verify the expected failure**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_minicpm_text_chat.py::test_run_cli_prints_reply_for_greeting_and_stops_on_quit -v`

Expected: FAIL because `examples.minicpm_text_chat` does not exist.

- [ ] **Step 3: Write the minimal executable and dependency file**

```python
def run_cli(chat, input_fn=input, output_fn=print):
    while True:
        try:
            message = input_fn("You: ").strip()
        except EOFError:
            return
        if message == "/quit":
            return
        if message:
            output_fn(f"AI: {chat.reply(message)}")
```

Create `requirements-minicpm.txt` with `transformers>=5.0,<6` and `accelerate>=1.12`. Document `python3 -m pip install -r requirements-minicpm.txt` and `python3 examples/minicpm_text_chat.py` in the README.

- [ ] **Step 4: Run focused and full unit tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q`

Expected: PASS for all existing tests plus the new MiniCPM tests.

- [ ] **Step 5: Commit the executable and documentation**

```bash
git add examples/minicpm_text_chat.py requirements-minicpm.txt README.md tests/test_minicpm_text_chat.py
git commit -m "feat: add MiniCPM keyboard chat example"
```

### Task 3: Live local model verification

**Files:**
- No source changes expected.

**Interfaces:**
- Consumes: `python3 examples/minicpm_text_chat.py`.
- Produces: a non-empty answer for `안녕`.

- [ ] **Step 1: Install optional inference dependencies**

Run: `python3 -m pip install -r requirements-minicpm.txt`

- [ ] **Step 2: Run a one-turn local chat**

Run: `printf '안녕\n/quit\n' | python3 examples/minicpm_text_chat.py`

Expected: model download on first run, then an `AI:` line with non-empty Korean text.

- [ ] **Step 3: Record actual outcome**

If the model returns text, keep the working command in the README. If model loading fails, stop and report the exact error without changing model architecture or adding a fallback model.
