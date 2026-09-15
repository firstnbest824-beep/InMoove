from pathlib import Path
import subprocess
import sys

import pytest
from PIL import Image

from examples.minicpm_text_chat import run_cli
from inmoove.inference.minicpm import MiniCPMTextChat


def test_reply_returns_generated_assistant_greeting():
    calls = []

    def pipeline(**kwargs):
        calls.append(kwargs)
        return [
            {
                "generated_text": [
                    {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hello! How can I help?"}],
                    },
                ]
            }
        ]

    chat = MiniCPMTextChat(pipeline_factory=lambda *_, **__: pipeline)

    assert chat.reply("Hello") == "Hello! How can I help?"
    assert calls == [
        {
            "text": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ],
            "max_new_tokens": 128,
            "do_sample": False,
        }
    ]


def test_blank_message_is_rejected_before_loading_pipeline():
    loaded = False

    def pipeline_factory(**_):
        nonlocal loaded
        loaded = True
        return lambda **__: []

    chat = MiniCPMTextChat(pipeline_factory=pipeline_factory)

    with pytest.raises(ValueError, match="message must not be blank"):
        chat.reply("  \n\t")

    assert loaded is False


def test_reply_rejects_missing_content_on_final_assistant_message():
    def pipeline(**_):
        return [
            {
                "generated_text": [
                    {"role": "assistant", "content": "Earlier response"},
                    {"role": "assistant", "content": ""},
                ]
            }
        ]

    chat = MiniCPMTextChat(pipeline_factory=lambda *_, **__: pipeline)

    with pytest.raises(RuntimeError, match="no final assistant content"):
        chat.reply("Hello")


def test_reply_json_returns_parsed_structured_response():
    calls = []

    def pipeline(**kwargs):
        calls.append(kwargs)
        return [
            {
                "generated_text": [
                    {
                        "role": "assistant",
                        "content": (
                            '{"response":"정말 잘됐네요!",'
                            '"expression":"happy","intensity":2}'
                        ),
                    }
                ]
            }
        ]

    chat = MiniCPMTextChat(pipeline_factory=lambda *_, **__: pipeline)

    assert chat.reply_json("좋은 소식이 있어.") == {
        "response": "정말 잘됐네요!",
        "expression": "happy",
        "intensity": 2,
    }
    prompt = calls[0]["text"][0]["content"][0]["text"]
    assert "valid JSON object" in prompt
    assert "좋은 소식이 있어." in prompt


@pytest.mark.parametrize(
    "model_output, error",
    [
        ("not json", "valid JSON object"),
        ("[]", "JSON object"),
        ('{"response":"안녕"}', "missing required fields"),
        (
            '{"response":"안녕","expression":"happy","intensity":"2"}',
            "intensity must be a JSON number",
        ),
    ],
)
def test_reply_json_rejects_invalid_structured_output(model_output, error):
    def pipeline(**_):
        return [
            {
                "generated_text": [
                    {"role": "assistant", "content": model_output},
                ]
            }
        ]

    chat = MiniCPMTextChat(pipeline_factory=lambda *_, **__: pipeline)

    with pytest.raises(ValueError, match=error):
        chat.reply_json("응답 형식을 지켜줘.")


def test_reply_with_image_returns_answer_for_local_photo(tmp_path):
    image_path = tmp_path / "desk.jpg"
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(image_path)
    calls = []

    def pipeline(**kwargs):
        calls.append(kwargs)
        return [
            {
                "generated_text": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "A person is smiling."}],
                    }
                ]
            }
        ]

    chat = MiniCPMTextChat(pipeline_factory=lambda *_, **__: pipeline)

    assert (
        chat.reply_with_image(image_path, "Describe the photo in one sentence.")
        == "A person is smiling."
    )
    image_content = calls[0]["text"][0]["content"][0]
    assert image_content["type"] == "image"
    assert image_content["image"].size == (2, 2)
    assert calls[0]["text"][0]["content"][1] == {
        "type": "text",
        "text": "Describe the photo in one sentence.",
    }


def test_reply_with_image_json_returns_parsed_structured_response(tmp_path):
    image_path = tmp_path / "desk.jpg"
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(image_path)
    calls = []

    def pipeline(**kwargs):
        calls.append(kwargs)
        return [
            {
                "generated_text": [
                    {
                        "role": "assistant",
                        "content": (
                            '{"response":"밝은 분위기입니다.",'
                            '"expression":"happy","intensity":2}'
                        ),
                    }
                ]
            }
        ]

    chat = MiniCPMTextChat(pipeline_factory=lambda *_, **__: pipeline)

    assert chat.reply_with_image_json(image_path, "사진 분위기를 설명해줘.") == {
        "response": "밝은 분위기입니다.",
        "expression": "happy",
        "intensity": 2,
    }
    content = calls[0]["text"][0]["content"]
    assert content[0]["type"] == "image"
    assert "valid JSON object" in content[1]["text"]
    assert "사진 분위기를 설명해줘." in content[1]["text"]


def test_reply_with_image_rejects_missing_file_before_loading_pipeline(tmp_path):
    loaded = False

    def pipeline_factory(**_):
        nonlocal loaded
        loaded = True
        return lambda **__: []

    chat = MiniCPMTextChat(pipeline_factory=pipeline_factory)

    with pytest.raises(FileNotFoundError, match="image file does not exist"):
        chat.reply_with_image(tmp_path / "missing.jpg", "Describe the photo.")

    assert loaded is False


def test_reply_with_image_rejects_blank_message_before_loading_pipeline(tmp_path):
    image_path = tmp_path / "desk.jpg"
    Image.new("RGB", (2, 2)).save(image_path)
    loaded = False

    def pipeline_factory(**_):
        nonlocal loaded
        loaded = True
        return lambda **__: []

    chat = MiniCPMTextChat(pipeline_factory=pipeline_factory)

    with pytest.raises(ValueError, match="message must not be blank"):
        chat.reply_with_image(image_path, "  \n\t")

    assert loaded is False


def test_image_cli_prints_reply_for_one_local_photo(tmp_path):
    from examples.minicpm_image_chat import run_once

    image_path = tmp_path / "photo.jpg"
    outputs: list[str] = []

    class ImageChat:
        def reply_with_image(self, received_path: Path, message: str) -> str:
            assert received_path == image_path
            assert message == "사진을 설명해줘."
            return "웃으며 엄지를 든 사람이 보입니다."

    run_once(ImageChat(), image_path, "사진을 설명해줘.", output_fn=outputs.append)

    assert outputs == ["AI: 웃으며 엄지를 든 사람이 보입니다."]


def test_run_cli_prints_reply_for_greeting_and_stops_on_quit():
    class GreetingChat:
        def reply(self, message: str) -> str:
            assert message == "안녕"
            return "안녕하세요!"

    messages = iter(["안녕", "/quit"])
    outputs: list[str] = []

    run_cli(GreetingChat(), input_fn=lambda _: next(messages), output_fn=outputs.append)

    assert outputs == ["AI: 안녕하세요!"]


def test_direct_script_exits_on_quit_without_loading_model():
    repository_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "examples/minicpm_text_chat.py"],
        cwd=repository_root,
        input="/quit\n",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
