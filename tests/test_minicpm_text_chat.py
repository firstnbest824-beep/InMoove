import pytest

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
