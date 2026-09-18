"""Integration tests for the keyboard-to-MockHead baseline."""

import pytest

from examples.text_robot_chat import run_cli, run_turn
from inmoove.face import MockHead, map_expression


class FakeJsonChat:
    def __init__(self, reply: dict):
        self.reply = reply
        self.received_messages: list[str] = []

    def reply_json(self, message: str) -> dict:
        self.received_messages.append(message)
        return self.reply


def test_run_turn_passes_text_through_json_mapping_to_mock_head():
    chat = FakeJsonChat(
        {
            "response": "정말 잘됐네요!",
            "expression": "happy",
            "intensity": 2,
        }
    )
    head = MockHead(verbose=False)
    outputs: list[str] = []

    result = run_turn(chat, head, "오늘 시험에 합격했어!", output_fn=outputs.append)

    expected = map_expression("happy", 2)
    assert chat.received_messages == ["오늘 시험에 합격했어!"]
    assert result.get_all_values() == expected.get_all_values()
    assert head.get_current_expression() is not None
    assert head.get_current_expression().get_all_values() == expected.get_all_values()
    assert len(head.get_expression_history()) == 1
    assert outputs == [
        "Robot response: 정말 잘됐네요!",
        "Expression: happy",
        "Intensity: 2",
        "",
        "FaceExpression:",
        *[
            f"{name} = {value}"
            for name, value in zip(expected.get_field_names(), expected.get_all_values())
        ],
    ]


def test_run_cli_skips_blank_input_and_exits_on_quit():
    chat = FakeJsonChat(
        {"response": "unused", "expression": "neutral", "intensity": 1}
    )
    head = MockHead(verbose=False)
    messages = iter(["   ", "/quit"])

    run_cli(chat, head, input_fn=lambda _: next(messages))

    assert chat.received_messages == []
    assert head.get_current_expression() is None
    assert head.get_expression_history() == []


@pytest.mark.parametrize(
    "expression, intensity, error",
    [
        ("surprised", 2, "unsupported expression"),
        ("happy", 4, "intensity must be an integer from 1 to 3"),
    ],
)
def test_run_cli_reports_invalid_expression_or_intensity_without_changing_head(
    expression, intensity, error
):
    chat = FakeJsonChat(
        {"response": "invalid", "expression": expression, "intensity": intensity}
    )
    head = MockHead(verbose=False)
    messages = iter(["테스트", "/quit"])
    outputs: list[str] = []

    run_cli(chat, head, input_fn=lambda _: next(messages), output_fn=outputs.append)

    assert chat.received_messages == ["테스트"]
    assert outputs == [f"Face expression error: {error}: {expression!r}" if expression == "surprised" else f"Face expression error: {error}"]
    assert head.get_current_expression() is None
    assert head.get_expression_history() == []
