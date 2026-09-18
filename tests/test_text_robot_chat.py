"""Integration tests for the keyboard-to-MockHead baseline."""

import pytest

from examples.text_robot_chat import run_application, run_cli, run_turn
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


def test_default_text_application_does_not_construct_jaw_hardware():
    chat = FakeJsonChat(
        {"response": "unused", "expression": "neutral", "intensity": 1}
    )
    factory_calls = []

    def failing_jaw_head_factory():
        factory_calls.append(True)
        raise AssertionError("default mode must not create jaw hardware")

    run_application(
        chat,
        jaw_hardware=False,
        input_fn=lambda _: "/quit",
        jaw_head_factory=failing_jaw_head_factory,
    )

    assert factory_calls == []


def test_jaw_hardware_option_enables_once_sends_each_expression_and_stops():
    chat = FakeJsonChat(
        {"response": "좋아요!", "expression": "happy", "intensity": 2}
    )
    events = []

    class FakeJawHead:
        def enable(self):
            events.append("enable")

        def set_expression(self, expression):
            events.append(("jaw", expression.jaw))

        def stop(self):
            events.append("stop")

    messages = iter(["좋은 소식이 있어", "/quit"])
    run_application(
        chat,
        jaw_hardware=True,
        input_fn=lambda _: next(messages),
        jaw_head_factory=FakeJawHead,
    )

    assert events == [
        "enable",
        ("jaw", map_expression("happy", 2).jaw),
        "stop",
    ]


def test_invalid_ai_expression_does_not_send_a_jaw_pulse():
    chat = FakeJsonChat(
        {"response": "invalid", "expression": "unsupported", "intensity": 2}
    )
    events = []

    class FakeJawHead:
        def enable(self):
            events.append("enable")

        def set_expression(self, expression):
            events.append(("pulse", expression.jaw))

        def stop(self):
            events.append("stop")

    messages = iter(["테스트", "/quit"])
    run_application(
        chat,
        jaw_hardware=True,
        input_fn=lambda _: next(messages),
        jaw_head_factory=FakeJawHead,
    )

    assert events == ["enable", "stop"]


def test_jaw_hardware_logs_the_normalized_value_pulse_command_and_response():
    from inmoove.hardware.jaw_head import JawCommandExchange, JawCommandResult

    chat = FakeJsonChat(
        {"response": "안녕하세요!", "expression": "neutral", "intensity": 1}
    )
    outputs: list[str] = []

    class FakeJawHead:
        def enable(self):
            pass

        def set_expression(self, expression):
            assert expression.jaw == 0.5
            return JawCommandResult(
                normalized_jaw=0.5,
                original_target_pulse=305,
                amplified_target_pulse=305,
                exchanges=(
                    JawCommandExchange(
                        command="CH15_SET,305",
                        arduino_response="OK CH15 jaw pulse set to 305",
                    ),
                    JawCommandExchange(
                        command="CH15_SET,305",
                        arduino_response="OK CH15 jaw pulse set to 305",
                    ),
                    JawCommandExchange(
                        command="CH15_SET,305",
                        arduino_response="OK CH15 jaw pulse set to 305",
                    ),
                ),
            )

        def stop(self):
            pass

    messages = iter(["안녕", "/quit"])
    run_application(
        chat,
        jaw_hardware=True,
        input_fn=lambda _: next(messages),
        output_fn=outputs.append,
        jaw_head_factory=FakeJawHead,
    )

    assert outputs[-10:] == [
        "Jaw hardware:",
        "normalized jaw = 0.5",
        "original target pulse = 305",
        "amplified target pulse = 305",
        "PC -> Arduino: CH15_SET,305",
        "Arduino -> PC: OK CH15 jaw pulse set to 305",
        "PC -> Arduino: CH15_SET,305",
        "Arduino -> PC: OK CH15 jaw pulse set to 305",
        "PC -> Arduino: CH15_SET,305",
        "Arduino -> PC: OK CH15 jaw pulse set to 305",
    ]


def test_default_text_application_does_not_print_jaw_hardware_logs():
    chat = FakeJsonChat(
        {"response": "안녕하세요!", "expression": "neutral", "intensity": 1}
    )
    outputs: list[str] = []
    messages = iter(["안녕", "/quit"])

    run_application(
        chat,
        jaw_hardware=False,
        input_fn=lambda _: next(messages),
        output_fn=outputs.append,
    )

    assert not any("Jaw hardware" in output or "CH15" in output for output in outputs)
