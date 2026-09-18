"""Tests for the provisional CH15/jaw-only hardware path."""

import pytest

from inmoove.face import FaceExpression


def face_with_jaw(jaw: float) -> FaceExpression:
    values = dict.fromkeys(FaceExpression.get_field_names(), 0.5)
    values["jaw"] = jaw
    return FaceExpression(**values)


@pytest.mark.parametrize(
    ("jaw", "expected_pulse"),
    [(0.0, 320), (0.5, 305), (1.0, 290)],
)
def test_provisional_jaw_calibration_maps_normalized_values_to_ch15_pulses(
    jaw, expected_pulse
):
    from inmoove.hardware.jaw_calibration import NormalizedPulseCalibration

    calibration = NormalizedPulseCalibration(closed_pulse=320, open_pulse=290)

    assert calibration.to_pulse(jaw) == expected_pulse


@pytest.mark.parametrize("jaw", [-0.01, 1.01, "0.5"])
def test_provisional_jaw_calibration_rejects_invalid_normalized_values(jaw):
    from inmoove.hardware.jaw_calibration import NormalizedPulseCalibration

    calibration = NormalizedPulseCalibration(closed_pulse=320, open_pulse=290)

    with pytest.raises((TypeError, ValueError)):
        calibration.to_pulse(jaw)


def test_ch15_transport_only_writes_enable_set_and_stop_commands():
    from inmoove.hardware.jaw_serial_transport import Ch15SerialTransport

    class FakeConnection:
        def __init__(self):
            self.writes: list[bytes] = []
            self.flush_calls = 0
            self.responses = [
                b"CH15 jaw test compiled; output disabled until CH15_SET\n",
                b"OK CH15 jaw test enabled; send CH15_SET,<290..320>\n",
                b"OK CH15 jaw pulse set to 305\n",
            ]

        def write(self, data: bytes) -> None:
            self.writes.append(data)

        def flush(self) -> None:
            self.flush_calls += 1

        def close(self) -> None:
            pass

        def readline(self) -> bytes:
            return self.responses.pop(0) if self.responses else b""

    connection = FakeConnection()
    transport = Ch15SerialTransport(
        connection_factory=lambda *_: connection,
        boot_wait_seconds=0,
        response_timeout_seconds=0.1,
    )

    transport.enable()
    response = transport.send_pulse(305)
    transport.stop()

    assert connection.writes == [
        b"CH15_TEST_ENABLE\n",
        b"CH15_SET,305\n",
        b"CH15_TEST_STOP\n",
    ]
    assert connection.flush_calls == 3
    assert response == "OK CH15 jaw pulse set to 305"
    assert all(b"ANGLES" not in command and b"CH0" not in command for command in connection.writes)


def test_ch15_enable_ignores_boot_banner_and_waits_for_its_confirmation():
    from inmoove.hardware.jaw_serial_transport import Ch15SerialTransport

    class FakeConnection:
        def __init__(self):
            self.writes: list[bytes] = []
            self.responses = [
                b"Step 5 receiver READY; PCA9685 output disabled\n",
                b"",
                b"OK CH15 jaw test enabled; send CH15_SET,<290..320>\n",
            ]

        def write(self, data: bytes) -> None:
            self.writes.append(data)

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

        def readline(self) -> bytes:
            return self.responses.pop(0) if self.responses else b""

    connection = FakeConnection()
    transport = Ch15SerialTransport(
        connection_factory=lambda *_: connection,
        boot_wait_seconds=0,
        response_timeout_seconds=0.1,
    )

    transport.enable()

    assert connection.writes == [b"CH15_TEST_ENABLE\n"]


def test_ch15_enable_times_out_without_confirmation_and_never_sends_a_pulse():
    from inmoove.hardware.jaw_serial_transport import Ch15SerialTransport

    class FakeConnection:
        def __init__(self):
            self.writes: list[bytes] = []

        def write(self, data: bytes) -> None:
            self.writes.append(data)

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

        def readline(self) -> bytes:
            return b""

    connection = FakeConnection()
    transport = Ch15SerialTransport(
        connection_factory=lambda *_: connection,
        boot_wait_seconds=0,
        response_timeout_seconds=0,
    )

    with pytest.raises(RuntimeError, match="did not confirm CH15 manual enable"):
        transport.enable()

    assert connection.writes == [b"CH15_TEST_ENABLE\n"]


def test_ch15_transport_rejects_a_pulse_outside_the_provisional_range():
    from inmoove.hardware.jaw_serial_transport import Ch15SerialTransport

    transport = Ch15SerialTransport(connection_factory=lambda *_: None)

    with pytest.raises(ValueError, match="290.*320"):
        transport.send_pulse(321)


def test_jaw_hardware_head_only_converts_and_sends_the_jaw_field():
    from inmoove.hardware.jaw_head import JawHardwareHead

    class FakeTransport:
        def __init__(self):
            self.events: list[tuple[str, int | None]] = []

        def enable(self) -> None:
            self.events.append(("enable", None))

        def send_pulse(self, pulse: int) -> str:
            self.events.append(("pulse", pulse))
            return f"OK CH15 jaw pulse set to {pulse}"

        def stop(self) -> None:
            self.events.append(("stop", None))

    transport = FakeTransport()
    pauses = []
    head = JawHardwareHead(transport=transport, sleep_fn=pauses.append)

    head.enable()
    result = head.set_expression(face_with_jaw(0.0))
    head.stop()

    assert transport.events == [
        ("enable", None),
        ("pulse", 305),
        ("pulse", 320),
        ("pulse", 305),
        ("stop", None),
    ]
    assert pauses == [0.2, 1.0]
    assert result.normalized_jaw == 0.0
    assert result.original_target_pulse == 320
    assert result.amplified_target_pulse == 320
    assert [exchange.command for exchange in result.exchanges] == [
        "CH15_SET,305",
        "CH15_SET,320",
        "CH15_SET,305",
    ]


@pytest.mark.parametrize(
    ("target", "expected"),
    [(290, 290), (305, 305), (309, 317), (320, 320)],
)
def test_jaw_demo_amplifies_away_from_neutral_and_clamps_to_safe_range(
    target, expected
):
    from inmoove.hardware.jaw_head import amplify_jaw_demo_pulse

    assert amplify_jaw_demo_pulse(target) == expected
