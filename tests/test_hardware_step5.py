"""Step 5의 named calibration, 16-channel mapping, serial transport tests."""

import importlib
from pathlib import Path

import pytest

from inmoove.face import FaceExpression


RECEIVER_SKETCH = (
    Path(__file__).resolve().parents[1]
    / "arduino"
    / "face_servo_receiver"
    / "face_servo_receiver.ino"
)


def hardware_modules():
    """Load Step 5 modules at test time so their absence is a test failure."""
    calibration = importlib.import_module("inmoove.hardware.servo_calibration")
    real_head = importlib.import_module("inmoove.hardware.real_head")
    transport = importlib.import_module("inmoove.hardware.serial_transport")
    return calibration, real_head, transport


def test_receiver_declares_a_guarded_ch15_jaw_manual_calibration_mode():
    """CH15/jaw starts released and requires an explicit serial enable command."""
    source = RECEIVER_SKETCH.read_text()

    assert "bool ch15TestEnabled = false;" in source
    assert 'const char *const CH15_ACTUATOR_NAME = "jaw";' in source
    assert "const uint8_t CH15_TEST_CHANNEL = 15;" in source
    assert "const uint16_t CH15_TEST_MIN_PULSE = 290;" in source
    assert "const uint16_t CH15_TEST_MAX_PULSE = 320;" in source
    assert "releaseCh15Output();  // Ensure CH15 is released at boot" in source


def test_receiver_has_ch15_only_manual_commands_and_release_path():
    """The jaw test has no automatic motion and releases only CH15 on stop."""
    source = RECEIVER_SKETCH.read_text()

    assert 'strcmp(packet, "CH15_TEST_ENABLE")' in source
    assert 'strcmp(packet, "CH15_TEST_STOP")' in source
    assert 'strcmp(command, "CH15_SET")' in source
    assert "ERR send CH15_TEST_ENABLE before CH15_SET" in source
    assert "pwm.setPWM(CH15_TEST_CHANNEL, 0, 0);" in source
    assert "pwm.setPWM(CH15_TEST_CHANNEL, 0, static_cast<uint16_t>(pulse));" in source


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, 10.0),
        (0.5, 20.0),
        (1.0, 30.0),
    ],
)
def test_servo_calibration_converts_normalized_endpoints_and_midpoint(value, expected):
    """A calibration linearly converts its validated normalized input."""
    calibration, _, _ = hardware_modules()
    servo = calibration.ServoCalibration(min_angle=10.0, max_angle=30.0)

    assert servo.to_angle(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, 30.0),
        (0.5, 20.0),
        (1.0, 10.0),
    ],
)
def test_reversed_servo_calibration_inverts_the_normalized_direction(value, expected):
    """Reversal changes direction without changing the named servo's limits."""
    calibration, _, _ = hardware_modules()
    servo = calibration.ServoCalibration(10.0, 30.0, reversed=True)

    assert servo.to_angle(value) == expected


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_servo_calibration_rejects_out_of_range_normalized_values(value):
    """Bad normalized data must be rejected instead of silently clamped."""
    calibration, _, _ = hardware_modules()
    servo = calibration.ServoCalibration(10.0, 30.0)

    with pytest.raises(ValueError, match="0.0.*1.0"):
        servo.to_angle(value)


def test_real_head_uses_named_calibrations_in_fixed_channel_order():
    """A distinct calibration per name proves jaw cannot use another channel's limits."""
    calibration, real_head, _ = hardware_modules()
    calibrations = {
        name: calibration.ServoCalibration(channel * 10.0, channel * 10.0 + 10.0)
        for name, channel in real_head.SERVO_CHANNELS.items()
    }
    head = real_head.RealHead(calibrations)

    assert head.convert(FaceExpression.neutral()) == tuple(
        channel * 10.0 + 5.0 for channel in range(16)
    )


def test_real_head_declares_the_face_expression_to_pca_channel_mapping():
    """CH0~CH15 exactly follow the canonical FaceExpression field order."""
    _, real_head, _ = hardware_modules()

    assert tuple(real_head.SERVO_CHANNELS.items()) == tuple(
        zip(FaceExpression.get_field_names(), range(16))
    )


def test_real_head_rejects_missing_named_calibration():
    """Every canonical actuator needs its own named calibration before conversion."""
    calibration, real_head, _ = hardware_modules()
    calibrations = {
        name: calibration.ServoCalibration(10.0, 30.0)
        for name in FaceExpression.get_field_names()
        if name != "jaw"
    }

    with pytest.raises(ValueError, match="calibration names"):
        real_head.RealHead(calibrations)


def test_real_head_sends_converted_angles_to_an_injected_transport():
    """Face conversion reaches a supplied transport without requiring a serial device."""
    calibration, real_head, _ = hardware_modules()
    calibrations = {
        name: calibration.ServoCalibration(10.0, 30.0)
        for name in FaceExpression.get_field_names()
    }

    class FakeTransport:
        def __init__(self):
            self.sent_angles = []

        def send_angles(self, angles):
            self.sent_angles.append(tuple(angles))

    fake_transport = FakeTransport()
    head = real_head.RealHead(calibrations, transport=fake_transport)

    head.set_expression(FaceExpression.neutral())

    assert fake_transport.sent_angles == [(20.0,) * 16]


def test_serial_packet_has_16_channel_ordered_angles_and_newline_framing():
    """The wire packet keeps channel position and has one unambiguous terminator."""
    _, _, transport = hardware_modules()

    assert transport.format_angles_packet(tuple(float(value) for value in range(16))) == (
        b"ANGLES,0.000000,1.000000,2.000000,3.000000,4.000000,5.000000,"
        b"6.000000,7.000000,8.000000,9.000000,10.000000,11.000000,12.000000,"
        b"13.000000,14.000000,15.000000\n"
    )


@pytest.mark.parametrize("angles", [tuple(range(15)), tuple(range(17))])
def test_serial_packet_rejects_wrong_servo_count(angles):
    """Malformed packets must not represent fewer or more than 16 channels."""
    _, _, transport = hardware_modules()

    with pytest.raises(ValueError, match="16"):
        transport.format_angles_packet(angles)


def test_serial_transport_sends_packet_through_injected_fake_connection_only():
    """Serial transport is testable without opening a real serial device."""
    _, _, transport = hardware_modules()

    class FakeConnection:
        def __init__(self):
            self.writes = []
            self.flush_calls = 0

        def write(self, data):
            self.writes.append(data)

        def flush(self):
            self.flush_calls += 1

    fake_connection = FakeConnection()
    factory_calls = []

    def fake_factory(port, baudrate):
        factory_calls.append((port, baudrate))
        return fake_connection

    serial_transport = transport.SerialTransport(connection_factory=fake_factory)

    assert factory_calls == []
    serial_transport.send_angles((0.0,) * 16)

    assert factory_calls == [("/dev/ttyACM0", 115200)]
    assert fake_connection.writes == [b"ANGLES," + b"0.000000," * 15 + b"0.000000\n"]
    assert fake_connection.flush_calls == 1
