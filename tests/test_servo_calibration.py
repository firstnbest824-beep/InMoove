"""ServoCalibration의 정규화 값-각도 변환 테스트."""

import importlib

import pytest


def load_servo_calibration():
    """아직 구현되지 않은 모듈도 테스트 실패로 보고할 수 있게 불러옵니다."""
    module = importlib.import_module("inmoove.hardware.servo_calibration")
    return module.ServoCalibration


class TestServoCalibration:
    """정규화된 표정 값을 안전한 서보 각도로 변환합니다."""

    @pytest.mark.parametrize(
        ("value", "expected_angle"),
        [
            (0.0, 20.0),
            (0.5, 45.0),
            (1.0, 70.0),
        ],
    )
    def test_converts_normalized_value_to_angle(
        self, value: float, expected_angle: float
    ):
        """최소·중간·최대 정규화 값이 설정된 각도 범위로 선형 변환됩니다."""
        ServoCalibration = load_servo_calibration()
        jaw = ServoCalibration(min_angle=20.0, max_angle=70.0)

        assert jaw.to_angle(value) == expected_angle

    @pytest.mark.parametrize(
        ("value", "expected_angle"),
        [
            (-0.2, 20.0),
            (1.2, 70.0),
        ],
    )
    def test_clamps_out_of_range_values_to_safe_angle_limits(
        self, value: float, expected_angle: float
    ):
        """범위를 벗어난 입력은 서보가 안전 범위를 넘지 않도록 제한됩니다."""
        ServoCalibration = load_servo_calibration()
        jaw = ServoCalibration(min_angle=20.0, max_angle=70.0)

        assert jaw.to_angle(value) == expected_angle

    @pytest.mark.parametrize(
        ("value", "expected_angle"),
        [
            (0.0, 70.0),
            (0.5, 45.0),
            (1.0, 20.0),
        ],
    )
    def test_reversed_calibration_inverts_input_direction(
        self, value: float, expected_angle: float
    ):
        """reversed 보정은 같은 안전 범위에서 입력 방향만 뒤집습니다."""
        ServoCalibration = load_servo_calibration()
        jaw = ServoCalibration(
            min_angle=20.0,
            max_angle=70.0,
            reversed=True,
        )

        assert jaw.to_angle(value) == expected_angle
