"""정규화된 액추에이터 값을 실제 서보 각도로 변환합니다."""

from dataclasses import dataclass


@dataclass
class ServoCalibration:
    """서보 하나의 안전한 가동 범위와 방향을 정의합니다."""

    min_angle: float
    max_angle: float
    reversed: bool = False

    def to_angle(self, value: float) -> float:
        """0.0~1.0 값을 설정된 실제 각도 범위로 변환합니다."""
        value = max(0.0, min(1.0, value))

        if self.reversed:
            value = 1.0 - value

        return self.min_angle + value * (self.max_angle - self.min_angle)
