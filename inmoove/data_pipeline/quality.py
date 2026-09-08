"""
데이터 품질 검증기

이 모듈은 추출된 frame 시퀀스를 검사하여 다음을 판단합니다.
- 얼굴이 충분히 검출되었는지
- timestamp가 정상적인지
- blendshape 값이 유효한지
- 연속적으로 얼굴이 빠진 구간이 얼마나 되는지

해당 결과는 datasets의 품질을 판단하는 engineering default로 사용됩니다.
"""

from __future__ import annotations

import math
from typing import Sequence

from .models import QualityReport, RawFaceFrame


class FaceDataQualityValidator:
    """
    얼굴 데이터 품질을 검증하는 클래스입니다.

    초기 단계에서는 strict하게 검사합니다.
    잘못된 값은 자동으로 보정하지 않고 보고서에 남깁니다.
    """

    def __init__(self, minimum_detection_rate: float = 0.95) -> None:
        self.minimum_detection_rate = minimum_detection_rate

    def validate(self, frames: Sequence[RawFaceFrame]) -> QualityReport:
        """
        frame 시퀀스를 검사하여 품질 보고서를 생성합니다.
        """
        total_frames = len(frames)
        if total_frames == 0:
            return QualityReport(
                total_frames=0,
                detected_frames=0,
                detection_rate=0.0,
                invalid_timestamp_count=0,
                invalid_value_count=0,
                longest_missing_run=0,
                grade="C",
            )

        detected_frames = sum(1 for frame in frames if frame.face_detected)
        detection_rate = detected_frames / total_frames

        invalid_timestamp_count = 0
        invalid_value_count = 0
        longest_missing_run = 0
        current_missing_run = 0
        previous_timestamp: int | None = None

        for frame in frames:
            if frame.timestamp_ms < 0:
                invalid_timestamp_count += 1
            if previous_timestamp is not None and frame.timestamp_ms < previous_timestamp:
                invalid_timestamp_count += 1
            previous_timestamp = frame.timestamp_ms

            if frame.face_detected:
                current_missing_run = 0
            else:
                current_missing_run += 1
                if current_missing_run > longest_missing_run:
                    longest_missing_run = current_missing_run

            for value in frame.blendshapes.values():
                numeric = float(value)
                if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                    invalid_value_count += 1

        grade = self._grade_for_rate(detection_rate)

        return QualityReport(
            total_frames=total_frames,
            detected_frames=detected_frames,
            detection_rate=detection_rate,
            invalid_timestamp_count=invalid_timestamp_count,
            invalid_value_count=invalid_value_count,
            longest_missing_run=longest_missing_run,
            grade=grade,
        )

    def _grade_for_rate(self, detection_rate: float) -> str:
        """
        초기 engineering 기준.

        현재 값은 임시 default이며 향후 데이터 분포를 기반으로 조정될 수 있습니다.
        """
        if detection_rate >= 0.95:
            return "A"
        if detection_rate >= 0.80:
            return "B"
        return "C"
