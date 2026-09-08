"""
Mock extractor

실제 영상이 없더라도 전체 데이터 파이프라인을 테스트할 수 있도록
일정한 규칙을 가진 fake face data를 생성합니다.

이 구현은 다음 의미를 가집니다.
- 실제 MediaPipe 기반 데이터의 형태를 흉내냄
- deterministic하게 동작하여 테스트 안정성 확보
- 나중에 실제 video + MediaPipe extractor로 교체 가능
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .extractor import FaceDataExtractor
from .models import RawFaceFrame


class MockFaceDataExtractor(FaceDataExtractor):
    """
    실험용 Mock 얼굴 데이터 추출기

    입력이 없는 상태에서도 데이터 파이프라인 전체가 동작하는지 검증할 수 있습니다.
    값은 과장 없이 자연스럽게 변화하는 형태로 구성합니다.
    """

    def __init__(
        self,
        frame_count: int = 120,
        fps: float = 30.0,
        source_id: Optional[str] = "mock-face-sequence",
        missing_frame_indices: Optional[set[int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.frame_count = frame_count
        self.fps = fps
        self.source_id = source_id
        self.missing_frame_indices = missing_frame_indices or set()
        self.seed = seed

    def extract(self, **kwargs: Any) -> List[RawFaceFrame]:
        """
        Mock 시퀀스를 생성합니다.

        Args:
            frame_count: 생성할 프레임 총 개수
            fps: 초당 프레임 수
            start_timestamp_ms: 시작 타임스탬프
            source_id: 원본 ID

        Returns:
            List[RawFaceFrame]: 정해진 규칙에 따라 생성된 프레임들
        """
        frame_count = int(kwargs.get("frame_count", self.frame_count))
        fps = float(kwargs.get("fps", self.fps))
        start_timestamp_ms = int(kwargs.get("start_timestamp_ms", 0))
        source_id = kwargs.get("source_id", self.source_id)

        if frame_count < 0:
            raise ValueError("frame_count는 0 이상이어야 합니다")
        if fps <= 0:
            raise ValueError("fps는 0보다 커야 합니다")

        frames: List[RawFaceFrame] = []
        for index in range(frame_count):
            timestamp_ms = start_timestamp_ms + int((index * 1000.0) / fps)
            face_detected = index not in self.missing_frame_indices
            blendshapes = self._build_blendshapes(index, face_detected)
            frames.append(
                RawFaceFrame(
                    frame_index=index,
                    timestamp_ms=timestamp_ms,
                    face_detected=face_detected,
                    blendshapes=blendshapes,
                    source_id=source_id,
                )
            )
        return frames

    def _build_blendshapes(self, index: int, face_detected: bool) -> Dict[str, float]:
        """
        deterministic한 얼굴 움직임을 생성합니다.

        이 규칙은 사람의 자연스러운 표정 변화에 가까운 단순 신호를 만듭니다.
        """
        if not face_detected:
            return {}

        phase = index / 9.0
        smile = 0.5 + 0.3 * math.sin(phase + 0.6)
        brow = 0.5 + 0.2 * math.sin(phase * 1.3)
        jaw = 0.2 + 0.3 * math.sin(phase * 0.8)
        blink = 0.1 + 0.22 * max(0.0, math.sin(phase * 2.2))

        # 좌우 대칭을 유지하되, 약간의 차이를 반영함
        left_smile = clamp(smile + 0.06 * math.sin(phase + 0.2), 0.0, 1.0)
        right_smile = clamp(smile + 0.04 * math.sin(phase + 0.9), 0.0, 1.0)
        left_brow = clamp(brow + 0.05 * math.cos(phase), 0.0, 1.0)
        right_brow = clamp(brow + 0.07 * math.sin(phase + 0.8), 0.0, 1.0)

        return {
            "jaw_open": clamp(jaw, 0.0, 1.0),
            "mouth_smile_left": clamp(left_smile, 0.0, 1.0),
            "mouth_smile_right": clamp(right_smile, 0.0, 1.0),
            "brow_outer_up_left": clamp(left_brow, 0.0, 1.0),
            "brow_outer_up_right": clamp(right_brow, 0.0, 1.0),
            "eye_blink_left": clamp(blink, 0.0, 1.0),
            "eye_blink_right": clamp(0.9 * blink, 0.0, 1.0),
            "mouth_frown_left": clamp(0.2 + 0.15 * math.sin(phase + 2.4), 0.0, 1.0),
            "mouth_frown_right": clamp(0.18 + 0.14 * math.sin(phase + 2.0), 0.0, 1.0),
        }


def clamp(value: float, minimum: float, maximum: float) -> float:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value
