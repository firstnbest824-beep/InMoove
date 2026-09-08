"""
인간 얼굴 동작 데이터 파이프라인에서 사용되는 핵심 데이터 모델입니다.

이 모듈은 실제 학습을 위한 데이터 표현만 담당합니다.
- RawFaceFrame: 한 프레임의 얼굴 움직임
- QualityReport: 데이터 품질 평가 결과
- DatasetMetadata: 데이터셋 메타데이터
- PipelineResult: 파이프라인 처리 결과

핵심 원칙:
- 실제 로봇 각도 값이나 calibration 값은 포함하지 않습니다
- 사람의 얼굴 움직임을 raw하게 보존하는 것이 목적입니다
- 잘못된 값은 조용히 clamp하지 않고 예외를 발생시킵니다
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class RawFaceFrame:
    """
    하나의 프레임에서 추출된 사람 얼굴 움직임 정보

    이 데이터 구조는 실제 MediaPipe 결과를 흡수하기 위해 설계되었습니다.
    나중에 Retargeting 단계에서 이 정보를 i2Head FaceExpression으로 변환할 수 있습니다.

    속성:
    - frame_index: 프레임 번호
    - timestamp_ms: 프레임의 절대 시간 (밀리초)
    - face_detected: 얼굴 검출 여부
    - blendshapes: MediaPipe blendshape 이름과 값의 맵
    - landmarks: optional한 좌표 정보
    - source_id: 원본 비디오/소스 식별자
    """

    frame_index: int
    timestamp_ms: int
    face_detected: bool
    blendshapes: Dict[str, float] = field(default_factory=dict)
    landmarks: Optional[Dict[str, Any]] = None
    source_id: Optional[str] = None

    def __post_init__(self) -> None:
        """
        RawFaceFrame의 입력값을 검증합니다.

        규칙:
        - frame_index는 0 이상이어야 함
        - timestamp_ms는 0 이상이어야 함
        - blendshape 이름은 문자열이어야 함
        - 값은 숫자여야 하며 NaN, inf를 허용하지 않음
        - 값은 0.0 ~ 1.0 범위여야 함
        - face_detected가 False면 blendshapes는 비어 있어도 허용됨
        """
        if self.frame_index < 0:
            raise ValueError(f"frame_index는 0 이상이어야 합니다. 받은 값: {self.frame_index}")
        if self.timestamp_ms < 0:
            raise ValueError(f"timestamp_ms는 0 이상이어야 합니다. 받은 값: {self.timestamp_ms}")
        if not isinstance(self.blendshapes, dict):
            raise TypeError("blendshapes는 dict여야 합니다")

        for name, value in self.blendshapes.items():
            if not isinstance(name, str):
                raise TypeError(f"blendshape 이름은 문자열이어야 합니다. 받은 값: {type(name)}")
            if not isinstance(value, (int, float)):
                raise TypeError(f"blendshape 값은 숫자여야 합니다. key={name}, type={type(value)}")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"blendshape 값은 유한 숫자여야 합니다. key={name}, value={value}")
            if not 0.0 <= numeric_value <= 1.0:
                raise ValueError(
                    f"blendshape 값이 허용 범위를 벗어났습니다. key={name}, value={numeric_value}, "
                    "기대 범위: 0.0 ~ 1.0"
                )

        if self.landmarks is not None and not isinstance(self.landmarks, dict):
            raise TypeError("landmarks는 dict 또는 None이어야 합니다")

    def to_dict(self) -> Dict[str, Any]:
        """
        JSON 직렬화에 적합한 형태로 변환합니다.
        """
        payload: Dict[str, Any] = {
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "face_detected": self.face_detected,
            "blendshapes": dict(self.blendshapes),
        }
        if self.landmarks is not None:
            payload["landmarks"] = self.landmarks
        if self.source_id is not None:
            payload["source_id"] = self.source_id
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RawFaceFrame":
        """
        직렬화된 dict에서 RawFaceFrame을 복원합니다.
        """
        return cls(
            frame_index=int(payload["frame_index"]),
            timestamp_ms=int(payload["timestamp_ms"]),
            face_detected=bool(payload.get("face_detected", False)),
            blendshapes=dict(payload.get("blendshapes", {})),
            landmarks=payload.get("landmarks"),
            source_id=payload.get("source_id"),
        )


@dataclass
class DatasetMetadata:
    """
    데이터셋 파일이 생성될 때 함께 남기는 메타데이터입니다.

    실제 영상 파일이 없더라도 mock 데이터에 대해 metadata를 남길 수 있습니다.
    나중에 연구용 데이터셋을 분류할 때 유용합니다.
    """

    source: str
    fps: Optional[float] = None
    total_frames: int = 0
    processed_frames: int = 0
    detected_frames: int = 0
    model: str = "mock"
    landmarks_saved: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "detected_frames": self.detected_frames,
            "model": self.model,
            "landmarks_saved": self.landmarks_saved,
            "created_at": self.created_at,
        }


@dataclass
class QualityReport:
    """
    한 데이터셋에 대해 품질을 요약하는 결과입니다.

    수치 기준은 초기 engineering default이며,
    실제 데이터 분포를 보고 조정할 수 있습니다.
    """

    total_frames: int
    detected_frames: int
    detection_rate: float
    invalid_timestamp_count: int = 0
    invalid_value_count: int = 0
    longest_missing_run: int = 0
    grade: str = "C"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "detected_frames": self.detected_frames,
            "detection_rate": self.detection_rate,
            "invalid_timestamp_count": self.invalid_timestamp_count,
            "invalid_value_count": self.invalid_value_count,
            "longest_missing_run": self.longest_missing_run,
            "grade": self.grade,
        }


@dataclass
class PipelineResult:
    """
    FaceDataPipeline의 최종 처리 결과입니다.

    저장된 데이터셋 경로, 메타데이터, 품질 보고서가 함께 포함됩니다.
    """

    frames: List[RawFaceFrame]
    report: QualityReport
    dataset_path: str
    metadata: DatasetMetadata
