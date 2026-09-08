"""
사람 얼굴 데이터 수집 파이프라인

이 패키지는 다음 흐름을 지원합니다.

실제 영상 또는 Mock 입력
    ↓
Face Data Extractor
    ↓
RawFaceFrame
    ↓
품질 검사
    ↓
JSONL / CSV 데이터셋 저장

중요:
- 이 단계는 실제 학습을 수행하지 않습니다
- 실제 MediaPipe 추론은 구현되어 있으나, 모델 파일/영상이 없으면
  테스트 코드가 깨지지 않도록 lazy import 방식을 사용합니다
- 실질적인 Human → Robot Retargeting은 별도 단계에서 구현합니다
"""

from .models import (
    DatasetMetadata,
    PipelineResult,
    QualityReport,
    RawFaceFrame,
)
from .extractor import FaceDataExtractor
from .mock_extractor import MockFaceDataExtractor
from .mediapipe_extractor import MediaPipeFaceDataExtractor
from .quality import FaceDataQualityValidator
from .retargeting import FaceRetargeter, NotImplementedRetargeter
from .writer import DatasetWriter
from .pipeline import FaceDataPipeline

__all__ = [
    "DatasetMetadata",
    "FaceDataExtractor",
    "FaceDataPipeline",
    "FaceDataQualityValidator",
    "FaceRetargeter",
    "MediaPipeFaceDataExtractor",
    "MockFaceDataExtractor",
    "NotImplementedRetargeter",
    "PipelineResult",
    "QualityReport",
    "RawFaceFrame",
    "DatasetWriter",
]
