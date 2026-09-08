"""
데이터 수집 파이프라인의 상위 조합기

실제 데이터 흐름:
Mock 또는 MediaPipe extractor
    ↓
RawFaceFrame 리스트
    ↓
QualityValidator
    ↓
DatasetWriter
    ↓
PipelineResult

중요:
- 상위 코드는 extractor가 Mock인지 MediaPipe인지 알 필요가 없습니다
- 입력 소스만 바꿔도 동일한 pipeline이 동작해야 합니다
"""

from __future__ import annotations

from typing import Any, Optional

from .extractor import FaceDataExtractor
from .models import DatasetMetadata, PipelineResult, QualityReport
from .quality import FaceDataQualityValidator
from .writer import DatasetWriter


class FaceDataPipeline:
    """
    얼굴 데이터 파이프라인의 상위 조합입니다.

    extractor가 실제 영상인지 mock인지 몰라도 동일한 방식으로 처리합니다.
    """

    def __init__(
        self,
        extractor: FaceDataExtractor,
        writer: DatasetWriter,
        validator: Optional[FaceDataQualityValidator] = None,
    ) -> None:
        self.extractor = extractor
        self.writer = writer
        self.validator = validator or FaceDataQualityValidator()

    def run(
        self,
        output_path: str = "output/mock_face_dataset.jsonl",
        dataset_name: Optional[str] = None,
        **kwargs: Any,
    ) -> PipelineResult:
        """
        전체 데이터를 추출하고, 검증하고, 저장하는 메인 함수입니다.

        Args:
            output_path: 출력 JSONL 파일 경로. 예: output/mock_face_dataset.jsonl
            dataset_name: filename으로 사용할 이름(없으면 output_path에서 추정)
            **kwargs: extractor.extract()에 전달할 인자

        Returns:
            PipelineResult: 품질과 파일 정보가 포함된 결과
        """
        frames = self.extractor.extract(**kwargs)
        report = self.validator.validate(frames)

        if dataset_name is None:
            dataset_name = output_path.split("/")[-1] if output_path else "dataset.jsonl"

        metadata = DatasetMetadata(
            source=kwargs.get("source_id", "unknown-source"),
            fps=kwargs.get("fps"),
            total_frames=len(frames),
            processed_frames=len(frames),
            detected_frames=report.detected_frames,
            model=getattr(self.extractor, "__class__", "unknown").__name__,
            landmarks_saved=False,
        )

        dataset_path = self.writer.write_jsonl(frames, dataset_name, metadata=metadata)
        return PipelineResult(
            frames=frames,
            report=report,
            dataset_path=str(dataset_path),
            metadata=metadata,
        )
