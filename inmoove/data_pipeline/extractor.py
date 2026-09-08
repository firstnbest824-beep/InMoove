"""
얼굴 데이터 추출기 추상 인터페이스

핵심 목적:
- 실제 MediaPipe 추출기와 Mock 추출기가 같은 인터페이스를 공유하게 함
- 상위 DataPipeline은 구현체를 몰라도 동작할 수 있게 함
- 실제 영상이 없더라도 Mock extractor를 통해 완전한 파이프라인 검증이 가능하게 함
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from .models import RawFaceFrame


class FaceDataExtractor(ABC):
    """
    입력 소스로부터 RawFaceFrame 시퀀스를 추출하는 공통 인터페이스입니다.

    구현체는 다음 둘 중 하나가 됩니다.
    - MockFaceDataExtractor: 실제 영상 없이 테스트용 생성기
    - MediaPipeFaceDataExtractor: 실제 MediaPipe Face Landmarker 추출기

    상위 코드에서는 구현체를 구체적으로 알 필요가 없습니다.
    """

    @abstractmethod
    def extract(self, **kwargs: Any) -> List[RawFaceFrame]:
        """
        얼굴 데이터 시퀀스를 생성하거나 읽어옵니다.

        Args:
            **kwargs: 구현체마다 다른 입력 형식

        Returns:
            List[RawFaceFrame]: 추출된 프레임 리스트
        """
        raise NotImplementedError
