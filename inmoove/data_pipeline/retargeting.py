"""
Human → Robot retargeting 준비 인터페이스

중요:
- 여기서는 실제 변환 알고리즘을 구현하지 않습니다
- 이 모듈은 향후 MediaPipe blendshape를 i2Head FaceExpression으로 연결하기 위한
  계층 분리 인터페이스만 제공합니다
- 현재는 "입력 RawFaceFrame → 출력 FaceExpression" 구조를 준비하는 단계입니다
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from inmoove.face import FaceExpression

from .models import RawFaceFrame


class FaceRetargeter(ABC):
    """
    사람 얼굴 motion을 i2Head FaceExpression으로 변환하는 추상 클래스입니다.

    실제 로직은 이후 작업에서 구현합니다.
    현재는 구조만 남겨 두고, 상위 pipeline이 이 인터페이스를 사용할 수 있게 합니다.
    """

    @abstractmethod
    def retarget(self, frame: RawFaceFrame) -> FaceExpression:
        """
        RawFaceFrame을 FaceExpression으로 변환합니다.

        이 구현은 현재 단계에서 동작하지 않으며,
        향후 실제 mapping logic을 넣는 위치입니다.
        """
        raise NotImplementedError


class NotImplementedRetargeter(FaceRetargeter):
    """
    향후 실제 Retargeter가 구현될 때까지 남겨둘 기본 stub입니다.

    현재는 사용하면 명확한 예외를 발생시켜서
    실수로 학습 데이터처럼 잘못 사용되는 일을 막습니다.
    """

    def retarget(self, frame: RawFaceFrame) -> FaceExpression:
        raise NotImplementedError(
            "Human → i2Head retargeting은 아직 구현되지 않았습니다. "
            "이 단계에서는 RawFaceFrame과 dataset pipeline만 검증합니다."
        )
