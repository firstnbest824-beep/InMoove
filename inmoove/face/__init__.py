"""
얼굴 표정 제어 모듈

주요 클래스:
- FaceExpression: 정규화된 16차원 얼굴 상태
- HeadInterface: 얼굴 제어 추상 인터페이스
- MockHead: 실제 하드웨어 없이 사용 가능한 Mock 구현
"""

from .expression import FaceExpression
from .interface import HeadInterface
from .mock_head import MockHead

__all__ = [
    "FaceExpression",
    "HeadInterface", 
    "MockHead",
]
