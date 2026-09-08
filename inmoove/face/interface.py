"""
로봇 헤드 제어 추상 인터페이스

설계 원칙:
- 상위 AI 코드는 실제 하드웨어 구현을 알 필요가 없습니다
- MockHead와 RealHead는 동일한 인터페이스를 구현합니다
- FaceExpression 객체만을 사용하여 통신합니다

데이터 흐름:
    LLM / MediaPipe / Retargeting
            ↓
      FaceExpression
            ↓
       HeadInterface
         ↓        ↓
     MockHead   RealHead
                 ↓
           MyRobotLab
                 ↓
          Arduino/PCA9685
                 ↓
              Servo
"""

from abc import ABC, abstractmethod
from .expression import FaceExpression


class HeadInterface(ABC):
    """
    로봇 헤드의 얼굴 제어를 위한 추상 인터페이스
    
    이 클래스를 상속받는 구현체들:
    - MockHead: 실제 하드웨어 없이 동작하는 Mock 구현
    - RealHead: 실제 MyRobotLab/PCA9685 서보와 통신하는 구현
    
    상위 애플리케이션은 이 인터페이스의 구체적 구현이 무엇인지
    알 필요가 없습니다.
    """
    
    @abstractmethod
    def set_expression(self, expression: FaceExpression) -> None:
        """
        로봇의 얼굴 표정을 설정합니다.
        
        Args:
            expression (FaceExpression): 설정할 얼굴 표정 상태
                                        모든 값은 0.0~1.0 범위여야 합니다
        
        Raises:
            ValueError: expression의 값이 유효 범위를 벗어났을 때
        
        예제:
            >>> head = MockHead()  # 또는 RealHead()
            >>> expr = FaceExpression(...)
            >>> head.set_expression(expr)
        """
        pass
    
    def rest(self) -> None:
        """
        로봇을 휴지 상태로 만듭니다.
        
        기본 구현은 중립 표정을 설정합니다.
        구현체에서 필요시 오버라이드하여 추가 동작을 정의할 수 있습니다.
        
        예: 서보 전원 끄기, 부드러운 전환 등
        """
        self.set_expression(FaceExpression.neutral())
