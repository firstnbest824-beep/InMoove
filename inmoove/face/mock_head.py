"""
Mock i2Head 구현

실제 하드웨어(PCA9685, Arduino 등)가 없어도
상위 AI/표정 생성 소프트웨어를 개발할 수 있도록 하는
더미 헤드 구현입니다.

특징:
- HeadInterface를 완전히 구현합니다
- 받은 표정 상태를 메모리에 저장합니다
- 호출 로그를 기록합니다
- 하드웨어 통신 코드는 전혀 포함하지 않습니다

향후 RealHead 구현 시:
- 이 클래스와 동일한 인터페이스를 따릅니다
- 다만 set_expression 내에서 실제 서보 제어 코드를 추가합니다
- 상위 코드는 변경되지 않습니다
"""

from typing import Optional, List
from datetime import datetime
from .interface import HeadInterface
from .expression import FaceExpression


class MockHead(HeadInterface):
    """
    테스트 및 개발용 Mock i2Head 구현
    
    이 클래스는 다음 기능을 제공합니다:
    - 표정 상태 저장 및 조회
    - 호출 이력 기록
    - 명확한 CLI 출력
    
    사용 예제:
        >>> head = MockHead()
        >>> smile = FaceExpression(eye_left_lr=0.6, eye_left_ud=0.5, ...)
        >>> head.set_expression(smile)
        >>> print(head.current_expression)
        
    참고:
    - 이 클래스에는 AI 추론이나 실제 하드웨어 코드가 없습니다
    - 순수하게 상태 관리와 로깅만 담당합니다
    """
    
    def __init__(self, verbose: bool = True):
        """
        MockHead를 초기화합니다.
        
        Args:
            verbose (bool): True이면 각 동작마다 상세 정보를 출력합니다
        """
        self.verbose = verbose
        self._current_expression: Optional[FaceExpression] = None
        self._expression_history: List[dict] = []
        
        if self.verbose:
            print("[MockHead] 초기화됨")
    
    def set_expression(self, expression: FaceExpression) -> None:
        """
        로봇의 얼굴 표정을 설정합니다.
        
        Args:
            expression (FaceExpression): 설정할 표정
        
        Raises:
            ValueError: expression이 유효하지 않을 때 (FaceExpression.__post_init__에서)
        
        기능:
        - 받은 표정을 현재 상태로 저장
        - 호출 이력에 기록
        - verbose=True면 상세 정보 출력
        """
        # FaceExpression의 __post_init__이 이미 유효성을 검증했으므로
        # 여기서는 추가 검증이 필요 없습니다
        
        self._current_expression = expression
        
        # 호출 이력 기록
        self._expression_history.append({
            "timestamp": datetime.now().isoformat(),
            "expression": expression,
        })
        
        if self.verbose:
            self._print_expression(expression)
    
    def get_current_expression(self) -> Optional[FaceExpression]:
        """
        현재 설정된 표정을 반환합니다.
        
        Returns:
            FaceExpression: 현재 표정 상태, 아직 설정되지 않았으면 None
        """
        return self._current_expression
    
    def get_expression_history(self) -> List[dict]:
        """
        지금까지의 모든 표정 변화 이력을 반환합니다.
        
        Returns:
            List[dict]: 각 항목이 {timestamp, expression}을 포함하는 리스트
        """
        return self._expression_history.copy()
    
    def rest(self) -> None:
        """
        로봇을 휴지 상태로 만듭니다.
        
        중립 표정으로 설정합니다.
        """
        if self.verbose:
            print("[MockHead] rest 동작 시작...")
        
        self.set_expression(FaceExpression.neutral())
        
        if self.verbose:
            print("[MockHead] rest 동작 완료")
    
    def clear_history(self) -> None:
        """
        표정 호출 이력을 초기화합니다.
        
        테스트 시 이전 호출의 영향을 제거할 때 유용합니다.
        """
        self._expression_history.clear()
        
        if self.verbose:
            print("[MockHead] 이력이 초기화되었습니다")
    
    # ========== Private Methods ==========
    
    def _print_expression(self, expression: FaceExpression) -> None:
        """
        표정 상태를 보기 좋게 터미널에 출력합니다.
        
        Args:
            expression (FaceExpression): 출력할 표정
        """
        print("\n" + "="*50)
        print("[MockHead] 표정 업데이트됨")
        print("="*50)
        
        fields = FaceExpression.get_field_names()
        values = expression.get_all_values()
        
        for field_name, value in zip(fields, values):
            # 필드 이름을 보기 좋게 포맷팅
            display_name = field_name.replace("_", " ").title()
            # 값을 백분율로 표시
            percentage = value * 100
            
            # 시각적 바 표시
            bar_length = 20
            filled = int(bar_length * value)
            bar = "■" * filled + "□" * (bar_length - filled)
            
            print(f"  {display_name:20} : {value:5.2f} [{bar}] {percentage:6.1f}%")
        
        print("="*50 + "\n")
