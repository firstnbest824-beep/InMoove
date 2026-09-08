"""
MockHead 기능 테스트

테스트 항목:
1. MockHead가 받은 표정을 정상 저장하는지
2. rest 동작 후 상태 확인
3. 서로 다른 FaceExpression을 연속 입력했을 때 마지막 상태 갱신
4. 호출 이력 기록
"""

import pytest
from inmoove.face import FaceExpression, MockHead


class TestMockHeadBasics:
    """MockHead 기본 기능 테스트"""
    
    def test_mockhead_creation(self):
        """
        MockHead를 생성할 수 있습니다.
        """
        head = MockHead(verbose=False)
        assert head.get_current_expression() is None
    
    def test_set_expression_stores_state(self):
        """
        set_expression()으로 받은 표정을 정상적으로 저장합니다.
        """
        head = MockHead(verbose=False)
        expr = FaceExpression.neutral()
        
        head.set_expression(expr)
        
        current = head.get_current_expression()
        assert current is not None
        assert current.eye_left_lr == expr.eye_left_lr
        assert current.jaw == expr.jaw
    
    def test_custom_expression(self):
        """
        맞춤 표정을 설정하고 조회합니다.
        """
        head = MockHead(verbose=False)
        
        expr = FaceExpression(
            eye_left_lr=0.7,
            eye_left_ud=0.6,
            eye_right_lr=0.7,
            eye_right_ud=0.6,
            eyelid_left_upper=0.9,
            eyelid_left_lower=0.1,
            eyelid_right_upper=0.9,
            eyelid_right_lower=0.1,
            eyebrow_left=0.8,
            eyebrow_right=0.8,
            cheek_left=0.7,
            cheek_right=0.7,
            forehead_left=0.3,
            forehead_right=0.3,
            upper_lip=0.6,
            jaw=0.2,
        )
        
        head.set_expression(expr)
        current = head.get_current_expression()
        
        assert current.eye_left_lr == 0.7
        assert current.jaw == 0.2
        assert current.eyebrow_left == 0.8


class TestMockHeadRest:
    """MockHead rest 기능 테스트"""
    
    def test_rest_sets_neutral(self):
        """
        rest()를 호출하면 중립 표정으로 설정됩니다.
        """
        head = MockHead(verbose=False)
        
        # 먼저 다른 표정을 설정
        custom = FaceExpression(
            eye_left_lr=0.9,
            eye_left_ud=0.9,
            eye_right_lr=0.9,
            eye_right_ud=0.9,
            eyelid_left_upper=1.0,
            eyelid_left_lower=0.0,
            eyelid_right_upper=1.0,
            eyelid_right_lower=0.0,
            eyebrow_left=1.0,
            eyebrow_right=1.0,
            cheek_left=1.0,
            cheek_right=1.0,
            forehead_left=0.0,
            forehead_right=0.0,
            upper_lip=0.8,
            jaw=0.8,
        )
        head.set_expression(custom)
        
        # rest 호출
        head.rest()
        
        # 중립 상태 확인
        current = head.get_current_expression()
        assert current.eye_left_lr == 0.5
        assert current.jaw == 0.5


class TestMockHeadSequentialUpdates:
    """MockHead 연속 표정 변환 테스트"""
    
    def test_multiple_expressions_update_correctly(self):
        """
        서로 다른 FaceExpression을 연속으로 입력할 때 
        마지막 상태가 정상 갱신됩니다.
        """
        head = MockHead(verbose=False)
        
        # 첫 번째 표정
        expr1 = FaceExpression(
            eye_left_lr=0.3,
            eye_left_ud=0.3,
            eye_right_lr=0.3,
            eye_right_ud=0.3,
            eyelid_left_upper=0.3,
            eyelid_left_lower=0.3,
            eyelid_right_upper=0.3,
            eyelid_right_lower=0.3,
            eyebrow_left=0.3,
            eyebrow_right=0.3,
            cheek_left=0.3,
            cheek_right=0.3,
            forehead_left=0.3,
            forehead_right=0.3,
            upper_lip=0.3,
            jaw=0.3,
        )
        head.set_expression(expr1)
        current = head.get_current_expression()
        assert current.eye_left_lr == 0.3
        
        # 두 번째 표정
        expr2 = FaceExpression(
            eye_left_lr=0.7,
            eye_left_ud=0.7,
            eye_right_lr=0.7,
            eye_right_ud=0.7,
            eyelid_left_upper=0.7,
            eyelid_left_lower=0.7,
            eyelid_right_upper=0.7,
            eyelid_right_lower=0.7,
            eyebrow_left=0.7,
            eyebrow_right=0.7,
            cheek_left=0.7,
            cheek_right=0.7,
            forehead_left=0.7,
            forehead_right=0.7,
            upper_lip=0.7,
            jaw=0.7,
        )
        head.set_expression(expr2)
        current = head.get_current_expression()
        assert current.eye_left_lr == 0.7
        assert current.jaw == 0.7


class TestMockHeadHistory:
    """MockHead 호출 이력 테스트"""
    
    def test_expression_history_recorded(self):
        """
        모든 set_expression 호출이 이력에 기록됩니다.
        """
        head = MockHead(verbose=False)
        
        expr1 = FaceExpression.neutral()
        expr2 = FaceExpression(
            eye_left_lr=0.8,
            eye_left_ud=0.5,
            eye_right_lr=0.8,
            eye_right_ud=0.5,
            eyelid_left_upper=0.9,
            eyelid_left_lower=0.1,
            eyelid_right_upper=0.9,
            eyelid_right_lower=0.1,
            eyebrow_left=0.7,
            eyebrow_right=0.7,
            cheek_left=0.6,
            cheek_right=0.6,
            forehead_left=0.4,
            forehead_right=0.4,
            upper_lip=0.5,
            jaw=0.3,
        )
        
        head.set_expression(expr1)
        head.set_expression(expr2)
        
        history = head.get_expression_history()
        assert len(history) == 2
        assert history[0]["expression"].eye_left_lr == 0.5
        assert history[1]["expression"].eye_left_lr == 0.8
    
    def test_history_includes_timestamp(self):
        """
        이력의 각 항목이 타임스탬프를 포함합니다.
        """
        head = MockHead(verbose=False)
        expr = FaceExpression.neutral()
        head.set_expression(expr)
        
        history = head.get_expression_history()
        assert len(history) == 1
        assert "timestamp" in history[0]
        assert "expression" in history[0]
    
    def test_clear_history(self):
        """
        clear_history()로 이력을 초기화할 수 있습니다.
        """
        head = MockHead(verbose=False)
        
        head.set_expression(FaceExpression.neutral())
        head.set_expression(FaceExpression.neutral())
        
        assert len(head.get_expression_history()) == 2
        
        head.clear_history()
        
        assert len(head.get_expression_history()) == 0
