"""
FaceExpression 유효성 검증 테스트

테스트 항목:
1. 정상적인 FaceExpression 생성
2. 0.0 값 허용
3. 1.0 값 허용
4. 0 미만 값 거부
5. 1 초과 값 거부
6. neutral() 생성
"""

import pytest
from inmoove.face import FaceExpression


class TestFaceExpressionCreation:
    """FaceExpression 생성 테스트"""
    
    def test_valid_expression_creation(self):
        """
        정상적인 FaceExpression을 생성할 수 있습니다.
        """
        expr = FaceExpression(
            eye_left_lr=0.5,
            eye_left_ud=0.5,
            eye_right_lr=0.5,
            eye_right_ud=0.5,
            eyelid_left_upper=0.5,
            eyelid_left_lower=0.5,
            eyelid_right_upper=0.5,
            eyelid_right_lower=0.5,
            eyebrow_left=0.5,
            eyebrow_right=0.5,
            cheek_left=0.5,
            cheek_right=0.5,
            forehead_left=0.5,
            forehead_right=0.5,
            upper_lip=0.5,
            jaw=0.5,
        )
        assert expr.eye_left_lr == 0.5
        assert expr.jaw == 0.5
    
    def test_minimum_value_allowed(self):
        """
        0.0 값(최소값)은 허용됩니다.
        """
        expr = FaceExpression(
            eye_left_lr=0.0,
            eye_left_ud=0.0,
            eye_right_lr=0.0,
            eye_right_ud=0.0,
            eyelid_left_upper=0.0,
            eyelid_left_lower=0.0,
            eyelid_right_upper=0.0,
            eyelid_right_lower=0.0,
            eyebrow_left=0.0,
            eyebrow_right=0.0,
            cheek_left=0.0,
            cheek_right=0.0,
            forehead_left=0.0,
            forehead_right=0.0,
            upper_lip=0.0,
            jaw=0.0,
        )
        assert expr.eye_left_lr == 0.0
        assert expr.jaw == 0.0
    
    def test_maximum_value_allowed(self):
        """
        1.0 값(최대값)은 허용됩니다.
        """
        expr = FaceExpression(
            eye_left_lr=1.0,
            eye_left_ud=1.0,
            eye_right_lr=1.0,
            eye_right_ud=1.0,
            eyelid_left_upper=1.0,
            eyelid_left_lower=1.0,
            eyelid_right_upper=1.0,
            eyelid_right_lower=1.0,
            eyebrow_left=1.0,
            eyebrow_right=1.0,
            cheek_left=1.0,
            cheek_right=1.0,
            forehead_left=1.0,
            forehead_right=1.0,
            upper_lip=1.0,
            jaw=1.0,
        )
        assert expr.eye_left_lr == 1.0
        assert expr.jaw == 1.0
    
    def test_below_minimum_value_rejected(self):
        """
        0.0 미만의 값은 거부됩니다.
        """
        with pytest.raises(ValueError) as excinfo:
            FaceExpression(
                eye_left_lr=-0.1,
                eye_left_ud=0.5,
                eye_right_lr=0.5,
                eye_right_ud=0.5,
                eyelid_left_upper=0.5,
                eyelid_left_lower=0.5,
                eyelid_right_upper=0.5,
                eyelid_right_lower=0.5,
                eyebrow_left=0.5,
                eyebrow_right=0.5,
                cheek_left=0.5,
                cheek_right=0.5,
                forehead_left=0.5,
                forehead_right=0.5,
                upper_lip=0.5,
                jaw=0.5,
            )
        assert "범위를 벗어났습니다" in str(excinfo.value)
        assert "eye_left_lr" in str(excinfo.value)
    
    def test_above_maximum_value_rejected(self):
        """
        1.0 초과의 값은 거부됩니다.
        """
        with pytest.raises(ValueError) as excinfo:
            FaceExpression(
                eye_left_lr=0.5,
                eye_left_ud=0.5,
                eye_right_lr=0.5,
                eye_right_ud=0.5,
                eyelid_left_upper=0.5,
                eyelid_left_lower=0.5,
                eyelid_right_upper=0.5,
                eyelid_right_lower=0.5,
                eyebrow_left=0.5,
                eyebrow_right=0.5,
                cheek_left=0.5,
                cheek_right=0.5,
                forehead_left=0.5,
                forehead_right=0.5,
                upper_lip=0.5,
                jaw=1.2,  # 1.0 초과
            )
        assert "범위를 벗어났습니다" in str(excinfo.value)
        assert "jaw" in str(excinfo.value)


class TestFaceExpressionNeutral:
    """중립 표정 테스트"""
    
    def test_neutral_creation(self):
        """
        neutral() 메서드로 중립 표정을 생성할 수 있습니다.
        """
        neutral = FaceExpression.neutral()
        assert neutral.eye_left_lr == 0.5
        assert neutral.jaw == 0.5
    
    def test_all_neutral_values_are_middle(self):
        """
        중립 표정은 모든 값이 0.5입니다.
        """
        neutral = FaceExpression.neutral()
        values = neutral.get_all_values()
        assert all(v == 0.5 for v in values)


class TestFaceExpressionConversions:
    """표정 변환 함수 테스트"""
    
    def test_to_dict_conversion(self):
        """
        to_dict()로 표정을 딕셔너리로 변환합니다.
        """
        expr = FaceExpression.neutral()
        d = expr.to_dict()
        
        assert isinstance(d, dict)
        assert "eye_left_lr" in d
        assert "jaw" in d
        assert len(d) == 16
        assert d["eye_left_lr"] == 0.5
    
    def test_get_all_values(self):
        """
        get_all_values()로 모든 값을 튜플로 반환합니다.
        """
        expr = FaceExpression.neutral()
        values = expr.get_all_values()
        
        assert isinstance(values, tuple)
        assert len(values) == 16
        assert all(v == 0.5 for v in values)
    
    def test_get_field_names(self):
        """
        get_field_names()로 필드명을 얻습니다.
        """
        names = FaceExpression.get_field_names()
        
        assert isinstance(names, tuple)
        assert len(names) == 16
        assert "eye_left_lr" in names
        assert "jaw" in names
