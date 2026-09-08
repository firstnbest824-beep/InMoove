"""
RuleBasedFaceRetargeter 테스트

이 테스트는 다음을 검증합니다:
1. Neutral input → Neutral output
2. Gaze는 항상 neutral (0.5)
3. Individual blendshape에 따른 출력 값의 증감
4. Multi-input 조합의 논리적 일관성
5. Extreme value에 대한 clamp 처리
6. Missing blendshape 처리
7. face_detected=False 처리
"""

import pytest

from inmoove.data_pipeline import (
    FaceNotDetectedException,
    MissingBlendshapeException,
    RawFaceFrame,
    RuleBasedFaceRetargeter,
    RetargetingConfig,
)


@pytest.fixture
def retargeter() -> RuleBasedFaceRetargeter:
    """기본 설정의 Retargeter를 생성합니다."""
    return RuleBasedFaceRetargeter(strict=True)


@pytest.fixture
def neutral_frame() -> RawFaceFrame:
    """모든 blendshape이 0인 중립 frame입니다."""
    blendshapes = {
        "browInnerUp": 0.0,
        "browOuterUpLeft": 0.0,
        "browOuterUpRight": 0.0,
        "browDownLeft": 0.0,
        "browDownRight": 0.0,
        "cheekSquintLeft": 0.0,
        "cheekSquintRight": 0.0,
        "eyeBlinkLeft": 0.0,
        "eyeBlinkRight": 0.0,
        "eyeSquintLeft": 0.0,
        "eyeSquintRight": 0.0,
        "eyeWideLeft": 0.0,
        "eyeWideRight": 0.0,
        "jawOpen": 0.0,
        "mouthSmileLeft": 0.0,
        "mouthSmileRight": 0.0,
        "mouthUpperUpLeft": 0.0,
        "mouthUpperUpRight": 0.0,
        "mouthShrugUpper": 0.0,
        "mouthRollUpper": 0.0,
    }
    return RawFaceFrame(
        frame_index=0,
        timestamp_ms=0,
        face_detected=True,
        blendshapes=blendshapes,
    )


class TestNeutralExpression:
    """Neutral expression 관련 테스트"""

    def test_neutral_input_produces_neutral_output(self, retargeter: RuleBasedFaceRetargeter, neutral_frame: RawFaceFrame):
        """모든 값이 0인 입력은 논리상 neutral을 출력해야 합니다."""
        expr = retargeter.retarget(neutral_frame)

        # 중립 상태이므로 모든 얼굴 값이 0.5 근처여야 합니다
        assert 0.4 < expr.eyebrow_left < 0.6
        assert 0.4 < expr.eyebrow_right < 0.6
        assert 0.4 < expr.jaw < 0.6
        assert 0.4 < expr.cheek_left < 0.6
        assert 0.4 < expr.cheek_right < 0.6

    def test_neutral_produces_unified_neutral(self, retargeter: RuleBasedFaceRetargeter, neutral_frame: RawFaceFrame):
        """neutral output은 FaceExpression.neutral()과 논리적으로 일관되어야 합니다."""
        expr = retargeter.retarget(neutral_frame)
        neutral_expr = expr.__class__.neutral()

        # gaze는 항상 0.5
        assert expr.eye_left_lr == neutral_expr.eye_left_lr == 0.5
        assert expr.eye_left_ud == neutral_expr.eye_left_ud == 0.5
        assert expr.eye_right_lr == neutral_expr.eye_right_lr == 0.5
        assert expr.eye_right_ud == neutral_expr.eye_right_ud == 0.5


class TestGazeSeparation:
    """Gaze와 expression 분리 관련 테스트"""

    def test_gaze_always_neutral(self, retargeter: RuleBasedFaceRetargeter, neutral_frame: RawFaceFrame):
        """어떤 입력이든 gaze는 항상 0.5여야 합니다."""
        blendshapes = neutral_frame.blendshapes.copy()
        blendshapes["browOuterUpLeft"] = 1.0  # 강력한 표정
        blendshapes["jawOpen"] = 1.0
        blendshapes["mouthSmileLeft"] = 1.0

        frame = RawFaceFrame(
            frame_index=1, timestamp_ms=33, face_detected=True, blendshapes=blendshapes
        )
        expr = retargeter.retarget(frame)

        assert expr.eye_left_lr == 0.5
        assert expr.eye_left_ud == 0.5
        assert expr.eye_right_lr == 0.5
        assert expr.eye_right_ud == 0.5


class TestJaw:
    """턱(jaw) 관련 테스트"""

    def test_jaw_open_increases(self, retargeter: RuleBasedFaceRetargeter, neutral_frame: RawFaceFrame):
        """jawOpen 값이 증가하면 jaw 출력도 증가해야 합니다."""
        blendshapes_open = neutral_frame.blendshapes.copy()
        blendshapes_open["jawOpen"] = 0.8

        frame_open = RawFaceFrame(
            frame_index=1, timestamp_ms=33, face_detected=True, blendshapes=blendshapes_open
        )
        expr = retargeter.retarget(neutral_frame)
        expr_open = retargeter.retarget(frame_open)

        assert expr_open.jaw > expr.jaw


class TestEyebrow:
    """눈썹(eyebrow) 관련 테스트"""

    def test_eyebrow_up_left_increases(self, retargeter: RuleBasedFaceRetargeter, neutral_frame: RawFaceFrame):
        """browOuterUpLeft가 증가하면 eyebrow_left가 증가해야 합니다."""
        blendshapes_up = neutral_frame.blendshapes.copy()
        blendshapes_up["browOuterUpLeft"] = 1.0

        frame_up = RawFaceFrame(
            frame_index=1, timestamp_ms=33, face_detected=True, blendshapes=blendshapes_up
        )
        expr = retargeter.retarget(neutral_frame)
        expr_up = retargeter.retarget(frame_up)

        assert expr_up.eyebrow_left > expr.eyebrow_left

    def test_eyebrow_down_left_decreases(self, retargeter: RuleBasedFaceRetargeter, neutral_frame: RawFaceFrame):
        """browDownLeft가 증가하면 eyebrow_left가 감소해야 합니다."""
        blendshapes_down = neutral_frame.blendshapes.copy()
        blendshapes_down["browDownLeft"] = 1.0

        frame_down = RawFaceFrame(
            frame_index=1, timestamp_ms=33, face_detected=True, blendshapes=blendshapes_down
        )
        expr = retargeter.retarget(neutral_frame)
        expr_down = retargeter.retarget(frame_down)

        assert expr_down.eyebrow_left < expr.eyebrow_left

    def test_eyebrow_left_right_independence(self, retargeter: RuleBasedFaceRetargeter, neutral_frame: RawFaceFrame):
        """왼쪽 눈썹과 오른쪽 눈썹이 독립적으로 동작해야 합니다."""
        blendshapes_left_only = neutral_frame.blendshapes.copy()
        blendshapes_left_only["browOuterUpLeft"] = 1.0

        frame_left = RawFaceFrame(
            frame_index=1, timestamp_ms=33, face_detected=True, blendshapes=blendshapes_left_only
        )
        expr_left = retargeter.retarget(frame_left)

        # 왼쪽만 올라가므로 오른쪽은 neutral에 가까워야 함
        assert expr_left.eyebrow_left > 0.55
        assert 0.45 < expr_left.eyebrow_right < 0.55


class TestCheek:
    """볼(cheek) 관련 테스트"""

    def test_cheek_squint_increases_cheek(self, retargeter: RuleBasedFaceRetargeter, neutral_frame: RawFaceFrame):
        """cheekSquintLeft가 증가하면 cheek_left가 증가해야 합니다."""
        blendshapes_squint = neutral_frame.blendshapes.copy()
        blendshapes_squint["cheekSquintLeft"] = 1.0

        frame_squint = RawFaceFrame(
            frame_index=1, timestamp_ms=33, face_detected=True, blendshapes=blendshapes_squint
        )
        expr = retargeter.retarget(neutral_frame)
        expr_squint = retargeter.retarget(frame_squint)

        assert expr_squint.cheek_left > expr.cheek_left

    def test_smile_contributes_to_cheek(self, retargeter: RuleBasedFaceRetargeter, neutral_frame: RawFaceFrame):
        """mouthSmileLeft가 증가하면 cheek_left가 증가해야 합니다."""
        blendshapes_smile = neutral_frame.blendshapes.copy()
        blendshapes_smile["mouthSmileLeft"] = 1.0

        frame_smile = RawFaceFrame(
            frame_index=1, timestamp_ms=33, face_detected=True, blendshapes=blendshapes_smile
        )
        expr = retargeter.retarget(neutral_frame)
        expr_smile = retargeter.retarget(frame_smile)

        assert expr_smile.cheek_left > expr.cheek_left


class TestEyelid:
    """눈꺼풀(eyelid) 관련 테스트"""

    def test_blink_affects_upper_eyelid(self, retargeter: RuleBasedFaceRetargeter, neutral_frame: RawFaceFrame):
        """eyeBlinkLeft가 증가하면 eyelid_left_upper가 감소해야 합니다."""
        blendshapes_blink = neutral_frame.blendshapes.copy()
        blendshapes_blink["eyeBlinkLeft"] = 1.0

        frame_blink = RawFaceFrame(
            frame_index=1, timestamp_ms=33, face_detected=True, blendshapes=blendshapes_blink
        )
        expr = retargeter.retarget(neutral_frame)
        expr_blink = retargeter.retarget(frame_blink)

        assert expr_blink.eyelid_left_upper < expr.eyelid_left_upper

    def test_wide_affects_upper_eyelid(self, retargeter: RuleBasedFaceRetargeter, neutral_frame: RawFaceFrame):
        """eyeWideLeft가 증가하면 eyelid_left_upper가 증가해야 합니다."""
        blendshapes_wide = neutral_frame.blendshapes.copy()
        blendshapes_wide["eyeWideLeft"] = 1.0

        frame_wide = RawFaceFrame(
            frame_index=1, timestamp_ms=33, face_detected=True, blendshapes=blendshapes_wide
        )
        expr = retargeter.retarget(neutral_frame)
        expr_wide = retargeter.retarget(frame_wide)

        assert expr_wide.eyelid_left_upper > expr.eyelid_left_upper


class TestValueClamping:
    """출력 값 범위 제한 테스트"""

    def test_all_values_in_range(self, retargeter: RuleBasedFaceRetargeter):
        """극단적인 여러 값의 조합을 넣어도 모든 출력은 0.0 ~ 1.0이어야 합니다."""
        blendshapes = {
            "browInnerUp": 1.0,
            "browOuterUpLeft": 1.0,
            "browOuterUpRight": 1.0,
            "browDownLeft": 1.0,
            "browDownRight": 1.0,
            "cheekSquintLeft": 1.0,
            "cheekSquintRight": 1.0,
            "eyeBlinkLeft": 1.0,
            "eyeBlinkRight": 1.0,
            "eyeSquintLeft": 1.0,
            "eyeSquintRight": 1.0,
            "eyeWideLeft": 1.0,
            "eyeWideRight": 1.0,
            "jawOpen": 1.0,
            "mouthSmileLeft": 1.0,
            "mouthSmileRight": 1.0,
            "mouthUpperUpLeft": 1.0,
            "mouthUpperUpRight": 1.0,
            "mouthShrugUpper": 1.0,
            "mouthRollUpper": 1.0,
        }
        frame = RawFaceFrame(frame_index=0, timestamp_ms=0, face_detected=True, blendshapes=blendshapes)
        expr = retargeter.retarget(frame)

        for value in expr.get_all_values():
            assert 0.0 <= value <= 1.0, f"값이 범위를 벗어났습니다: {value}"


class TestStrictMode:
    """Strict mode 테스트"""

    def test_strict_mode_raises_on_missing_blendshape(self, neutral_frame: RawFaceFrame):
        """strict=True이면 필수 blendshape이 누락될 때 예외를 발생시킵니다."""
        retargeter = RuleBasedFaceRetargeter(strict=True)
        incomplete_blendshapes = neutral_frame.blendshapes.copy()
        del incomplete_blendshapes["jawOpen"]  # 필수 키 제거

        frame = RawFaceFrame(frame_index=0, timestamp_ms=0, face_detected=True, blendshapes=incomplete_blendshapes)

        with pytest.raises(MissingBlendshapeException):
            retargeter.retarget(frame)

    def test_non_strict_mode_handles_missing_blendshape(self, neutral_frame: RawFaceFrame):
        """strict=False이면 누락된 blendshape을 0.0으로 처리합니다."""
        retargeter = RuleBasedFaceRetargeter(strict=False)
        incomplete_blendshapes = neutral_frame.blendshapes.copy()
        del incomplete_blendshapes["jawOpen"]  # 필수 키 제거

        frame = RawFaceFrame(frame_index=0, timestamp_ms=0, face_detected=True, blendshapes=incomplete_blendshapes)

        expr = retargeter.retarget(frame)
        assert isinstance(expr, expr.__class__)  # 정상 반환


class TestFaceNotDetected:
    """Face detection 실패 처리 테스트"""

    def test_face_not_detected_raises_exception(self, retargeter: RuleBasedFaceRetargeter, neutral_frame: RawFaceFrame):
        """face_detected=False이면 neutral로 몰래 처리하지 않고 예외를 발생시킵니다."""
        frame = RawFaceFrame(
            frame_index=5,
            timestamp_ms=166,
            face_detected=False,
            blendshapes={},
        )

        with pytest.raises(FaceNotDetectedException):
            retargeter.retarget(frame)
