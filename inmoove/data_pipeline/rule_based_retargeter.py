"""
Rule-Based Face Retargeting 구현

MediaPipe에서 추출한 사람 얼굴 움직임(blendshape)을
i2Head 로봇의 16차원 FaceExpression으로 변환합니다.

설계 원칙:

1. Baseline Heuristic
   - 이 mapping은 연구적으로 최적화된 "정답"이 아닙니다
   - 전체 software architecture를 검증하기 위한 초기 baseline입니다
   - 실제 i2Head 데이터와 hardware 검증 후 수정 예정입니다

2. MediaPipe 값 ≠ Servo Angle
   - MediaPipe의 blendshape (0.0~1.0) ≠ 실제 Servo 각도
   - 이 Retargeter는 logical i2Head FaceExpression을 생성하며,
     실제 servo angle은 ServoCalibration 계층에서 처리합니다

3. Gaze와 Expression 분리
   - eye_left_lr, eye_left_ud, eye_right_lr, eye_right_ud는
     항상 neutral (0.5)을 유지합니다
   - 이유: 로봇이 웃으면서도 상대방을 계속 바라봐야 합니다
   - 시선 제어는 별도 Gaze Controller에서 수행합니다

4. Neutral-centered Mapping
   - 모든 i2Head 값은 neutral 0.5 기준으로 움직입니다
   - 증가 = 0.5보다 커짐
   - 감소 = 0.5보다 작아짐
   - 이는 actual servo direction과는 무관합니다

5. Multi-to-One Mapping
   - 여러 사람 얼굴 특징(blendshape)을 하나의 로봇 액추에이터로 조합
   - 예: browInnerUp + browOuterUpLeft - browDownLeft → eyebrow_left
   - 사람과 로봇의 얼굴 구조가 다르기 때문입니다

6. Strict Input Validation
   - RawFaceFrame 입력은 엄격하게 검증 (이미 __post_init__에서 처리됨)
   - Rule computation 후 output saturation (clamp)은 허용
   - 이 구분이 중요합니다

7. Face Not Detected 처리
   - face_detected=False인 경우 neutral로 몰래 처리하면 안 됩니다
   - 데이터상: 얼굴 검출 실패 ≠ 사람이 neutral 표정을 지었다
   - 따라서 명확한 예외를 발생시킵니다
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from inmoove.face import FaceExpression

from .models import RawFaceFrame
from .retargeting import FaceRetargeter
from .retargeting_config import RetargetingConfig, get_default_config


class FaceNotDetectedException(Exception):
    """얼굴이 검출되지 않았을 때 발생하는 예외입니다."""

    pass


class MissingBlendshapeException(Exception):
    """필수 blendshape가 누락되었을 때 발생하는 예외입니다."""

    pass


class RuleBasedFaceRetargeter(FaceRetargeter):
    """
    Rule-based heuristic을 사용하여 사람 얼굴 motion을 로봇 표정으로 변환합니다.

    이 구현은 MediaPipe의 blendshape 값을 heuristic rule의 조합으로
    i2Head FaceExpression의 16개 차원으로 매핑합니다.

    필수 이해사항:
    - 이 mapping은 baseline이며, 실제 데이터 기반 최적화가 필요합니다
    - 모든 로봇 값은 neutral (0.5) 기준의 logical representation입니다
    - 시선(gaze)은 표정(expression)과 완전히 분리됩니다
    """

    # 이 Retargeter가 필요로 하는 필수 blendshape 목록
    REQUIRED_BLENDSHAPES = {
        # 눈썹
        "browInnerUp",
        "browOuterUpLeft",
        "browOuterUpRight",
        "browDownLeft",
        "browDownRight",
        # 볼
        "cheekSquintLeft",
        "cheekSquintRight",
        # 눈
        "eyeBlinkLeft",
        "eyeBlinkRight",
        "eyeSquintLeft",
        "eyeSquintRight",
        "eyeWideLeft",
        "eyeWideRight",
        # 입
        "jawOpen",
        "mouthSmileLeft",
        "mouthSmileRight",
        "mouthUpperUpLeft",
        "mouthUpperUpRight",
        "mouthShrugUpper",
        "mouthRollUpper",
    }

    def __init__(self, config: Optional[RetargetingConfig] = None, strict: bool = True) -> None:
        """
        RuleBasedFaceRetargeter를 초기화합니다.

        Args:
            config: Retargeting 설정값 (기본값: RetargetingConfig())
            strict: True이면 missing blendshape에서 예외 발생,
                   False이면 누락된 값을 0.0으로 처리

        주의:
            strict=True가 기본값입니다. 이는 missing data를 명시적으로 드러내기 위함입니다.
        """
        self.config = config or get_default_config()
        self.strict = strict

    def retarget(self, frame: RawFaceFrame) -> FaceExpression:
        """
        RawFaceFrame을 FaceExpression으로 변환합니다.

        Args:
            frame: 변환할 얼굴 프레임

        Returns:
            FaceExpression: 로봇 얼굴 표정

        Raises:
            FaceNotDetectedException: 얼굴이 검출되지 않았을 때
            MissingBlendshapeException: strict=True 상태에서 필수 blendshape이 누락되었을 때
        """
        if not frame.face_detected:
            raise FaceNotDetectedException(
                f"Frame {frame.frame_index}에서 얼굴이 검출되지 않았습니다. "
                "이는 neutral 표정이 아니라 데이터 부재를 의미합니다."
            )

        self._validate_blendshapes(frame.blendshapes)
        bs = frame.blendshapes

        # 각 로봇 액추에이터 계산
        eyebrow_left = self._compute_eyebrow_left(bs)
        eyebrow_right = self._compute_eyebrow_right(bs)
        forehead_left = self._compute_forehead_left(bs)
        forehead_right = self._compute_forehead_right(bs)
        cheek_left = self._compute_cheek_left(bs)
        cheek_right = self._compute_cheek_right(bs)
        eyelid_left_upper = self._compute_eyelid_left_upper(bs)
        eyelid_left_lower = self._compute_eyelid_left_lower(bs)
        eyelid_right_upper = self._compute_eyelid_right_upper(bs)
        eyelid_right_lower = self._compute_eyelid_right_lower(bs)
        upper_lip = self._compute_upper_lip(bs)
        jaw = self._compute_jaw(bs)

        return FaceExpression(
            eye_left_lr=self.config.neutral_value,
            eye_left_ud=self.config.neutral_value,
            eye_right_lr=self.config.neutral_value,
            eye_right_ud=self.config.neutral_value,
            eyelid_left_upper=eyelid_left_upper,
            eyelid_left_lower=eyelid_left_lower,
            eyelid_right_upper=eyelid_right_upper,
            eyelid_right_lower=eyelid_right_lower,
            eyebrow_left=eyebrow_left,
            eyebrow_right=eyebrow_right,
            cheek_left=cheek_left,
            cheek_right=cheek_right,
            forehead_left=forehead_left,
            forehead_right=forehead_right,
            upper_lip=upper_lip,
            jaw=jaw,
        )

    # ========== Private Helper Methods ==========

    def _validate_blendshapes(self, blendshapes: Dict[str, float]) -> None:
        """
        필수 blendshape이 모두 존재하는지 검증합니다.

        Raises:
            MissingBlendshapeException: strict=True이고 필수 key가 누락될 때
        """
        if self.strict:
            missing = self.REQUIRED_BLENDSHAPES - set(blendshapes.keys())
            if missing:
                raise MissingBlendshapeException(
                    f"필수 blendshape이 누락되었습니다: {missing}"
                )

    def _get_blendshape(self, blendshapes: Dict[str, float], name: str, default: float = 0.0) -> float:
        """blendshape 값을 안전하게 가져옵니다."""
        if name in blendshapes:
            return float(blendshapes[name])
        if self.strict and name in self.REQUIRED_BLENDSHAPES:
            raise MissingBlendshapeException(f"필수 blendshape '{name}'이 누락되었습니다")
        return default

    def _compute_eyebrow_left(self, bs: Dict[str, float]) -> float:
        """
        왼쪽 눈썹 계산

        매핑 규칙:
        browInnerUp + browOuterUpLeft를 사용하여 위로 올려지는 동작
        browDownLeft를 사용하여 아래로 내려가는 동작
        """
        up_component = (
            self.config.brow_inner_up_gain * self._get_blendshape(bs, "browInnerUp")
            + self.config.brow_up_gain * self._get_blendshape(bs, "browOuterUpLeft")
        )
        down_component = self.config.brow_down_gain * self._get_blendshape(bs, "browDownLeft")

        delta = up_component - down_component
        result = self.config.neutral_value + 0.35 * delta

        return self._clamp01(result)

    def _compute_eyebrow_right(self, bs: Dict[str, float]) -> float:
        """오른쪽 눈썹 계산 (왼쪽과 동일한 논리, 오른쪽 값 사용)"""
        up_component = (
            self.config.brow_inner_up_gain * self._get_blendshape(bs, "browInnerUp")
            + self.config.brow_up_gain * self._get_blendshape(bs, "browOuterUpRight")
        )
        down_component = self.config.brow_down_gain * self._get_blendshape(bs, "browDownRight")

        delta = up_component - down_component
        result = self.config.neutral_value + 0.35 * delta

        return self._clamp01(result)

    def _compute_forehead_left(self, bs: Dict[str, float]) -> float:
        """
        왼쪽 이마 계산

        이마는 눈썹과 유사한 움직임을 보이지만, 더 미묘합니다.
        browInnerUp이 주요 요인이며, browOuterUp이 보조합니다.
        """
        delta = (
            self.config.forehead_inner_up_gain * self._get_blendshape(bs, "browInnerUp")
            + self.config.forehead_outer_up_gain * self._get_blendshape(bs, "browOuterUpLeft")
        )

        result = self.config.neutral_value + 0.40 * delta

        return self._clamp01(result)

    def _compute_forehead_right(self, bs: Dict[str, float]) -> float:
        """오른쪽 이마 계산"""
        delta = (
            self.config.forehead_inner_up_gain * self._get_blendshape(bs, "browInnerUp")
            + self.config.forehead_outer_up_gain * self._get_blendshape(bs, "browOuterUpRight")
        )

        result = self.config.neutral_value + 0.40 * delta

        return self._clamp01(result)

    def _compute_cheek_left(self, bs: Dict[str, float]) -> float:
        """
        왼쪽 볼 계산

        매핑 규칙:
        cheekSquintLeft (눈 주변 긴장): 볼 올려짐
        mouthSmileLeft (웃음): 볼 올려짐
        """
        delta = (
            self.config.cheek_squint_gain * self._get_blendshape(bs, "cheekSquintLeft")
            + self.config.smile_to_cheek_gain * self._get_blendshape(bs, "mouthSmileLeft")
        )

        result = self.config.neutral_value + 0.40 * delta

        return self._clamp01(result)

    def _compute_cheek_right(self, bs: Dict[str, float]) -> float:
        """오른쪽 볼 계산"""
        delta = (
            self.config.cheek_squint_gain * self._get_blendshape(bs, "cheekSquintRight")
            + self.config.smile_to_cheek_gain * self._get_blendshape(bs, "mouthSmileRight")
        )

        result = self.config.neutral_value + 0.40 * delta

        return self._clamp01(result)

    def _compute_eyelid_left_upper(self, bs: Dict[str, float]) -> float:
        """
        왼쪽 눈 위꺼풀 계산

        매핑 규칙:
        eyeBlinkLeft 증가 → 눈꺼풀 닫힘 (값 감소)
        eyeWideLeft 증가 → 눈꺼풀 열림 (값 증가)
        두 효과는 반대 방향입니다.
        """
        blink_effect = -self.config.blink_upper_gain * self._get_blendshape(bs, "eyeBlinkLeft")
        wide_effect = self.config.wide_upper_gain * self._get_blendshape(bs, "eyeWideLeft")

        delta = blink_effect + wide_effect
        result = self.config.neutral_value + 0.35 * delta

        return self._clamp01(result)

    def _compute_eyelid_right_upper(self, bs: Dict[str, float]) -> float:
        """오른쪽 눈 위꺼풀 계산"""
        blink_effect = -self.config.blink_upper_gain * self._get_blendshape(bs, "eyeBlinkRight")
        wide_effect = self.config.wide_upper_gain * self._get_blendshape(bs, "eyeWideRight")

        delta = blink_effect + wide_effect
        result = self.config.neutral_value + 0.35 * delta

        return self._clamp01(result)

    def _compute_eyelid_left_lower(self, bs: Dict[str, float]) -> float:
        """
        왼쪽 눈 아래꺼풀 계산

        매핑 규칙:
        eyeSquintLeft: 눈을 깔짝거림 (아래꺼풀 올려짐)
        eyeBlinkLeft의 일부: blink의 일부가 아래꺼풀에도 영향
        """
        squint_effect = self.config.squint_lower_gain * self._get_blendshape(bs, "eyeSquintLeft")
        blink_effect = self.config.blink_lower_gain * self._get_blendshape(bs, "eyeBlinkLeft")

        delta = squint_effect + blink_effect
        result = self.config.neutral_value + 0.35 * delta

        return self._clamp01(result)

    def _compute_eyelid_right_lower(self, bs: Dict[str, float]) -> float:
        """오른쪽 눈 아래꺼풀 계산"""
        squint_effect = self.config.squint_lower_gain * self._get_blendshape(bs, "eyeSquintRight")
        blink_effect = self.config.blink_lower_gain * self._get_blendshape(bs, "eyeBlinkRight")

        delta = squint_effect + blink_effect
        result = self.config.neutral_value + 0.35 * delta

        return self._clamp01(result)

    def _compute_upper_lip(self, bs: Dict[str, float]) -> float:
        """
        윗입술 계산

        매핑 규칙:
        mouthUpperUp: 입술 올려짐
        mouthShrugUpper: 입술 올려짐
        mouthRollUpper: 입술 내려짐
        """
        delta = (
            self.config.upper_up_gain
            * (
                self._get_blendshape(bs, "mouthUpperUpLeft")
                + self._get_blendshape(bs, "mouthUpperUpRight")
            )
            / 2.0
            + self.config.mouth_shrug_upper_gain * self._get_blendshape(bs, "mouthShrugUpper")
            - self.config.mouth_roll_upper_gain * self._get_blendshape(bs, "mouthRollUpper")
        )

        result = self.config.neutral_value + 0.40 * delta

        return self._clamp01(result)

    def _compute_jaw(self, bs: Dict[str, float]) -> float:
        """
        턱 계산

        가장 명확한 매핑 중 하나입니다.
        jawOpen 값이 증가하면 턱이 벌려집니다.
        """
        jaw_open = self._get_blendshape(bs, "jawOpen")
        delta = jaw_open

        result = self.config.neutral_value + 0.35 * delta

        return self._clamp01(result)

    @staticmethod
    def _clamp01(value: float) -> float:
        """
        값을 0.0 ~ 1.0 범위로 제한합니다.

        주요 포인트:
        - RawFaceFrame 입력은 __post_init__에서 이미 엄격히 검증됩니다
        - 여기서의 clamp는 여러 rule의 조합으로 나타나는 logical output saturation입니다
        - 향후 smoothing/interpolation을 추가할 때도 이 범위는 유지합니다
        """
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value
