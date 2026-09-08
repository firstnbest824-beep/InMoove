"""
Rule-Based Face Retargeting 설정

MediaPipe blendshape를 i2Head FaceExpression으로 변환할 때 사용하는
모든 파라미터를 한곳에서 관리합니다.

이번 mapping은 연구적으로 최적화된 정답이 아니라, 
전체 software architecture를 검증하기 위한 초기 baseline입니다.

실제 i2Head hardware와 데이터를 확보한 후 이 값들을 조정할 것입니다.
"""

from dataclasses import dataclass


@dataclass
class RetargetingConfig:
    """
    Rule-based face retargeting의 모든 gain과 neutral 설정을 중앙화합니다.

    각 값은 다음처럼 해석됩니다.
    - neutral_value: 중립 얼굴 상태 (일반적으로 0.5)
    - 각 gain: MediaPipe blendshape의 영향도 (0.0 ~ 1.0)

    원칙:
    - 값을 직접 코드에 흩뿌리지 않음
    - 전체 시스템의 일관성 유지
    - 향후 데이터 기반 최적화 가능하게 설계
    """

    # === Neutral 기준값 ===
    neutral_value: float = 0.5

    # === 눈썹 (eyebrow) ===
    # 위로 올라갈 때: browInnerUp + browOuterUp
    # 아래로 내려갈 때: browDown
    brow_up_gain: float = 0.55  # browOuterUp의 영향도
    brow_inner_up_gain: float = 0.45  # browInnerUp의 영향도
    brow_down_gain: float = 0.70  # browDown의 영향도 (감소)

    # === 이마 (forehead) ===
    # 눈썹보다는 약하게, 주로 browInnerUp의 영향을 받음
    forehead_inner_up_gain: float = 0.35
    forehead_outer_up_gain: float = 0.25

    # === 볼 (cheek) ===
    # cheekSquint와 mouthSmile의 조합
    cheek_squint_gain: float = 0.60
    smile_to_cheek_gain: float = 0.40

    # === 눈꺼풀 (eyelid) ===
    # Upper: blink와 wide가 반대 방향으로 작용
    # Lower: squint와 일부 blink
    blink_upper_gain: float = 0.70  # blink가 증가하면 눈꺼풀 닫힘
    wide_upper_gain: float = 0.80  # wide가 증가하면 눈꺼풀 열림

    blink_lower_gain: float = 0.50
    squint_lower_gain: float = 0.65

    # === 윗입술 (upper_lip) ===
    upper_up_gain: float = 0.50  # mouthUpperUp의 영향도
    mouth_shrug_upper_gain: float = 0.35  # mouthShrugUpper의 영향도
    mouth_roll_upper_gain: float = 0.25  # mouthRollUpper의 영향도 (감소)

    # === 턱 (jaw) ===
    # 가장 단순한 매핑: jawOpen이 증가하면 jaw가 증가
    jaw_open_gain: float = 0.80

    # === 눈 방향 (gaze) ===
    # 주의: 현재 이 시스템은 gaze를 expression과 분리합니다.
    # 이유: 로봇이 웃는 동안에도 상대방을 계속 바라볼 수 있어야 하기 때문입니다.
    # 따라서 eye_left_lr, eye_left_ud, eye_right_lr, eye_right_ud는
    # 항상 neutral_value (0.5)를 유지합니다.
    # 시선 제어는 별도의 Gaze Controller 계층에서 수행됩니다.


def get_default_config() -> RetargetingConfig:
    """기본 설정을 반환합니다."""
    return RetargetingConfig()
