"""
얼굴 표정 데이터 구조 정의

FaceExpression은 i2Head 로봇의 16개 액추에이터 상태를 정규화된 값(0.0~1.0)으로
표현하는 데이터 클래스입니다.

설계 원칙:
- 모든 값은 0.0 ~ 1.0 범위로 정규화됨
- 범위를 벗어난 값은 자동으로 조정되지 않고 명확한 예외 발생
- 실제 서보 각도나 하드웨어 값을 포함하지 않음
- 나중에 시간축을 따라 보간할 수 있도록 설계됨

향후 확장:
- 시간축 기반 표정 부드러운 전이 (smoothing/interpolation)
- MediaPipe 얼굴 인식과의 retargeting
- LLM 기반 표정 생성
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class FaceExpression:
    """
    i2Head 로봇의 얼굴 표정을 나타내는 16차원 정규화 벡터
    
    각 필드는 특정 얼굴 액추에이터의 상태를 0.0(최소) ~ 1.0(최대)로 표현합니다.
    
    참고:
    - 이 클래스는 순수 데이터 구조입니다
    - 실제 서보 제어 로직은 HeadInterface 구현체에 위임됩니다
    - 실제 서보 각도로의 변환은 향후 ServoCalibration 모듈에서 처리합니다
    """
    
    # 좌측 눈 (상좌우, 상하)
    eye_left_lr: float   # 0.0(좌측) ~ 1.0(우측)
    eye_left_ud: float   # 0.0(아래) ~ 1.0(위)
    
    # 우측 눈 (상좌우, 상하)
    eye_right_lr: float  # 0.0(좌측) ~ 1.0(우측)
    eye_right_ud: float  # 0.0(아래) ~ 1.0(위)
    
    # 좌측 눈꺼풀 (위, 아래)
    eyelid_left_upper: float   # 0.0(닫힘) ~ 1.0(뜸)
    eyelid_left_lower: float   # 0.0(뜸) ~ 1.0(닫힘)
    
    # 우측 눈꺼풀 (위, 아래)
    eyelid_right_upper: float  # 0.0(닫힘) ~ 1.0(뜸)
    eyelid_right_lower: float  # 0.0(뜸) ~ 1.0(닫힘)
    
    # 눈썹
    eyebrow_left: float   # 0.0(내려옴) ~ 1.0(올려짐)
    eyebrow_right: float  # 0.0(내려옴) ~ 1.0(올려짐)
    
    # 볼
    cheek_left: float   # 0.0(내려옴) ~ 1.0(올려짐)
    cheek_right: float  # 0.0(내려옴) ~ 1.0(올려짐)
    
    # 이마
    forehead_left: float   # 0.0(이완) ~ 1.0(주름)
    forehead_right: float  # 0.0(이완) ~ 1.0(주름)
    
    # 입술과 턱
    upper_lip: float  # 0.0(내려옴) ~ 1.0(올려짐)
    jaw: float        # 0.0(닫힘) ~ 1.0(벌림)
    
    # 유효성 검사를 위한 상수
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0
    
    def __post_init__(self):
        """
        모든 필드가 유효한 범위(0.0 ~ 1.0)에 있는지 검증합니다.
        
        범위를 벗어난 값이 있으면 즉시 예외를 발생시킵니다.
        이는 AI 출력 오류를 초기에 발견하기 위한 의도적 설계입니다.
        
        Raises:
            ValueError: 범위를 벗어난 값이 있을 때
        """
        for field_name, value in self.__dict__.items():
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{field_name}은 숫자여야 합니다, 받은 값: {type(value)}"
                )
            
            if not (self.MIN_VALUE <= value <= self.MAX_VALUE):
                raise ValueError(
                    f"{field_name} 값이 범위를 벗어났습니다. "
                    f"기대값: {self.MIN_VALUE} ~ {self.MAX_VALUE}, "
                    f"받은 값: {value}"
                )
    
    @classmethod
    def neutral(cls) -> "FaceExpression":
        """
        중립 얼굴 상태를 생성합니다.
        
        모든 액추에이터가 중간값(0.5)에 설정된 상태입니다.
        
        참고:
        - 이는 테스트 및 개발용 기본값입니다
        - 실제 i2Head 로봇의 중립 서보 각도는 향후 calibration 과정에서
          결정되어야 합니다
        - 현재는 정규화 공간에서의 순수한 "중간값"만 의미합니다
        
        Returns:
            FaceExpression: 모든 값이 0.5인 중립 표정
        """
        return cls(
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
    
    def to_dict(self) -> Dict[str, float]:
        """
        표정 상태를 딕셔너리로 변환합니다.
        
        Returns:
            Dict[str, float]: 필드명과 값의 매핑
        """
        return {
            "eye_left_lr": self.eye_left_lr,
            "eye_left_ud": self.eye_left_ud,
            "eye_right_lr": self.eye_right_lr,
            "eye_right_ud": self.eye_right_ud,
            "eyelid_left_upper": self.eyelid_left_upper,
            "eyelid_left_lower": self.eyelid_left_lower,
            "eyelid_right_upper": self.eyelid_right_upper,
            "eyelid_right_lower": self.eyelid_right_lower,
            "eyebrow_left": self.eyebrow_left,
            "eyebrow_right": self.eyebrow_right,
            "cheek_left": self.cheek_left,
            "cheek_right": self.cheek_right,
            "forehead_left": self.forehead_left,
            "forehead_right": self.forehead_right,
            "upper_lip": self.upper_lip,
            "jaw": self.jaw,
        }
    
    def get_all_values(self) -> Tuple[float, ...]:
        """
        모든 값을 순서대로 튜플로 반환합니다.
        
        Returns:
            Tuple[float, ...]: 16개의 정규화된 값
        """
        d = self.to_dict()
        return tuple(d.values())
    
    @staticmethod
    def get_field_names() -> Tuple[str, ...]:
        """
        모든 필드명을 순서대로 반환합니다.
        
        Returns:
            Tuple[str, ...]: 16개의 필드 이름
        """
        return (
            "eye_left_lr",
            "eye_left_ud",
            "eye_right_lr",
            "eye_right_ud",
            "eyelid_left_upper",
            "eyelid_left_lower",
            "eyelid_right_upper",
            "eyelid_right_lower",
            "eyebrow_left",
            "eyebrow_right",
            "cheek_left",
            "cheek_right",
            "forehead_left",
            "forehead_right",
            "upper_lip",
            "jaw",
        )
