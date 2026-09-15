# InMoove i2Head 대화형 로봇 얼굴 제어 시스템

실제 하드웨어 없이도 LLM 기반의 자연스러운 로봇 표정 제어를 개발할 수 있는 프로젝트입니다.

## 🎯 프로젝트 목표

최종 목표는 LLM이 대화 상황을 이해하고, 고정된 HAPPY/SAD 같은 표정 프리셋만 선택하는 것이 아니라 여러 얼굴 액추에이터를 연속적인 값으로 제어하여 자연스러운 표정을 생성하는 것입니다.

## 📐 아키텍처

```
LLM / MediaPipe / Retargeting
            ↓
      FaceExpression (16차원 정규화)
            ↓
       HeadInterface (추상 인터페이스)
         ↓        ↓
     MockHead   RealHead (향후)
                 ↓
           MyRobotLab
                 ↓
          Arduino/PCA9685
                 ↓
              Servo Motor
```

### 핵심 설계 원칙

- **AI와 하드웨어 분리**: 상위 AI 코드는 하드웨어 구현을 알 필요가 없음
- **정규화된 값**: 모든 표정 값은 0.0 ~ 1.0 범위
- **명확한 예외**: 범위를 벗어난 값은 조용히 잘라내지 않고 예외 발생
- **Mock 지원**: 실제 하드웨어 없이도 개발 가능

## 📦 주요 컴포넌트

### 1. FaceExpression (16차원)

i2Head의 얼굴 상태를 표현하는 데이터 클래스:

```python
@dataclass
class FaceExpression:
    # 눈 (좌우, 상하)
    eye_left_lr: float
    eye_left_ud: float
    eye_right_lr: float
    eye_right_ud: float
    
    # 눈꺼풀 (위, 아래)
    eyelid_left_upper: float
    eyelid_left_lower: float
    eyelid_right_upper: float
    eyelid_right_lower: float
    
    # 눈썹
    eyebrow_left: float
    eyebrow_right: float
    
    # 볼
    cheek_left: float
    cheek_right: float
    
    # 이마
    forehead_left: float
    forehead_right: float
    
    # 입술과 턱
    upper_lip: float
    jaw: float
```

- 모든 값은 0.0 ~ 1.0 범위
- 범위를 벗어나면 `ValueError` 발생
- `FaceExpression.neutral()`: 중립 표정 생성

### 2. HeadInterface (추상 인터페이스)

상위 코드와 하드웨어 구현 사이의 계약:

```python
class HeadInterface(ABC):
    @abstractmethod
    def set_expression(self, expression: FaceExpression) -> None:
        """얼굴 표정 설정"""
        pass
    
    def rest(self) -> None:
        """휴지 상태 (기본: 중립 표정)"""
        pass
```

### 3. MockHead (Mock 구현)

실제 하드웨어 없이 동작하는 구현:

```python
head = MockHead()
expr = FaceExpression(...)
head.set_expression(expr)
```

기능:
- 표정 상태 저장 및 조회
- 호출 이력 기록
- 터미널 출력 (CLI 시각화)

## 🚀 시작하기

### 설치

```bash
# 저장소 클론
git clone <repository>
cd Inmoove

# 의존성 설치
pip install -r requirements.txt
```

### 선택 사항: MiniCPM 키보드 채팅 및 이미지 질의

MiniCPM 예제에 필요한 선택 의존성을 설치합니다.

```bash
python3 -m pip install -r requirements-minicpm.txt
python3 examples/minicpm_text_chat.py
```

첫 실행에서는 약 2.6GB 모델을 다운로드합니다. 대화를 끝내려면 `/quit`을 입력하세요.

저장된 사진 한 장과 질문으로 Step 2를 실행하려면 다음처럼 입력합니다. `--image`에는 로컬 이미지 파일 경로를, `--message`에는 사진을 보고 답할 문장을 전달합니다.

```bash
python3 examples/minicpm_image_chat.py \
  --image assets/step2_photo.jpg \
  --message "사진 속 인물의 표정과 행동, 주변 환경을 한국어 한 문장으로 설명해줘."
```

RTX 5080 개발 환경에서는 Step 1에서 검증한 PyTorch CUDA 조합과 호환되는 `torchvision`을 유지한 가상환경으로 위 명령을 실행하세요.

### 간단한 예제 실행

```bash
python examples/simple_demo.py
```

출력 예:

```
==================================================
[MockHead] 표정 업데이트됨
==================================================
  Eye Left Lr          :  0.50 [■■■■■■■■■■□□□□□□□□□□] 50.0%
  Eye Left Ud          :  0.60 [■■■■■■■□□□□□□□□□□□□□] 60.0%
  ...
  Jaw                  :  0.30 [■■■■■■□□□□□□□□□□□□□□] 30.0%
==================================================
```

### 테스트 실행

```bash
# 모든 테스트 실행
pytest

# 상세한 출력
pytest -v

# 특정 테스트 파일
pytest tests/test_face_expression.py

# 커버리지 확인
pytest --cov=inmoove
```

## 🎬 Face Retargeting (Rule-Based Approach)

### 개요

**Face Retargeting**은 인간의 얼굴 움직임 (MediaPipe blendshape)을 로봇의 표정 (FaceExpression)으로 변환하는 계층입니다.

**중요**: 이 retargeting은 **baseline heuristic**이며, 실제 로봇 동작을 위한 근거 있는 최종 매개변수가 아닙니다. 향후 실제 데이터를 수집하고 분석하여 이 매핑을 개선할 수 있습니다.

### 데이터 파이프라인

```
Human Face Motion (MediaPipe)
         ↓
    20 Blendshapes
         ↓
  RawFaceFrame
         ↓
RuleBasedFaceRetargeter
         ↓
   FaceExpression (16D)
         ↓
    MockHead/RealHead
```

### 핵심 설계 원칙

1. **Expression과 Gaze 분리**
   - Eye gaze (좌우, 상하): 항상 neutral (0.5) 유지
   - Expression (눈썹, 뺨, 눈꺼풀 등): MediaPipe 기반 변환
   - Future: Separate gaze controller layer

2. **Neutral-Centered Mapping**
   - 0.5 = 중립 상태
   - 0.0 ~ 0.5: 감소 방향
   - 0.5 ~ 1.0: 증가 방향

3. **Multi-to-Single Mapping**
   - 여러 MediaPipe blendshape → 하나의 robot actuator
   - 예: eyebrow = 0.45×browInnerUp + 0.55×browOuterUp - 0.70×browDown

4. **Strict Input Validation**
   - face_detected=False 시 명시적 예외 발생
   - Missing blendshape 시 선택적 처리 (strict mode)

### 사용 예제

```python
from inmoove.data_pipeline import (
    MockFaceDataExtractor,
    RuleBasedFaceRetargeter,
    get_default_config
)

# 1. Face data 추출 (Mock용)
extractor = MockFaceDataExtractor(frame_count=100, fps=30)
raw_frames = list(extractor.extract())

# 2. Retargeter 생성
retargeter = RuleBasedFaceRetargeter(strict=False)

# 3. Human face → Robot face 변환
for frame in raw_frames:
    if frame.face_detected:
        robot_expression = retargeter.retarget(frame)
        head.set_expression(robot_expression)
```

### Retargeting Config

모든 gain 값은 중앙화되어 관리됩니다:

```python
from inmoove.data_pipeline import get_default_config

config = get_default_config()
# config.brow_up_gain = 0.55
# config.cheek_squint_gain = 0.60
# config.jaw_open_gain = 0.80
# ... (18개 파라미터)
```

### 매핑 예시

**눈썹 (Eyebrow)**
```
eyebrow_left = 0.5 + 0.35 × (
    0.45 × browInnerUp
    + 0.55 × browOuterUpLeft
    - 0.70 × browDownLeft
)
```

**뺨 (Cheek)**
```
cheek_left = 0.5 + 0.40 × (
    0.60 × cheekSquintLeft
    + 0.40 × mouthSmileLeft
)
```

**턱 (Jaw)**
```
jaw = 0.5 + 0.80 × jawOpen
```

### Baseline Heuristic 특성

이 매핑이 "baseline"이라는 의미:

1. **검증되지 않은 초기 추측**: 로봇 기술자와 협력하여 수작업으로 결정
2. **개선 대상**: 실제 로봇 데이터를 수집한 후 조정 필요
3. **아키텍처 검증용**: 파이프라인이 올바르게 작동하는지 확인하는 용도
4. **Future 머신러닝 기반**: 데이터 충분 시 learned model로 대체 가능

### 데모 실행

전체 파이프라인을 테스트하려면:

```bash
# Face retargeting 데모 (20 frame, MockHead 적용)
python examples/retargeting_demo.py
```

출력: 20개 프레임의 MediaPipe blendshape을 robot expression으로 변환하고, MockHead에 적용한 결과를 표시합니다.

## 📋 테스트 항목

✅ FaceExpression
- 정상적인 생성
- 0.0, 1.0 값 허용
- 범위 벗어난 값 거부
- neutral() 생성
- 변환 함수 (to_dict, get_all_values 등)

✅ MockHead
- 표정 저장 및 조회
- rest 동작
- 연속 표정 변환
- 호출 이력 기록

## 🔄 데이터 흐름 예제

```python
# 1. MockHead 생성
head = MockHead(verbose=True)

# 2. 중립 표정
head.set_expression(FaceExpression.neutral())

# 3. 웃음 표정
smile = FaceExpression(
    eye_left_lr=0.5,
    eye_left_ud=0.6,
    # ... 14개 더
)
head.set_expression(smile)

# 4. 현재 상태 조회
current = head.get_current_expression()
print(current.jaw)  # 0.3

# 5. 호출 이력 조회
history = head.get_expression_history()
for entry in history:
    print(entry["timestamp"], entry["expression"])

# 6. 휴지 상태
head.rest()
```

## 🚫 현재 범위 밖의 기능

다음은 향후 단계에서 추가될 예정입니다:

- LLM 연동
- MediaPipe 얼굴 인식
- 표정 retargeting
- 시간축 기반 부드러운 전환 (smoothing/interpolation)
- 실제 MyRobotLab 통신
- PCA9685 서보 제어
- 서보 calibration
- GUI

## 📚 다음 단계

### 1단계: HeadInterface 구현 (현재 ✅)
- FaceExpression 정의
- HeadInterface 추상 클래스
- MockHead 구현

### 2단계: RealHead 구현 (향후)
```
FaceExpression → RealHead → ServoCalibration → MyRobotLab → Arduino/PCA9685
```

### 3단계: AI 통합
- LLM 기반 표정 생성
- MediaPipe retargeting

## 📖 코드 구조

```
inmoove/
├── __init__.py
└── face/
    ├── __init__.py
    ├── expression.py      # FaceExpression 클래스
    ├── interface.py       # HeadInterface 추상 클래스
    └── mock_head.py       # MockHead 구현

tests/
├── __init__.py
├── test_face_expression.py
└── test_mock_head.py

examples/
└── simple_demo.py

.gitignore
.gitignore
pytest.ini
requirements.txt
README.md
```

## 💡 설계 철학

### 왜 정규화된 값(0.0 ~ 1.0)?

- 서보 각도의 실제 값에 독립적
- 여러 하드웨어 플랫폼에 쉽게 적응 가능
- LLM 출력을 직접 사용 가능 (범위가 명확함)

### 왜 HeadInterface를 추상 클래스로?

- MockHead와 RealHead의 동일한 계약 보장
- 나중에 RealHead를 추가해도 상위 코드 변경 불필요
- 의존성 주입(Dependency Injection) 지원

### 왜 예외를 발생시키는가?

- AI 출력 오류를 초기에 발견
- 자동 조정은 숨겨진 버그를 발생시킬 수 있음
- 명확한 문제 진단

## 👥 기여

특정 기능 제안이나 버그 리포트는 이슈 탭에서 제출해주세요.

## 📄 라이선스

이 프로젝트는 InMoove 커뮤니티를 위해 개발되었습니다.

---

**현재 상태**: MockHead 구현 완료, 테스트 통과
**다음 단계**: RealHead 구현 (실제 서보 드라이버 도착 후)
