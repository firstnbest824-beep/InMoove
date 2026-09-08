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
