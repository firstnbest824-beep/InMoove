#!/usr/bin/env python3
"""
Face Retargeting 데모

이 데모는 다음과 같은 전체 파이프라인을 보여줍니다:
1. MockFaceDataExtractor로 mock human face motion 생성 (20개 blendshape)
2. RuleBasedFaceRetargeter로 robot face expression으로 변환
3. MockHead로 표현을 적용

중요: 이 retargeting의 값은 로봇 얼굴 학습의 정답 데이터가 아니라,
향후 실제 데이터 기반 방법과 비교하기 위한 baseline입니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inmoove.face import FaceExpression, MockHead
from inmoove.data_pipeline import (
    MockFaceDataExtractor,
    RuleBasedFaceRetargeter,
    get_default_config,
)


def print_header(title: str, width: int = 80):
    """데모 섹션 헤더를 출력합니다."""
    print("\n" + "=" * width)
    print(f" {title}".ljust(width))
    print("=" * width)


def print_expressions(expressions: list[FaceExpression], title: str = "Robot Face Expressions"):
    """여러 표현을 표 형식으로 출력합니다."""
    print_header(title)
    
    # 헤더
    print(f"{'Frame':<8} {'Eyebrow L':<12} {'Eyebrow R':<12} {'Cheek L':<12} {'Cheek R':<12}")
    print(f"       {'Eyelid U':<12} {'Upper Lip':<12} {'Jaw':<12}")
    print("-" * 80)
    
    # 데이터
    for idx, expr in enumerate(expressions):
        if idx < 10:  # 처음 10개 프레임만 표시
            print(
                f"{idx:<8} "
                f"{expr.eyebrow_left:.4f}    "
                f"{expr.eyebrow_right:.4f}    "
                f"{expr.cheek_left:.4f}    "
                f"{expr.cheek_right:.4f}"
            )
            print(
                f"       "
                f"{expr.eyelid_left_upper:.4f}    "
                f"{expr.upper_lip:.4f}    "
                f"{expr.jaw:.4f}"
            )
        elif idx == 10:
            print("[... 나머지 프레임은 생략됨 ...]")
            break


def main():
    """메인 데모 로직"""
    
    print_header("Face Retargeting Demo - 인모브 로봇 얼굴 리타게팅", 80)
    print("""
이 데모는 다음을 보여줍니다:
1. Mock human face motion 데이터 생성 (MediaPipe 스타일 blendshape)
2. Robot face expression으로 변환 (RuleBasedFaceRetargeter)
3. Robot head (MockHead)에 적용

주의: 이 mapping은 baseline heuristic이며, 실제 로봇 얼굴 조정이 필요합니다.
""")
    
    # 1. Extractor 생성 및 데이터 추출
    print_header("1단계: Mock Face Data Extraction")
    print("MockFaceDataExtractor를 사용하여 20개 프레임의 mock face data 생성...")
    
    extractor = MockFaceDataExtractor(
        frame_count=20,
        fps=30,
        source_id="demo_mock_camera"
    )
    raw_frames = list(extractor.extract())
    print(f"✓ {len(raw_frames)}개 프레임 생성 완료")
    
    # 프레임 샘플 출력
    first_frame = raw_frames[0]
    print(f"\n첫 번째 프레임 샘플:")
    print(f"  - Frame index: {first_frame.frame_index}")
    print(f"  - Timestamp: {first_frame.timestamp_ms}ms")
    print(f"  - Face detected: {first_frame.face_detected}")
    print(f"  - Blendshapes count: {len(first_frame.blendshapes)}")
    print(f"  - Sample blendshapes:")
    sample_blendshapes = list(first_frame.blendshapes.items())[:5]
    for name, value in sample_blendshapes:
        print(f"      {name}: {value:.4f}")
    
    # 2. Retargeter 생성 및 변환
    print_header("2단계: Face Expression Retargeting")
    print("RuleBasedFaceRetargeter를 사용하여 human face → robot face 변환...")
    
    config = get_default_config()
    print(f"\nRetargeting 설정:")
    print(f"  - Neutral value: {config.neutral_value}")
    print(f"  - Brow gains: up={config.brow_up_gain}, inner_up={config.brow_inner_up_gain}, down={config.brow_down_gain}")
    print(f"  - Cheek gains: squint={config.cheek_squint_gain}, smile={config.smile_to_cheek_gain}")
    print(f"  - Jaw gain: {config.jaw_open_gain}")
    
    retargeter = RuleBasedFaceRetargeter(strict=False)  # demo에서는 관대하게
    robot_expressions = []
    
    for frame in raw_frames:
        try:
            if frame.face_detected:
                expr = retargeter.retarget(frame)
                robot_expressions.append(expr)
        except Exception as e:
            print(f"⚠ Frame {frame.frame_index} 변환 실패: {e}")
    
    print(f"✓ {len(robot_expressions)}개 표현 변환 완료")
    
    # 표현 샘플 출력
    print_expressions(robot_expressions, "Robot Face Expressions (처음 10개 프레임)")
    
    # 3. Robot Head 적용
    print_header("3단계: Robot Head Application")
    print("MockHead를 사용하여 robot face 표현 적용...")
    
    head = MockHead()
    applied_count = 0
    
    for idx, expr in enumerate(robot_expressions[:5]):  # 처음 5개만 적용
        try:
            head.set_expression(expr)
            applied_count += 1
            print(f"  Frame {idx}: 표현 적용 완료")
            print(f"    eyebrow_left={expr.eyebrow_left:.4f}, "
                  f"eyebrow_right={expr.eyebrow_right:.4f}, "
                  f"jaw={expr.jaw:.4f}")
        except Exception as e:
            print(f"  Frame {idx}: 적용 실패 - {e}")
    
    print(f"✓ {applied_count}개 표현 robot에 적용 완료")
    
    # 4. Neutral 상태로 복구
    print_header("4단계: Robot Reset")
    print("Robot을 neutral 상태로 복구...")
    
    try:
        head.rest()
        print("✓ Robot을 rest 상태(neutral)로 설정 완료")
    except Exception as e:
        print(f"⚠ Rest 적용 중 오류: {e}")
    
    # 최종 요약
    print_header("파이프라인 요약")
    print(f"""
변환 결과:
  - 입력: {len(raw_frames)} frame × 20 blendshape (human face)
  - 출력: {len(robot_expressions)} frame × 16 dimension (robot face)
  - 적용: {applied_count} frame을 MockHead에 적용

주요 학습 포인트:
  1. Human face (20 blendshape) → Robot face (16 dimension) 변환 성공
  2. Gaze 분리: eye_*_lr/ud는 항상 0.5 (neutral) 유지
  3. Expression 매핑:
     - 눈썹: browInnerUp, browOuterUp*, browDown* 조합
     - 볼: cheekSquint*, mouthSmile* 조합
     - 눈꺼풀: eyeBlink*, eyeWide*, eyeSquint* 조합
     - 입: mouthUpperUp*, mouthShrugUpper, mouthRollUpper 조합
     - 턱: jawOpen 직접 매핑
  4. 모든 출력값은 clamp(0.0, 1.0)으로 범위 제한

주의사항:
  ⚠ 이 retargeting 값은 "baseline heuristic"입니다!
  ⚠ 실제 로봇 동작을 위해:
     1. 서보 캘리브레이션이 필요합니다 (logical space → physical angle)
     2. 실제 데이터와 비교하여 gain 값을 조정해야 합니다
     3. 로봇의 실제 facial structure를 반영한 매핑이 필요할 수 있습니다
""")
    
    print_header("파이프라인 성공 완료", 80)


if __name__ == "__main__":
    main()
