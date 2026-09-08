"""
MockHead 간단한 데모

실제 하드웨어 없이 i2Head 표정 제어를 테스트합니다.

실행: python examples/simple_demo.py
"""

from inmoove.face import FaceExpression, MockHead


def main():
    """기본 데모 실행"""
    
    print("\n" + "="*60)
    print("InMoove i2Head MockHead 데모")
    print("="*60)
    
    # MockHead 생성 (verbose=True로 설정하여 상세 정보 출력)
    head = MockHead(verbose=True)
    
    # 1. 중립 표정
    print("\n[1단계] 중립 표정 설정")
    print("-" * 60)
    neutral = FaceExpression.neutral()
    head.set_expression(neutral)
    
    # 2. 웃음 표정
    print("\n[2단계] 웃음 표정 설정")
    print("-" * 60)
    smile = FaceExpression(
        eye_left_lr=0.5,
        eye_left_ud=0.6,
        eye_right_lr=0.5,
        eye_right_ud=0.6,
        eyelid_left_upper=0.85,
        eyelid_left_lower=0.15,
        eyelid_right_upper=0.85,
        eyelid_right_lower=0.15,
        eyebrow_left=0.6,
        eyebrow_right=0.6,
        cheek_left=0.75,
        cheek_right=0.75,
        forehead_left=0.4,
        forehead_right=0.4,
        upper_lip=0.7,
        jaw=0.3,
    )
    head.set_expression(smile)
    
    # 3. 슬픈 표정
    print("\n[3단계] 슬픈 표정 설정")
    print("-" * 60)
    sad = FaceExpression(
        eye_left_lr=0.5,
        eye_left_ud=0.3,
        eye_right_lr=0.5,
        eye_right_ud=0.3,
        eyelid_left_upper=0.6,
        eyelid_left_lower=0.4,
        eyelid_right_upper=0.6,
        eyelid_right_lower=0.4,
        eyebrow_left=0.3,
        eyebrow_right=0.3,
        cheek_left=0.2,
        cheek_right=0.2,
        forehead_left=0.7,
        forehead_right=0.7,
        upper_lip=0.3,
        jaw=0.2,
    )
    head.set_expression(sad)
    
    # 4. 놀란 표정
    print("\n[4단계] 놀란 표정 설정")
    print("-" * 60)
    surprised = FaceExpression(
        eye_left_lr=0.5,
        eye_left_ud=0.8,
        eye_right_lr=0.5,
        eye_right_ud=0.8,
        eyelid_left_upper=1.0,
        eyelid_left_lower=0.0,
        eyelid_right_upper=1.0,
        eyelid_right_lower=0.0,
        eyebrow_left=0.9,
        eyebrow_right=0.9,
        cheek_left=0.5,
        cheek_right=0.5,
        forehead_left=0.8,
        forehead_right=0.8,
        upper_lip=0.8,
        jaw=0.6,
    )
    head.set_expression(surprised)
    
    # 5. Rest 동작
    print("\n[5단계] Rest 동작 (중립 표정으로 복귀)")
    print("-" * 60)
    head.rest()
    
    # 호출 이력 출력
    print("\n" + "="*60)
    print("표정 변화 이력")
    print("="*60)
    history = head.get_expression_history()
    for i, entry in enumerate(history, 1):
        timestamp = entry["timestamp"]
        expr = entry["expression"]
        
        expression_types = ["중립", "웃음", "슬픔", "놀람", "중립(rest)"]
        expr_type = expression_types[i-1] if i <= len(expression_types) else "알 수 없음"
        
        print(f"{i}. [{timestamp}] {expr_type}")
        print(f"   jaw: {expr.jaw:.2f}, eyebrow_left: {expr.eyebrow_left:.2f}")
    
    print("\n" + "="*60)
    print("데모 완료!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
