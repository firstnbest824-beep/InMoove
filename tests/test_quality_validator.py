"""FaceDataQualityValidator 테스트"""

from inmoove.data_pipeline import MockFaceDataExtractor, FaceDataQualityValidator, RawFaceFrame


def test_validator_scores_clean_data_as_a_grade():
    frames = MockFaceDataExtractor(frame_count=50, fps=30.0).extract()
    report = FaceDataQualityValidator().validate(frames)
    assert report.total_frames == 50
    assert report.detection_rate > 0.95
    assert report.grade == "A"


def test_validator_detects_timestamp_order_issue():
    frames = [
        RawFaceFrame(frame_index=0, timestamp_ms=0, face_detected=True, blendshapes={"jaw_open": 0.1}),
        RawFaceFrame(frame_index=1, timestamp_ms=50, face_detected=True, blendshapes={"jaw_open": 0.2}),
        RawFaceFrame(frame_index=2, timestamp_ms=20, face_detected=True, blendshapes={"jaw_open": 0.3}),
    ]
    report = FaceDataQualityValidator().validate(frames)
    assert report.invalid_timestamp_count >= 1


def test_validator_detects_invalid_values():
    frames = [
        RawFaceFrame(frame_index=0, timestamp_ms=0, face_detected=True, blendshapes={"jaw_open": 0.5}),
        RawFaceFrame(frame_index=1, timestamp_ms=33, face_detected=True, blendshapes={"jaw_open": 0.7}),
    ]
    frames[1].blendshapes["jaw_open"] = 1.5
    report = FaceDataQualityValidator().validate(frames)
    assert report.invalid_value_count >= 1


def test_validator_counts_longest_missing_run():
    frames = [
        RawFaceFrame(frame_index=0, timestamp_ms=0, face_detected=True, blendshapes={"jaw_open": 0.1}),
        RawFaceFrame(frame_index=1, timestamp_ms=33, face_detected=False, blendshapes={}),
        RawFaceFrame(frame_index=2, timestamp_ms=66, face_detected=False, blendshapes={}),
        RawFaceFrame(frame_index=3, timestamp_ms=99, face_detected=True, blendshapes={"jaw_open": 0.2}),
    ]
    report = FaceDataQualityValidator().validate(frames)
    assert report.longest_missing_run == 2
