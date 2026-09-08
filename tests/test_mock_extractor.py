"""MockFaceDataExtractor 테스트"""

from inmoove.data_pipeline import MockFaceDataExtractor


def test_mock_extractor_generates_expected_length():
    extractor = MockFaceDataExtractor(frame_count=12, fps=30.0, missing_frame_indices={5, 9})
    frames = extractor.extract()
    assert len(frames) == 12


def test_mock_extractor_maintains_monotonic_timestamp():
    extractor = MockFaceDataExtractor(frame_count=20, fps=30.0)
    frames = extractor.extract()
    timestamps = [frame.timestamp_ms for frame in frames]
    assert timestamps == sorted(timestamps)
    assert frames[0].timestamp_ms == 0


def test_mock_extractor_marks_missing_frames_as_not_detected():
    extractor = MockFaceDataExtractor(frame_count=10, fps=30.0, missing_frame_indices={3, 7})
    frames = extractor.extract()
    assert frames[3].face_detected is False
    assert frames[7].face_detected is False
    assert frames[2].face_detected is True


def test_mock_extractor_produces_blendshapes_in_range():
    extractor = MockFaceDataExtractor(frame_count=10, fps=30.0)
    frames = extractor.extract()
    for frame in frames:
        if frame.face_detected:
            for value in frame.blendshapes.values():
                assert 0.0 <= value <= 1.0
