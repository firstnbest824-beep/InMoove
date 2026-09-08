"""RawFaceFrame 테스트"""

import math

import pytest

from inmoove.data_pipeline import RawFaceFrame


def test_raw_face_frame_valid_creation():
    frame = RawFaceFrame(
        frame_index=0,
        timestamp_ms=0,
        face_detected=True,
        blendshapes={
            "jaw_open": 0.3,
            "mouth_smile_left": 0.7,
            "mouth_smile_right": 0.75,
        },
    )
    assert frame.frame_index == 0
    assert frame.face_detected is True
    assert frame.blendshapes["jaw_open"] == 0.3


def test_raw_face_frame_allows_zero_and_one():
    frame = RawFaceFrame(
        frame_index=1,
        timestamp_ms=33,
        face_detected=True,
        blendshapes={
            "eye_blink_left": 0.0,
            "eye_blink_right": 1.0,
        },
    )
    assert frame.blendshapes["eye_blink_left"] == 0.0
    assert frame.blendshapes["eye_blink_right"] == 1.0


def test_raw_face_frame_rejects_negative_frame_index():
    with pytest.raises(ValueError, match="frame_index"):
        RawFaceFrame(frame_index=-1, timestamp_ms=0, face_detected=True)


def test_raw_face_frame_rejects_negative_timestamp():
    with pytest.raises(ValueError, match="timestamp_ms"):
        RawFaceFrame(frame_index=0, timestamp_ms=-1, face_detected=True)


def test_raw_face_frame_rejects_values_below_zero():
    with pytest.raises(ValueError, match="범위"):
        RawFaceFrame(
            frame_index=0,
            timestamp_ms=0,
            face_detected=True,
            blendshapes={"jaw_open": -0.1},
        )


def test_raw_face_frame_rejects_values_above_one():
    with pytest.raises(ValueError, match="범위"):
        RawFaceFrame(
            frame_index=0,
            timestamp_ms=0,
            face_detected=True,
            blendshapes={"jaw_open": 1.2},
        )


def test_raw_face_frame_rejects_nan():
    with pytest.raises(ValueError, match="유한"):
        RawFaceFrame(
            frame_index=0,
            timestamp_ms=0,
            face_detected=True,
            blendshapes={"jaw_open": float("nan")},
        )


def test_raw_face_frame_rejects_inf():
    with pytest.raises(ValueError, match="유한"):
        RawFaceFrame(
            frame_index=0,
            timestamp_ms=0,
            face_detected=True,
            blendshapes={"jaw_open": float("inf")},
        )


def test_face_detected_false_can_have_empty_blendshapes():
    frame = RawFaceFrame(frame_index=10, timestamp_ms=333, face_detected=False, blendshapes={})
    assert frame.face_detected is False
    assert frame.blendshapes == {}
