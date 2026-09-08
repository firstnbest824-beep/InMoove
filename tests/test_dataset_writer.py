"""DatasetWriter 테스트"""

import json

from inmoove.data_pipeline import DatasetWriter, RawFaceFrame


def test_dataset_writer_writes_jsonl(tmp_path):
    writer = DatasetWriter(output_dir=str(tmp_path))
    frames = [
        RawFaceFrame(frame_index=0, timestamp_ms=0, face_detected=True, blendshapes={"jaw_open": 0.1}),
        RawFaceFrame(frame_index=1, timestamp_ms=33, face_detected=True, blendshapes={"jaw_open": 0.2}),
    ]

    output_path = writer.write_jsonl(frames, "mock_dataset.jsonl")
    assert output_path.exists()

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["frame_index"] == 0
    assert payload["blendshapes"]["jaw_open"] == 0.1


def test_dataset_writer_writes_csv(tmp_path):
    writer = DatasetWriter(output_dir=str(tmp_path))
    frames = [
        RawFaceFrame(frame_index=0, timestamp_ms=0, face_detected=True, blendshapes={"jaw_open": 0.1}),
        RawFaceFrame(frame_index=1, timestamp_ms=33, face_detected=True, blendshapes={"jaw_open": 0.2}),
    ]

    output_path = writer.write_csv(frames, "mock_dataset.csv")
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "frame_index" in content
    assert "jaw_open" in content
