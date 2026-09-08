"""Data pipeline 전체 테스트"""

from inmoove.data_pipeline import FaceDataPipeline, FaceDataQualityValidator, MockFaceDataExtractor, DatasetWriter


def test_data_pipeline_runs_with_mock_extractor(tmp_path):
    extractor = MockFaceDataExtractor(frame_count=15, fps=30.0, missing_frame_indices={5, 10})
    writer = DatasetWriter(output_dir=str(tmp_path))
    pipeline = FaceDataPipeline(extractor=extractor, writer=writer, validator=FaceDataQualityValidator())

    result = pipeline.run(
        output_path="mock_face_dataset.jsonl",
        dataset_name="mock_face_dataset.jsonl",
        frame_count=15,
        fps=30.0,
        source_id="mock-test",
    )

    assert len(result.frames) == 15
    assert result.report.total_frames == 15
    assert result.dataset_path.endswith("mock_face_dataset.jsonl")
    assert result.metadata.source == "mock-test"
