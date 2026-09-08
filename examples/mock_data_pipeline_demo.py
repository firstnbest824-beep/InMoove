"""
Mock 데이터 파이프라인 데모

이 데모는 실제 영상이 없더라도 전체 데이터 파이프라인이 동작하는지 보여줍니다.

실행:
    PYTHONPATH=. python examples/mock_data_pipeline_demo.py
"""

from inmoove.data_pipeline import DatasetWriter, FaceDataPipeline, FaceDataQualityValidator, MockFaceDataExtractor


def main() -> None:
    """Mock 입력으로 전체 수집 파이프라인을 실행합니다."""
    extractor = MockFaceDataExtractor(frame_count=90, fps=30.0, missing_frame_indices={12, 28, 58})
    writer = DatasetWriter(output_dir="output")
    validator = FaceDataQualityValidator()
    pipeline = FaceDataPipeline(extractor=extractor, writer=writer, validator=validator)

    result = pipeline.run(
        output_path="output/mock_face_dataset.jsonl",
        dataset_name="mock_face_dataset.jsonl",
        frame_count=90,
        fps=30.0,
        source_id="mock-demo-video",
    )

    print("[Mock Face Data Pipeline]")
    print(f"Generated frames   : {result.report.total_frames}")
    print(f"Detected frames    : {result.report.detected_frames}")
    print(f"Detection rate     : {result.report.detection_rate:.2%}")
    print(f"Longest missing    : {result.report.longest_missing_run} frames")
    print(f"Quality grade     : {result.report.grade}")
    print(f"Dataset file      : {result.dataset_path}")
    print(f"Metadata file     : {result.dataset_path[:-6] + '.meta.json'}")


if __name__ == "__main__":
    main()
