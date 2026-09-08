"""
데이터 세트 writer

추출된 얼굴 프레임을 JSONL 또는 CSV 형식으로 저장합니다.

기본 형식은 JSONL이며, 이유는 다음과 같습니다.
- 프레임 단위 streaming 저장이 쉽다
- dict 구조를 그대로 저장할 수 있다
- 대규모 데이터 처리에 적합하다
- 향후 음악/자막/시간축 정보와 결합이 쉬움
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .models import DatasetMetadata, RawFaceFrame


class DatasetWriter:
    """
    RawFaceFrame 시퀀스를 파일로 기록하는 클래스입니다.
    """

    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(
        self,
        frames: Sequence[RawFaceFrame],
        filename: str,
        metadata: Optional[DatasetMetadata] = None,
    ) -> Path:
        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as handle:
            for frame in frames:
                handle.write(json.dumps(frame.to_dict(), ensure_ascii=False))
                handle.write("\n")

        if metadata is not None:
            meta_path = path.with_suffix(".meta.json")
            with meta_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata.to_dict(), handle, ensure_ascii=False, indent=2)

        return path

    def write_csv(
        self,
        frames: Sequence[RawFaceFrame],
        filename: str,
        metadata: Optional[DatasetMetadata] = None,
    ) -> Path:
        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["frame_index", "timestamp_ms", "face_detected"]
        all_keys = sorted({key for frame in frames for key in frame.blendshapes.keys()})
        fieldnames.extend(all_keys)

        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for frame in frames:
                row = {
                    "frame_index": frame.frame_index,
                    "timestamp_ms": frame.timestamp_ms,
                    "face_detected": frame.face_detected,
                }
                for key in all_keys:
                    row[key] = frame.blendshapes.get(key, 0.0)
                writer.writerow(row)

        if metadata is not None:
            meta_path = path.with_suffix(".meta.json")
            with meta_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata.to_dict(), handle, ensure_ascii=False, indent=2)

        return path
