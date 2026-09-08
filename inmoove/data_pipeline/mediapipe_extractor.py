"""
실제 MediaPipe 기반 얼굴 데이터 추출기

이 구현은 실제 video를 받아 MediaPipe Face Landmarker를 사용해
blendshape 시퀀스를 생성하는 역할을 합니다.

중요:
- 여기서는 실제 영상/모델이 없더라도 import 자체는 가능해야 합니다
- model 파일이 없으면 이해하기 쉬운 FileNotFoundError를 발생시킵니다
- 전체 테스트 suite는 Mock extractor 중심으로 동작해야 합니다
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from .extractor import FaceDataExtractor
from .models import RawFaceFrame


class MediaPipeFaceDataExtractor(FaceDataExtractor):
    """
    MediaPipe Face Landmarker를 사용하는 추출기입니다.

    실제 영상이 준비되면 다음 흐름으로 동작할 수 있습니다.
    - OpenCV로 비디오 프레임 읽기
    - MediaPipe Face Landmarker로 face blendshape 계산
    - RawFaceFrame 생성
    """

    def __init__(
        self,
        model_path: str,
        num_faces: int = 1,
        include_landmarks: bool = False,
    ) -> None:
        self.model_path = model_path
        self.num_faces = num_faces
        self.include_landmarks = include_landmarks

    def extract(self, **kwargs: Any) -> List[RawFaceFrame]:
        """
        실제 MediaPipe를 사용해 비디오에서 얼굴 프레임을 추출합니다.

        Args:
            video_path: 처리할 비디오 파일 경로
            max_frames: 최대 처리 프레임 수
            fps: 가정 fps. None이면 OpenCV에서 자동 추정
            source_id: 데이터 소스 ID

        Returns:
            List[RawFaceFrame]: 추출된 얼굴 프레임 리스트
        """
        video_path = kwargs.get("video_path")
        if video_path is None:
            raise ValueError("video_path가 필요합니다")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"MediaPipe model file not found: {self.model_path}. "
                "모델 파일을 지정하거나 실제 영상/모델을 준비한 뒤 사용하십시오."
            )

        try:
            import cv2
            import mediapipe as mp
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MediaPipe 또는 OpenCV가 설치되지 않았습니다. "
                "pip install mediapipe opencv-python 를 먼저 수행하세요."
            ) from exc

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"비디오를 열 수 없습니다: {video_path}")

        fps = kwargs.get("fps")
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0

        max_frames = kwargs.get("max_frames")
        source_id = kwargs.get("source_id", os.path.basename(video_path))

        base_options = mp.tasks.BaseOptions(model_asset_path=self.model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=self.num_faces,
            output_face_blendshapes=True,
        )

        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            frames: List[RawFaceFrame] = []
            frame_index = 0
            timestamp_ms = 0

            while True:
                ok, bgr_frame = cap.read()
                if not ok:
                    break
                if max_frames is not None and frame_index >= max_frames:
                    break

                rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                detection = landmarker.detect_for_video(image, timestamp_ms)

                blendshapes: dict[str, float] = {}
                if detection.face_blendshapes:
                    # MediaPipe 결과는 보통 한 사람당 여러 blendshape 카테고리 리스트를 담고 있습니다.
                    # 구조가 약간 다를 수 있으므로 안전하게 파싱합니다.
                    for face_blendshapes in detection.face_blendshapes:
                        if not isinstance(face_blendshapes, (list, tuple)):
                            entries = [face_blendshapes]
                        else:
                            entries = face_blendshapes
                        for item in entries:
                            category_name = getattr(item, "category_name", None)
                            score = getattr(item, "score", None)
                            if category_name is None or score is None:
                                continue
                            normalized_name = str(category_name).lower().replace(" ", "_")
                            numeric_score = float(score)
                            blendshapes[normalized_name] = max(0.0, min(1.0, numeric_score))

                face_detected = bool(detection.face_blendshapes)
                landmarks = None
                if self.include_landmarks and detection.face_landmarks:
                    landmarks = {
                        "face_landmarks": detection.face_landmarks,
                    }

                frames.append(
                    RawFaceFrame(
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        face_detected=face_detected,
                        blendshapes=blendshapes,
                        landmarks=landmarks,
                        source_id=source_id,
                    )
                )

                frame_index += 1
                timestamp_ms = int((1000.0 * frame_index) / fps)

            cap.release()
            return frames
