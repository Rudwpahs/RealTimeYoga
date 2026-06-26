import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from app_paths import resource_path

MODEL_PATH = Path(resource_path("pose_landmarker_lite.task"))


class poseDetector:
    def __init__(self, mode=False, complexity=1, landmarks=True, enable_seg=False,
                 smooth_seg=True, det_conf=0.5, track_conf=0.5):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing MediaPipe pose model: {MODEL_PATH}")

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=det_conf,
            min_pose_presence_confidence=det_conf,
            min_tracking_confidence=track_conf,
            output_segmentation_masks=enable_seg,
        )
        self.pose = vision.PoseLandmarker.create_from_options(options)
        self.results = None
        self._last_timestamp_ms = 0

    def findPose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        self.results = self.pose.detect_for_video(mp_image, timestamp_ms)

        if draw and self.results.pose_landmarks:
            vision.drawing_utils.draw_landmarks(
                img,
                self.results.pose_landmarks[0],
                vision.PoseLandmarksConnections.POSE_LANDMARKS,
            )

        return img

    def findPosition(self, img, draw=True):
        lmlist = []
        if self.results and self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks[0]):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmlist

    def close(self):
        self.pose.close()
