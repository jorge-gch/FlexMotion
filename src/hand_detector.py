
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Connections between the 21 hand landmarks.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class HandDetector:
    def __init__(
        self,
        model_path="models/hand_landmarker.task",
        max_hands=1,
        detection_confidence=0.5,
    ):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=model_path
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.detector = vision.HandLandmarker.create_from_options(
            options
        )

        self.start_time = time.monotonic()
        self.last_timestamp = -1

    def detect(self, frame):
        # Convert OpenCV BGR images to RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        # VIDEO mode requires increasing timestamps.
        timestamp = int(
            (time.monotonic() - self.start_time) * 1000
        )
        timestamp = max(timestamp, self.last_timestamp + 1)
        self.last_timestamp = timestamp

        results = self.detector.detect_for_video(
            mp_image,
            timestamp,
        )

        hands = []

        for landmarks in results.hand_landmarks:
            points = [
                [point.x, point.y, point.z]
                for point in landmarks
            ]
            hands.append(points)

        return hands, results

    def draw(self, frame, results):
        height, width = frame.shape[:2]

        for landmarks in results.hand_landmarks:
            points = [
                (int(point.x * width), int(point.y * height))
                for point in landmarks
            ]

            for start, end in HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    2,
                )

            for x, y in points:
                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 0, 255),
                    -1,
                )

        return frame

    def close(self):
        self.detector.close()