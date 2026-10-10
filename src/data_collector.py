
import argparse
import csv
import time
from pathlib import Path

import cv2

from src.hand_detector import HandDetector


DATA_DIR = Path("data/raw")
CSV_PATH = DATA_DIR / "gestures.csv"

GESTURES = [
    "thumbs_up",
    "open_palm",
    "fist",
    "ok",
]


def save_sample(writer, label, points):
    features = [
        coordinate
        for point in points
        for coordinate in point
    ]

    writer.writerow([label] + features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gesture",
        choices=GESTURES,
        required=True,
        help="Gesture label to collect",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=300,
        help="Number of samples to collect",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = CSV_PATH.exists() and CSV_PATH.stat().st_size > 0

    detector = HandDetector()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        detector.close()
        raise RuntimeError("Could not open the camera")

    collected = 0
    recording = False
    last_capture = 0.0
    capture_interval = 0.08

    try:
        with CSV_PATH.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(
                    ["label"]
                    + [
                        f"f{i}"
                        for i in range(63)
                    ]
                )

            while collected < args.samples:
                success, frame = camera.read()

                if not success:
                    break

                frame = cv2.flip(frame, 1)
                hands, results = detector.detect(frame)
                detector.draw(frame, results)

                if recording and hands:
                    now = time.monotonic()

                    if now - last_capture >= capture_interval:
                        save_sample(writer, args.gesture, hands[0])
                        file.flush()

                        collected += 1
                        last_capture = now

                status = "RECORDING" if recording else "PAUSED"

                cv2.putText(
                    frame,
                    f"Gesture: {args.gesture}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"{status} | Samples: {collected}/{args.samples}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    "SPACE: start/pause | Q: quit",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("Gesture Data Collector", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord(" "):
                    recording = not recording
                    last_capture = 0.0

                elif key == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        detector.close()

    print(f"Collected {collected} samples for {args.gesture}")
    print(f"Dataset: {CSV_PATH}")


if __name__ == "__main__":
    main()