
import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch

from src.classifier import GestureClassifier
from src.hand_detector import HandDetector
from src.project_paths import project_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run real-time hand gesture recognition.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index to use.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(project_path("models", "gesture_model.pth")),
        help="Path to the trained gesture model.",
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default=str(project_path("models", "scaler.joblib")),
        help="Path to the trained feature scaler.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.60,
        help="Minimum model confidence needed to accept a prediction.",
    )
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_path(path_value)


def main():
    args = parse_args()
    model_path = resolve_path(args.model_path)
    scaler_path = resolve_path(args.scaler_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    class_names = checkpoint["class_names"]

    model = GestureClassifier(
        input_size=checkpoint["input_size"],
        num_classes=checkpoint["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = joblib.load(scaler_path)
    print("Loaded gesture classes:", class_names)

    detector = HandDetector()
    cap = cv2.VideoCapture(args.camera_index)

    if not cap.isOpened():
        print("Could not open the camera.")
        detector.close()
        raise SystemExit(1)

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read a camera frame.")
                break

            frame = cv2.flip(frame, 1)
            hands, results = detector.detect(frame)
            detector.draw(frame, results)

            gesture_text = "No hand detected"
            confidence_text = ""

            if len(hands) > 0:
                points = np.asarray(hands[0], dtype=np.float32).reshape(1, -1)

                if points.shape[1] == checkpoint["input_size"]:
                    scaled_points = scaler.transform(points)
                    features = torch.tensor(scaled_points, dtype=torch.float32, device=device)

                    with torch.no_grad():
                        outputs = model(features)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidence, predicted_index = probabilities.max(dim=1)

                    confidence = confidence.item()
                    predicted_index = predicted_index.item()

                    if confidence >= args.confidence_threshold:
                        gesture_text = class_names[predicted_index]
                        confidence_text = f"Confidence: {confidence:.1%}"
                    else:
                        gesture_text = "Unknown gesture"
                        confidence_text = f"Confidence: {confidence:.1%}"
                else:
                    gesture_text = "Invalid landmark data"

            cv2.rectangle(frame, (10, 10), (430, 100), (30, 30, 30), -1)
            cv2.putText(frame, gesture_text, (25, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, confidence_text, (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, "Press Q to quit", (15, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Hand Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()