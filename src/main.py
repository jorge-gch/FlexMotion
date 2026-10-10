
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch

from src.hand_detector import HandDetector
from src.classifier import GestureClassifier



# Configuration
MODEL_PATH = Path("models/gesture_model.pth")
SCALER_PATH = Path("models/scaler.joblib")
CONFIDENCE_THRESHOLD = 0.60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Load trained model
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

class_names = checkpoint["class_names"]

model = GestureClassifier(
    input_size=checkpoint["input_size"],
    num_classes=checkpoint["num_classes"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

scaler = joblib.load(SCALER_PATH)

print("Loaded gesture classes:", class_names)



# Initialize detector
detector = HandDetector()
cap = cv2.VideoCapture(0)

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

        # Mirror the image
        frame = cv2.flip(frame, 1)

        # Detect hands and draw landmarks
        hands, results = detector.detect(frame)
        detector.draw(frame, results)

        gesture_text = "No hand detected"
        confidence_text = ""

        if len(hands) > 0:
            # Use the first detected hand
            points = np.asarray(
                hands[0],
                dtype=np.float32
            ).reshape(1, -1)

            if points.shape[1] == checkpoint["input_size"]:
                # Apply the same scaling used during training
                scaled_points = scaler.transform(points)

                # Convert features to a PyTorch tensor
                features = torch.tensor(
                    scaled_points,
                    dtype=torch.float32,
                    device=device,
                )

                # Predict gesture
                with torch.no_grad():
                    outputs = model(features)
                    probabilities = torch.softmax(outputs, dim=1)

                    confidence, predicted_index = probabilities.max(dim=1)

                confidence = confidence.item()
                predicted_index = predicted_index.item()

                if confidence >= CONFIDENCE_THRESHOLD:
                    gesture_text = class_names[predicted_index]
                    confidence_text = f"Confidence: {confidence:.1%}"
                else:
                    gesture_text = "Unknown gesture"
                    confidence_text = f"Confidence: {confidence:.1%}"

            else:
                gesture_text = "Invalid landmark data"

        # Display prediction
        cv2.rectangle(frame, (10, 10), (430, 100), (30, 30, 30), -1)

        cv2.putText(
            frame,
            gesture_text,
            (25, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            confidence_text,
            (25, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (15, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        cv2.imshow("Hand Gesture Recognition", frame)

        # Exit when Q is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    detector.close()