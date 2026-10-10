
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.dataset import prepare_data
from src.classifier import GestureClassifier


# Configuration
MODEL_PATH = Path("models/gesture_model.pth")
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the test dataset
_, _, test_loader, encoder, _ = prepare_data()

# Load the saved checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

class_names = checkpoint["class_names"]

# Recreate the model architecture and load its learned weights
model = GestureClassifier(
    input_size=checkpoint["input_size"],
    num_classes=checkpoint["num_classes"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Collect predictions and actual labels
all_predictions = []
all_labels = []

with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)

        outputs = model(features)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_predictions)

print("\n========== TEST RESULTS ==========")
print(f"Test accuracy: {accuracy:.2%}")

print("\nClassification report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
)

# Build and display the confusion matrix
matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names))),
)

print("Confusion matrix:")
print(matrix)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=class_names,
)

fig, ax = plt.subplots(figsize=(8, 6))
display.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Gesture Classification - Test Set")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.close()

print(f"\nConfusion matrix saved to: {OUTPUT_DIR / 'confusion_matrix.png'}")