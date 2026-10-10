
from pathlib import Path
import json

import matplotlib.pyplot as plt
import torch
from torch import nn

from src.dataset import prepare_data
from src.classifier import GestureClassifier


# Configuration
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 12
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the datasets
train_loader, val_loader, test_loader, encoder, scaler = prepare_data()

num_classes = len(encoder.classes_)

print("Classes:", list(encoder.classes_))
print("Training samples:", len(train_loader.dataset))
print("Validation samples:", len(val_loader.dataset))
print("Test samples:", len(test_loader.dataset))

# Create the model
model = GestureClassifier(num_classes=num_classes).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# Training history
history = {
    "train_loss": [],
    "val_loss": [],
    "val_accuracy": []
}

best_val_loss = float("inf")
patience_counter = 0

checkpoint_path = MODEL_DIR / "gesture_model.pth"

# Training loop
for epoch in range(EPOCHS):

    # Training
    model.train()
    train_loss_total = 0.0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss_total += loss.item() * features.size(0)

    train_loss = train_loss_total / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss_total = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * features.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_loss = val_loss_total / len(val_loader.dataset)
    val_accuracy = correct / total

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    print(
        f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2%}"
    )

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": encoder.classes_.tolist(),
                "input_size": 63,
                "num_classes": num_classes,
            },
            checkpoint_path
        )

        print("  Best model saved.")

    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= PATIENCE:
        print("Early stopping: validation loss did not improve.")
        break


# Save the scaler used to normalize the features
import joblib

joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

# Save training history
with open(MODEL_DIR / "training_history.json", "w") as file:
    json.dump(history, file, indent=4)

# Plot training curves
plt.figure(figsize=(10, 5))
plt.plot(history["train_loss"], label="Training loss")
plt.plot(history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(MODEL_DIR / "training_curves.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(MODEL_DIR / "validation_accuracy.png")
plt.close()

print("\nTraining finished.")
print(f"Best model: {checkpoint_path}")
print(f"Scaler: {MODEL_DIR / 'scaler.joblib'}")
print(f"Training curves: {MODEL_DIR / 'training_curves.png'}")