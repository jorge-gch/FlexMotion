
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


DATA_PATH = Path("data/raw/gestures.csv")
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("The dataset is empty.")

    if df.isnull().any().any():
        raise ValueError("The dataset contains missing values.")

    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    y_text = df["label"].to_numpy()

    if X.shape[1] != 63:
        raise ValueError(f"Expected 63 features, got {X.shape[1]}.")

    if not np.isfinite(X).all():
        raise ValueError("The dataset contains invalid numeric values.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    return X, y, encoder


def prepare_data(batch_size=32):
    X, y, encoder = load_data()

    # First reserve 20% for the final test.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Reserve 20% of the remaining data for validation.
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )

    # Fit preprocessing only on training data.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    def make_loader(features, labels, shuffle=False):
        dataset = TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val)
    test_loader = make_loader(X_test, y_test)

    return (
        train_loader,
        val_loader,
        test_loader,
        encoder,
        scaler,
    )


if __name__ == "__main__":
    train_loader, val_loader, test_loader, encoder, scaler = (
        prepare_data()
    )

    print("Classes:", list(encoder.classes_))
    print("Training examples:", len(train_loader.dataset))
    print("Validation examples:", len(val_loader.dataset))
    print("Test examples:", len(test_loader.dataset))

    X_batch, y_batch = next(iter(train_loader))
    print("Batch features:", X_batch.shape)
    print("Batch labels:", y_batch.shape)