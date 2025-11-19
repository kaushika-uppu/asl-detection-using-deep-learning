#!/usr/bin/env python3
"""Train and evaluate an LSTM classifier on preprocessed WLASL landmark sequences."""

from __future__ import annotations

import argparse
import os
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM model on WLASL landmark sequences and report accuracy.")
    parser.add_argument("--data", default="wlasl_landmarks.npz", help=".npz file produced by wlasl_preprocess.py")
    parser.add_argument("--model-out", default="wlasl_sequence_model.keras", help="Path to save trained model")
    parser.add_argument("--labels-out", default="wlasl_labels.npy", help="Numpy file for the label order")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data reserved for evaluation")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument(
        "--lstm-units",
        type=int,
        nargs=2,
        default=[128, 64],
        metavar=("U1", "U2"),
        help="Hidden units for the stacked LSTM layers",
    )
    parser.add_argument("--dense-units", type=int, default=64, help="Units in the dense layer before softmax")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate applied after each LSTM layer")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate for Adam")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on validation accuracy")
    return parser.parse_args()


def build_model(
    input_shape: tuple[int, int],
    num_classes: int,
    lstm_units: Sequence[int],
    dense_units: int,
    dropout: float,
    learning_rate: float,
) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Masking(mask_value=0.0),
        layers.LSTM(lstm_units[0], return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(lstm_units[1]),
        layers.Dropout(dropout),
        layers.Dense(dense_units, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found at {args.data}. Run wlasl_preprocess.py first.")

    keras.utils.set_random_seed(args.random_state)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    data = np.load(args.data)
    sequences = data["sequences"].astype(np.float32)
    labels = data["labels"]

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    np.save(args.labels_out, label_encoder.classes_)
    print(f"Saved label order to {args.labels_out} ({len(label_encoder.classes_)} classes)")

    X_train, X_test, y_train, y_test = train_test_split(
        sequences,
        encoded_labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=encoded_labels,
    )

    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(label_encoder.classes_)

    print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]} | Sequence length: {input_shape[0]}")

    model = build_model(
        input_shape=input_shape,
        num_classes=num_classes,
        lstm_units=args.lstm_units,
        dense_units=args.dense_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=args.patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(args.model_out, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)

    # Ensure final weights saved even if early stopping triggered before best checkpoint
    model.save(args.model_out)
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
