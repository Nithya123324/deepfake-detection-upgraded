import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from dataset_utils import collect_video_paths, create_sequence_dataset, split_videos_by_video
from video_model import build_video_deepfake_model, compile_model


DATASET_DIR = "dataset"
FRAMES_PER_VIDEO = 60
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42
MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)


def save_confusion_matrix(y_true, y_pred, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    plt.xticks([0, 1], ["REAL", "FAKE"])
    plt.yticks([0, 1], ["REAL", "FAKE"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_predictions_csv(video_paths, y_true, y_pred, y_prob, save_path: Path):
    df = pd.DataFrame(
        {
            "video_name": [Path(p).name for p in video_paths],
            "actual_label": ["REAL" if int(v) == 1 else "FAKE" for v in y_true],
            "predicted_label": ["REAL" if int(v) == 1 else "FAKE" for v in y_pred],
            "confidence": [float(max(p, 1.0 - p)) for p in y_prob],
        }
    )
    df.to_csv(save_path, index=False)
    return df


def evaluate_model(model, test_dataset, test_video_paths):
    probabilities = model.predict(test_dataset, verbose=1).flatten()
    predictions = (probabilities >= 0.5).astype(int)
    true_labels = []
    for _, y in test_dataset.unbatch().batch(1):
        true_labels.extend(y.numpy().astype(int).tolist())

    if len(true_labels) != len(predictions):
        true_labels = true_labels[: len(predictions)]

    true_labels = np.asarray(true_labels)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)

    metrics = {
        "accuracy": accuracy_score(true_labels, predictions),
        "precision": precision_score(true_labels, predictions, zero_division=0),
        "recall": recall_score(true_labels, predictions, zero_division=0),
        "f1": f1_score(true_labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(true_labels, probabilities),
    }

    save_confusion_matrix(true_labels, predictions, MODEL_DIR / "confusion_matrix.png")
    save_predictions_csv(test_video_paths, true_labels, predictions, probabilities, MODEL_DIR / "test_predictions.csv")

    with open(MODEL_DIR / "metrics_report.json", "w", encoding="utf-8") as f:
        import json
        json.dump(metrics, f, indent=2)

    print("\nEvaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train a video-level deepfake detection model.")
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR)
    parser.add_argument("--frames_per_video", type=int, default=FRAMES_PER_VIDEO)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--test_ratio", type=float, default=TEST_RATIO)
    args = parser.parse_args()

    video_paths, labels, _ = collect_video_paths(args.dataset_dir)
    split_data = split_videos_by_video(
        video_paths,
        labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=RANDOM_STATE,
    )

    train_paths, train_labels = split_data["train"]
    val_paths, val_labels = split_data["val"]
    test_paths, test_labels = split_data["test"]

    print(f"Train videos: {len(train_paths)}")
    print(f"Validation videos: {len(val_paths)}")
    print(f"Test videos: {len(test_paths)}")

    train_dataset = create_sequence_dataset(
        train_paths,
        train_labels,
        frames_per_video=args.frames_per_video,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_dataset = create_sequence_dataset(
        val_paths,
        val_labels,
        frames_per_video=args.frames_per_video,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_dataset = create_sequence_dataset(
        test_paths,
        test_labels,
        frames_per_video=args.frames_per_video,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_video_deepfake_model(frames_per_video=args.frames_per_video)
    model = compile_model(model, learning_rate=args.learning_rate)

    checkpoint_path = MODEL_DIR / "best_video_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=6,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=3,
            factor=0.5,
            min_lr=1e-7,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nTraining finished.")
    print(f"Best model saved to: {checkpoint_path}")

    evaluate_model(model, test_dataset, test_paths)


if __name__ == "__main__":
    main()
