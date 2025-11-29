"""
Evaluate fine-tuned eye CNN on test set.

Outputs (in evaluation_results/):
- metrics_<timestamp>.txt  (accuracy, precision, recall, F1, etc.)
- confusion_matrix_<timestamp>.png
- roc_curve_<timestamp>.png
"""

import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

import matplotlib.pyplot as plt

# ---------------- Config ---------------- #

MODEL_PATH = "eye_mobilenet_finetuned_v2.h5"
DATASET_PATH = "/Users/sakethgudi/drowsy_hybrid/dataset"  # change if needed
IMG_SIZE = 160
BATCH_SIZE = 32

RESULTS_DIR = "evaluation_results"

# ---------------------------------------- #

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[INFO] Loading model from {MODEL_PATH} ...")
    model = load_model(MODEL_PATH)
    print(model.summary())

    # Data generator
    test_dir = os.path.join(DATASET_PATH, "test")
    print(f"[INFO] Loading test data from {test_dir}")

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    # Ground truth labels
    y_true = test_gen.classes  # 0/1
    class_indices = test_gen.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}
    print("[INFO] Class index mapping:", class_indices)

    # Predictions
    print("[INFO] Predicting on test set ...")
    y_prob = model.predict(test_gen, verbose=1).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cls_report = classification_report(
        y_true, y_pred, target_names=[idx_to_class[0], idx_to_class[1]]
    )

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    print("\n=== Test Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print("\nClassification Report:\n", cls_report)
    print("Confusion Matrix:\n", cm)

    # ----- Save metrics to TXT ----- #
    metrics_path = os.path.join(RESULTS_DIR, f"metrics_{ts}.txt")
    with open(metrics_path, "w") as f:
        f.write("Eye CNN Evaluation\n")
        f.write(f"Timestamp: {ts}\n\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall   : {rec:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n")
        f.write(f"AUC      : {auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(cls_report + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
    print(f"[INFO] Saved metrics to {metrics_path}")

    # ----- Confusion matrix figure ----- #
    fig_cm, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([idx_to_class[0], idx_to_class[1]])
    ax.set_yticklabels([idx_to_class[0], idx_to_class[1]])

    # add text
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black",
                fontsize=12,
            )

    fig_cm.colorbar(im, ax=ax)
    cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{ts}.png")
    fig_cm.tight_layout()
    fig_cm.savefig(cm_path, dpi=150)
    plt.close(fig_cm)
    print(f"[INFO] Saved confusion matrix to {cm_path}")

    # ----- ROC curve ----- #
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig_roc, ax2 = plt.subplots(figsize=(4, 4))
    ax2.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", label="Random")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    roc_path = os.path.join(RESULTS_DIR, f"roc_curve_{ts}.png")
    fig_roc.tight_layout()
    fig_roc.savefig(roc_path, dpi=150)
    plt.close(fig_roc)
    print(f"[INFO] Saved ROC curve to {roc_path}")

    print("\n[INFO] Evaluation complete.")

if __name__ == "__main__":
    main()
