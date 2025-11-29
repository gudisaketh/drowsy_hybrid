"""
Template to evaluate the EAR+CNN hybrid system.

Expected data format (you can change this):
- A CSV file with rows:
    image_path, label
  where:
    image_path -> path to a frame image (with full face)
    label      -> 0 for AWAKE, 1 for DROWSY (or 'awake'/'drowsy')

You must:
- Generate such frames (e.g., from videos) and labels separately.
- Update CSV_PATH and ROOT_IMG_DIR.
"""

import os
import csv
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ---------- Config (edit these) ---------- #

MODEL_PATH = "eye_mobilenet_finetuned_v2.h5"
CSV_PATH = "hybrid_eval_labels.csv"     # your annotated file
ROOT_IMG_DIR = "hybrid_eval_frames"     # prefix for relative paths in CSV
IMG_SIZE = 160

EAR_THRESH = 0.23
CNN_CLOSED_THRESH = 0.5

RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]

# ---------------------------------------- #

def eye_aspect_ratio(eye_pts):
    p1, p2, p3, p4, p5, p6 = eye_pts
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)
    return (A + B) / (2.0 * C + 1e-6)

def crop_eye(frame, eye_pts, margin=0.35):
    h, w, _ = frame.shape
    xs = eye_pts[:, 0]; ys = eye_pts[:, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    bw, bh = x_max - x_min, y_max - y_min
    x_min = max(int(x_min - margin * bw), 0)
    x_max = min(int(x_max + margin * bw), w)
    y_min = max(int(y_min - margin * bh), 0)
    y_max = min(int(y_max + margin * bh), h)
    return frame[y_min:y_max, x_min:x_max]

def preprocess_eye(img):
    if img is None or img.size == 0:
        return None
    eye = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)
    eye = eye.astype("float32") / 255.0
    return np.expand_dims(eye, axis=0)

def predict_eye_closed(model, eye_img):
    inp = preprocess_eye(eye_img)
    if inp is None:
        return None
    prob = model.predict(inp, verbose=0)[0][0]
    return float(prob)

def hybrid_decision(ear, cnn_prob_closed):
    ear_closed = ear < EAR_THRESH
    if cnn_prob_closed is None:
        return ear_closed
    cnn_closed = cnn_prob_closed > CNN_CLOSED_THRESH
    return ear_closed and cnn_closed  # AND fusion

def main():
    print("[INFO] Loading model ...")
    model = load_model(MODEL_PATH)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    y_true = []
    y_pred = []

    with open(CSV_PATH, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # assume header row "image_path,label"
        for row in reader:
            rel_path, label_raw = row[0], row[1]
            label = 1 if str(label_raw).lower() in ["1", "drowsy"] else 0
            img_path = os.path.join(ROOT_IMG_DIR, rel_path)

            frame = cv2.imread(img_path)
            if frame is None:
                print(f"[WARN] Could not read {img_path}, skipping.")
                continue

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                # no face detected; you might choose to skip or treat as awake
                continue

            landmarks = np.array(
                [[lm.x * w, lm.y * h] for lm in results.multi_face_landmarks[0].landmark]
            )
            right_eye_pts = landmarks[RIGHT_EYE_IDX]
            left_eye_pts = landmarks[LEFT_EYE_IDX]

            right_ear = eye_aspect_ratio(right_eye_pts)
            left_ear = eye_aspect_ratio(left_eye_pts)
            mean_ear = (right_ear + left_ear) / 2.0

            right_eye_img = crop_eye(frame, right_eye_pts)
            left_eye_img = crop_eye(frame, left_eye_pts)
            right_prob = predict_eye_closed(model, right_eye_img)
            left_prob = predict_eye_closed(model, left_eye_img)

            right_closed = hybrid_decision(right_ear, right_prob)
            left_closed = hybrid_decision(left_ear, left_prob)
            both_closed = right_closed and left_closed

            # Map to awake/drowsy label: here we say closed => drowsy
            pred_label = 1 if both_closed else 0

            y_true.append(label)
            y_pred.append(pred_label)

    if not y_true:
        print("[ERROR] No samples evaluated. Check CSV_PATH and ROOT_IMG_DIR.")
        return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["AWAKE", "DROWSY"])

    print("\n=== Hybrid System Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

if __name__ == "__main__":
    main()
