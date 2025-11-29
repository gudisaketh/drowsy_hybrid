"""
Hybrid Driver Drowsiness Detection UI (EAR or Hybrid EAR+CNN)
--------------------------------------------------------------

- Modes:
    * EAR only
    * Hybrid (EAR + CNN eye classifier)

- Features:
    * Auto-calibrated EAR threshold (with manual slider override)
    * Single-eye fallback for CNN (works if one eye is occluded)
    * Robust hybrid fusion (EAR + CNN)
    * Session analytics:
        - EAR over time plot
        - EAR distribution histogram (frequency vs EAR)
        - Time distribution pie chart
        - Detailed stats table
        - Log file path shown in stats page
    * Outputs:
        - PNGs: piechart, earplot, ear_hist, session_stats
        - CSV log file
        - PDF report including stats, pie, EAR curve, histogram
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import subprocess
import os
import csv
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages

import simpleaudio as sa
import tensorflow as tf
from tensorflow.keras.models import load_model


MODEL_PATH = "eye_mobilenet_finetuned_v2.h5"
IMG_SIZE = 160

# We assume the CNN outputs P(open).
# So we convert to P(closed) = 1 - P(open).
CNN_CLOSED_THRESH = 0.6         # threshold for saying "CNN thinks eye is closed"
HYBRID_HISTORY_LEN = 30         # number of frames in sliding window
HYBRID_DROWSY_RATIO = 0.5       # >50% closed in last N frames => drowsy


# -------------------- Theme detection -------------------- #

def detect_theme():
    is_dark = False
    try:
        out = subprocess.check_output(
            ["defaults", "read", "-g", "AppleInterfaceStyle"],
            stderr=subprocess.STDOUT
        ).decode().strip()
        if out.lower() == "dark":
            is_dark = True
    except Exception:
        is_dark = False

    if is_dark:
        return {
            "bg": "#202124",
            "panel_bg": "#303134",
            "fg": "#FFFFFF",
            "accent": "#0B84FF",
            "btn_start": "#34C759",
            "btn_stop": "#FF3B30",
            "entry_bg": "#3C4043",
            "entry_fg": "#FFFFFF"
        }
    else:
        return {
            "bg": "#F5F5F5",
            "panel_bg": "#FFFFFF",
            "fg": "#111111",
            "accent": "#1565C0",
            "btn_start": "#4CAF50",
            "btn_stop": "#E53935",
            "entry_bg": "#FFFFFF",
            "entry_fg": "#111111"
        }


THEME = detect_theme()


# -------------------- Audio -------------------- #

def create_beep_wave(frequency=880, duration=0.25, volume=0.5, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t) * volume
    audio = (tone * 32767).astype(np.int16)
    return sa.WaveObject(audio, 1, 2, sample_rate)


BEEP_WAVE = create_beep_wave()


# -------------------- EAR & CNN helpers -------------------- #

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


def calculate_ear(landmarks, indices):
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    ear = (A + B) / (2.0 * C + 1e-6)
    return ear


def seconds_to_hms(total_sec: float):
    total_sec = int(round(total_sec))
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    return h, m, s


def hms_str(total_sec: float):
    h, m, s = seconds_to_hms(total_sec)
    return f"{h:02d}:{m:02d}:{s:02d}"


def crop_eye(frame, eye_pts, margin=1.2):
    """
    Crop a patch around the eye landmarks (pixel coords).
    Larger margin for robustness with glasses / reflections.
    """
    h, w, _ = frame.shape
    xs = eye_pts[:, 0]
    ys = eye_pts[:, 1]

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    box_w = x_max - x_min
    box_h = y_max - y_min

    x_min = max(int(x_min - margin * box_w), 0)
    x_max = min(int(x_max + margin * box_w), w)
    y_min = max(int(y_min - margin * box_h), 0)
    y_max = min(int(y_max + margin * box_h), h)

    eye_img = frame[y_min:y_max, x_min:x_max]
    return eye_img


def preprocess_eye(eye_img):
    if eye_img is None or eye_img.size == 0:
        return None
    eye = cv2.resize(eye_img, (IMG_SIZE, IMG_SIZE))
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)
    eye = eye.astype("float32") / 255.0
    eye = np.expand_dims(eye, axis=0)
    return eye


# -------------------- Main App -------------------- #

class HybridDrowsinessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Drowsiness Detection (Hybrid EAR+CNN UI)")
        self.root.configure(bg=THEME["bg"])
        self.root.geometry("1100x750")

        # Scrollable UI
        self.canvas = tk.Canvas(self.root, bg=THEME["bg"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scroll_frame = tk.Frame(self.canvas, bg=THEME["bg"])
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Mode: EAR or HYBRID
        self.mode = tk.StringVar(value="HYBRID")

        # Camera & detection state
        self.cap = None
        self.running = False
        self.alarm_flag = False
        self.drowsy_start_time = None

        self.ear_threshold = tk.DoubleVar(value=0.25)
        self.drowsy_seconds = tk.DoubleVar(value=4.0)
        self.status_text = tk.StringVar(value="STATUS: STOPPED")
        self.current_ear_display = tk.StringVar(value="EAR: 0.000")

        # Auto-calibration
        self.is_calibrating = False
        self.calibration_ears = []
        self.calibration_seconds = 3.0
        self.calibration_start_time = None
        self.auto_ear_label_text = tk.StringVar(
            value="EAR Auto-Calibrated: N/A (will calibrate on START)"
        )

        # Logs
        self.log_timestamps = []
        self.log_ears = []
        self.log_status = []
        self.log_closed_flag = []

        # Smoothing
        self.ear_history = []
        self.ear_smooth_window = 8   # slightly more smoothing

        # Hybrid history (sliding window)
        self.closed_history = []

        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # CNN model
        print(f"Loading hybrid CNN model from {MODEL_PATH} ...")
        self.cnn_model = load_model(MODEL_PATH)
        print("Model loaded.")

        self.pie_canvas = None
        self.graph_canvas = None
        self.hist_canvas = None

        self._build_ui()

    # ------------- Scrolling ------------- #

    def _on_mousewheel(self, event):
        if event.delta:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ------------- UI Layout ------------- #

    def _build_ui(self):
        ctrl = tk.Frame(self.scroll_frame, bg=THEME["bg"])
        ctrl.pack(side=tk.TOP, fill=tk.X, pady=10)

        status_label = tk.Label(
            ctrl, textvariable=self.status_text,
            font=("Helvetica", 14, "bold"),
            fg=THEME["accent"], bg=THEME["bg"]
        )
        status_label.pack(side=tk.TOP, pady=(0, 3))

        # Prominent auto-calibration banner (Option 2)
        auto_label = tk.Label(
            ctrl,
            textvariable=self.auto_ear_label_text,
            font=("Helvetica", 11, "bold"),
            fg=THEME["accent"],
            bg=THEME["bg"]
        )
        auto_label.pack(side=tk.TOP, pady=(0, 5))

        # Mode toggle
        mode_frame = tk.Frame(ctrl, bg=THEME["bg"])
        mode_frame.pack(side=tk.TOP, pady=(0, 5))
        tk.Label(
            mode_frame, text="Mode:", fg=THEME["fg"],
            bg=THEME["bg"], font=("Helvetica", 10)
        ).pack(side=tk.LEFT, padx=(0, 5))
        tk.Radiobutton(
            mode_frame, text="EAR only", variable=self.mode, value="EAR",
            bg=THEME["bg"], fg=THEME["fg"], selectcolor=THEME["panel_bg"],
            command=self._on_mode_change
        ).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(
            mode_frame, text="Hybrid (EAR + CNN)", variable=self.mode, value="HYBRID",
            bg=THEME["bg"], fg=THEME["fg"], selectcolor=THEME["panel_bg"],
            command=self._on_mode_change
        ).pack(side=tk.LEFT, padx=5)

        sliders = tk.Frame(ctrl, bg=THEME["bg"])
        sliders.pack(side=tk.TOP)

        # EAR threshold slider (manual override still allowed)
        tk.Label(
            sliders, text="EAR Threshold (can override auto-calibrated value)",
            fg=THEME["fg"], bg=THEME["bg"], font=("Helvetica", 10)
        ).grid(row=0, column=0, padx=10, sticky="w")

        ear_scale = tk.Scale(
            sliders, from_=0.15, to=0.35, resolution=0.005,
            orient=tk.HORIZONTAL, length=250,
            variable=self.ear_threshold,
            fg=THEME["fg"], bg=THEME["bg"],
            troughcolor=THEME["panel_bg"],
            highlightthickness=0
        )
        ear_scale.grid(row=1, column=0, padx=10)

        self.ear_value_label = tk.Label(
            sliders, text=f"{self.ear_threshold.get():.3f}",
            fg=THEME["fg"], bg=THEME["bg"], font=("Helvetica", 10)
        )
        self.ear_value_label.grid(row=2, column=0, pady=(0, 5))

        def on_ear_change(val):
            self.ear_value_label.config(text=f"{float(val):.3f}")

        ear_scale.config(command=on_ear_change)

        # Drowsy duration
        tk.Label(
            sliders, text="Drowsy Duration (seconds)",
            fg=THEME["fg"], bg=THEME["bg"], font=("Helvetica", 10)
        ).grid(row=0, column=1, padx=10, sticky="w")

        self.dur_entry = tk.Entry(
            sliders, width=8,
            bg=THEME["entry_bg"], fg=THEME["entry_fg"],
            insertbackground=THEME["fg"],
            justify="center"
        )
        self.dur_entry.insert(0, "4.0")
        self.dur_entry.grid(row=1, column=1, padx=10)

        # Buttons
        btn_frame = tk.Frame(ctrl, bg=THEME["bg"])
        btn_frame.pack(side=tk.TOP, pady=5)

        self.start_btn = tk.Button(
            btn_frame, text="START",
            command=self.start_monitoring,
            bg=THEME["btn_start"], fg="white",
            activebackground=THEME["btn_start"],
            activeforeground="white",
            relief=tk.FLAT, width=10, font=("Helvetica", 11, "bold")
        )
        self.start_btn.grid(row=0, column=0, padx=10)

        self.stop_btn = tk.Button(
            btn_frame, text="STOP",
            command=self.stop_monitoring,
            bg=THEME["btn_stop"], fg="white",
            activebackground=THEME["btn_stop"],
            activeforeground="white",
            relief=tk.FLAT, width=10, font=("Helvetica", 11, "bold")
        )
        self.stop_btn.grid(row=0, column=1, padx=10)

        # Current EAR display
        self.ear_label = tk.Label(
            ctrl, textvariable=self.current_ear_display,
            fg=THEME["fg"], bg=THEME["bg"], font=("Helvetica", 10)
        )
        self.ear_label.pack(side=tk.TOP, pady=(0, 5))

        # Camera frame
        cam_frame = tk.Frame(self.scroll_frame, bg=THEME["bg"])
        cam_frame.pack(side=tk.TOP, pady=10)
        self.camera_label = tk.Label(cam_frame, bg=THEME["panel_bg"])
        self.camera_label.pack()

        # Pie chart + EAR graph frames
        self.pie_frame = tk.Frame(self.scroll_frame, bg=THEME["bg"])
        self.pie_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(10, 5))

        self.graph_frame = tk.Frame(self.scroll_frame, bg=THEME["bg"])
        self.graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(5, 20))

    # ------------- Mode change ------------- #

    def _on_mode_change(self):
        if self.mode.get() == "EAR":
            self.root.title("Driver Drowsiness Detection (EAR-only UI)")
        else:
            self.root.title("Driver Drowsiness Detection (Hybrid EAR+CNN UI)")

    # ------------- Start / Stop ------------- #

    def start_monitoring(self):
        if self.running:
            return

        try:
            dur_val = float(self.dur_entry.get())
            if dur_val <= 0:
                dur_val = 4.0
        except ValueError:
            dur_val = 4.0
        self.drowsy_seconds.set(dur_val)

        # Reset logs
        self.log_timestamps = []
        self.log_ears = []
        self.log_status = []
        self.log_closed_flag = []
        self.ear_history = []
        self.closed_history = []
        self.drowsy_start_time = None

        # Start auto-calibration
        self.is_calibrating = True
        self.calibration_ears = []
        self.calibration_start_time = time.time()
        self.status_text.set("STATUS: CALIBRATING (Keep eyes open)")
        self.auto_ear_label_text.set("Calibrating EAR... Please keep your eyes open")

        self.running = True
        self.alarm_flag = False

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 550)

        self._update_frame()

    def stop_monitoring(self):
        if not self.running:
            return
        self.running = False
        self.status_text.set("STATUS: STOPPED")
        self._stop_alarm()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self._plot_and_save_results()

    # ------------- Frame loop ------------- #

    def _update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self._update_frame)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, _ = frame.shape
        results = self.face_mesh.process(rgb)

        ear = None
        status_for_display = "AWAKE"
        status_for_log = "AWAKE"
        closed_flag = False

        cnn_text = ""

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            # EAR
            left_ear = calculate_ear(face_landmarks, LEFT_EYE_INDICES)
            right_ear = calculate_ear(face_landmarks, RIGHT_EYE_INDICES)
            ear = (left_ear + right_ear) / 2.0

            # Smoothing
            self.ear_history.append(ear)
            if len(self.ear_history) > self.ear_smooth_window:
                self.ear_history.pop(0)
            ear = float(np.mean(self.ear_history))
            self.current_ear_display.set(f"EAR: {ear:.3f}")

            # AUTO-CALIBRATION PHASE
            if self.is_calibrating:
                # collect EAR samples while assuming user is awake
                self.calibration_ears.append(ear)
                elapsed = time.time() - self.calibration_start_time

                # Require minimum samples to avoid degenerate calibration
                if elapsed >= self.calibration_seconds and len(self.calibration_ears) >= 10:
                    arr = np.array(self.calibration_ears, dtype=np.float32)
                    median_ear = float(np.median(arr))

                    # Filter out blinks: drop values much lower than median
                    filtered = arr[arr > 0.6 * median_ear]
                    if len(filtered) < 5:
                        filtered = arr

                    open_ear = float(np.mean(filtered))
                    dynamic_thresh = open_ear * 0.70

                    # Update slider + labels
                    self.ear_threshold.set(dynamic_thresh)
                    self.ear_value_label.config(text=f"{dynamic_thresh:.3f}")
                    self.auto_ear_label_text.set(
                        f"EAR Auto-Calibrated ✔ Personal Threshold: {dynamic_thresh:.3f}"
                    )
                    self.status_text.set("STATUS: MONITORING")
                    self.is_calibrating = False

                # During calibration, treat as awake
                status_for_display = "CALIBRATING..."
                status_for_log = "AWAKE"
                closed_flag = False

            else:
                # ------ NORMAL MONITORING (EAR or HYBRID) ------ #
                thresh = self.ear_threshold.get()
                now = time.time()
                dur_needed = self.drowsy_seconds.get()

                if self.mode.get() == "EAR":
                    # --- Pure EAR logic ---
                    if ear < thresh:
                        closed_flag = True
                        if self.drowsy_start_time is None:
                            self.drowsy_start_time = now
                        elif now - self.drowsy_start_time >= dur_needed:
                            status_for_display = "DROWSY"
                            status_for_log = "DROWSY"
                            self.status_text.set("STATUS: DROWSY")
                            self._start_alarm()
                    else:
                        status_for_display = "AWAKE"
                        status_for_log = "AWAKE"
                        if self.running:
                            self.status_text.set("STATUS: MONITORING")
                        self.drowsy_start_time = None
                        self._stop_alarm()

                else:
                    # --- Hybrid logic (EAR + CNN) ---
                    landmarks_np = np.array(
                        [[lm.x * w, lm.y * h] for lm in face_landmarks]
                    )
                    right_eye_pts = landmarks_np[RIGHT_EYE_INDICES]
                    left_eye_pts = landmarks_np[LEFT_EYE_INDICES]

                    # CNN eye crops (larger margin helps with glasses)
                    right_eye_img = crop_eye(frame, right_eye_pts, margin=1.2)
                    left_eye_img = crop_eye(frame, left_eye_pts, margin=1.2)

                    def cnn_closed_prob(eye_img):
                        inp = preprocess_eye(eye_img)
                        if inp is None:
                            return None
                        raw_prob = float(self.cnn_model.predict(inp, verbose=0)[0][0])
                        # raw_prob ~ P(open)  => closed_prob = 1 - raw_prob
                        closed_prob = 1.0 - raw_prob
                        return closed_prob

                    r_closed_prob = cnn_closed_prob(right_eye_img)
                    l_closed_prob = cnn_closed_prob(left_eye_img)

                    # Handle single-eye cases robustly
                    valid_probs = [p for p in [r_closed_prob, l_closed_prob] if p is not None]
                    if len(valid_probs) == 0:
                        avg_closed_prob = None
                    elif len(valid_probs) == 1:
                        avg_closed_prob = valid_probs[0]
                    else:
                        avg_closed_prob = sum(valid_probs) / len(valid_probs)

                    if r_closed_prob is not None and l_closed_prob is not None:
                        cnn_text = f"CNN closed prob R/L: {r_closed_prob:.2f}/{l_closed_prob:.2f}"
                    elif r_closed_prob is not None:
                        cnn_text = f"CNN closed prob R: {r_closed_prob:.2f}"
                    elif l_closed_prob is not None:
                        cnn_text = f"CNN closed prob L: {l_closed_prob:.2f}"

                    # EAR-based closed check
                    ear_closed = ear < thresh

                    # CNN-based closed check
                    cnn_closed = False
                    if avg_closed_prob is not None:
                        # Strong evidence of open eyes
                        if avg_closed_prob < 0.30:
                            cnn_closed = False
                        # Strong evidence of closed eyes
                        elif avg_closed_prob > CNN_CLOSED_THRESH:
                            cnn_closed = True
                        # 0.30 - 0.60 => uncertain, rely more on EAR

                    # FINAL HYBRID DECISION:
                    # If CNN not available => use EAR only
                    # If CNN strongly says open => override EAR false dips
                    # Else require BOTH EAR and CNN to say closed
                    if avg_closed_prob is None:
                        is_closed = ear_closed
                    else:
                        if avg_closed_prob < 0.30:
                            is_closed = False
                        else:
                            is_closed = ear_closed and cnn_closed

                    closed_flag = is_closed

                    # Maintain sliding closed history
                    self.closed_history.append(1 if is_closed else 0)
                    if len(self.closed_history) > HYBRID_HISTORY_LEN:
                        self.closed_history.pop(0)
                    closed_ratio = (
                        sum(self.closed_history) / len(self.closed_history)
                        if self.closed_history else 0.0
                    )

                    if len(self.closed_history) == HYBRID_HISTORY_LEN and closed_ratio > HYBRID_DROWSY_RATIO:
                        status_for_display = "DROWSY"
                        status_for_log = "DROWSY"
                        self.status_text.set("STATUS: DROWSY")
                        self._start_alarm()
                    elif is_closed:
                        status_for_display = "Blink / Closing..."
                        status_for_log = "AWAKE"
                        if self.running:
                            self.status_text.set("STATUS: MONITORING")
                        self._stop_alarm()
                    else:
                        status_for_display = "AWAKE"
                        status_for_log = "AWAKE"
                        if self.running:
                            self.status_text.set("STATUS: MONITORING")
                        self._stop_alarm()

                    # Draw eye landmarks
                    for pt in np.vstack((right_eye_pts, left_eye_pts)):
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (255, 0, 0), -1)

                    cv2.putText(
                        frame, f"Closed ratio: {closed_ratio:.2f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2
                    )

            # Common overlay
            cv2.putText(
                frame,
                f"EAR: {ear:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            if cnn_text:
                cv2.putText(
                    frame, cnn_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    2,
                )

            color = (0, 255, 0) if status_for_display.startswith("AWAKE") or status_for_display.startswith("CALIBRATING") else (0, 0, 255)
            cv2.putText(
                frame,
                status_for_display,
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                3,
            )

        # Log
        ts = time.time()
        self.log_timestamps.append(ts)
        self.log_ears.append(ear if ear is not None else 0.0)
        self.log_status.append(status_for_log)
        self.log_closed_flag.append(closed_flag)

        # Show in Tkinter
        rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_display)
        img = img.resize((900, 550), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)

        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk, bg=THEME["panel_bg"])

        self.root.after(10, self._update_frame)

    # ------------- Alarm ------------- #

    def _alarm_loop(self):
        while self.alarm_flag:
            try:
                play_obj = BEEP_WAVE.play()
                play_obj.wait_done()
            except Exception:
                pass
            time.sleep(0.3)

    def _start_alarm(self):
        if self.alarm_flag:
            return
        self.alarm_flag = True        # type: ignore
        t = threading.Thread(target=self._alarm_loop, daemon=True)
        t.start()

    def _stop_alarm(self):
        self.alarm_flag = False

    # ------------- Metrics & plots ------------- #

    def _compute_time_distribution(self):
        if len(self.log_timestamps) < 2:
            return 0.0, 0.0, 0.0, 0.0

        awake_time = 0.0
        drowsy_time = 0.0
        closed_time = 0.0

        for i in range(1, len(self.log_timestamps)):
            dt = self.log_timestamps[i] - self.log_timestamps[i - 1]
            prev_state = self.log_status[i - 1]
            prev_closed = self.log_closed_flag[i - 1]

            if prev_state == "AWAKE":
                awake_time += dt
            elif prev_state == "DROWSY":
                drowsy_time += dt

            if prev_closed:
                closed_time += dt

        total = awake_time + drowsy_time
        return awake_time, drowsy_time, closed_time, total

    def _extract_events(self, drowsy_seconds):
        if len(self.log_timestamps) < 2:
            return {"drowsy_durations": [], "blink_durations": []}

        closed_segments = []
        in_segment = False
        seg_start = None

        for i in range(len(self.log_timestamps)):
            closed = self.log_closed_flag[i]
            t = self.log_timestamps[i]

            if closed and not in_segment:
                in_segment = True
                seg_start = t
            elif not closed and in_segment:
                in_segment = False
                seg_end = t
                closed_segments.append(seg_end - seg_start)

        if in_segment and seg_start is not None:
            seg_end = self.log_timestamps[-1]
            closed_segments.append(seg_end - seg_start)

        drowsy_durations = []
        blink_durations = []

        for dur in closed_segments:
            if dur >= drowsy_seconds:
                drowsy_durations.append(dur)
            else:
                blink_durations.append(dur)

        return {"drowsy_durations": drowsy_durations, "blink_durations": blink_durations}

    def _clear_old_charts(self):
        if self.pie_canvas is not None:
            self.pie_canvas.get_tk_widget().destroy()
            self.pie_canvas = None
        if self.graph_canvas is not None:
            self.graph_canvas.get_tk_widget().destroy()
            self.graph_canvas = None
        if self.hist_canvas is not None:
            self.hist_canvas.get_tk_widget().destroy()
            self.hist_canvas = None

    def _plot_and_save_results(self):
        if len(self.log_ears) < 5:
            return

        self._clear_old_charts()
        results_dir = "session_results"
        os.makedirs(results_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        t0 = self.log_timestamps[0]
        times = [t - t0 for t in self.log_timestamps]
        ears = self.log_ears

        awake_sec, drowsy_sec, closed_sec, total_sec = self._compute_time_distribution()
        if total_sec <= 0:
            total_sec = 1e-6

        ear_arr = np.array(ears, dtype=np.float32)
        mean_ear = float(np.mean(ear_arr))
        median_ear = float(np.median(ear_arr))
        min_ear = float(np.min(ear_arr))
        max_ear = float(np.max(ear_arr))
        std_ear = float(np.std(ear_arr))

        events = self._extract_events(self.drowsy_seconds.get())
        drowsy_durations = events["drowsy_durations"]
        blink_durations = events["blink_durations"]

        num_drowsy = len(drowsy_durations)
        total_drowsy_event_time = float(sum(drowsy_durations))
        avg_drowsy = float(total_drowsy_event_time / num_drowsy) if num_drowsy > 0 else 0.0
        max_drowsy = float(max(drowsy_durations)) if num_drowsy > 0 else 0.0

        num_blinks = len(blink_durations)
        total_blink_time = float(sum(blink_durations))
        blink_freq = float(num_blinks / (total_sec / 60.0)) if total_sec > 0 else 0.0

        awake_pct = 100.0 * awake_sec / total_sec
        drowsy_pct = 100.0 * drowsy_sec / total_sec
        closed_pct = 100.0 * closed_sec / total_sec

        # ----- EAR curve -----
        fig_ear = Figure(figsize=(7, 2.5), dpi=100)
        ax1 = fig_ear.add_subplot(111)
        ax1.plot(times, ears, label="EAR")
        ax1.axhline(self.ear_threshold.get(), color="red", linestyle="--", label="Threshold")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("EAR")
        ax1.set_title("EAR Over Time")
        ax1.grid(True)
        ax1.legend()

        self.graph_canvas = FigureCanvasTkAgg(fig_ear, master=self.graph_frame)
        self.graph_canvas.draw()
        self.graph_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ----- EAR Distribution Histogram -----
        fig_hist = Figure(figsize=(4, 2.5), dpi=100)
        axh = fig_hist.add_subplot(111)
        axh.hist(ears, bins=20, alpha=0.8)
        axh.set_title("EAR Distribution (Frequency vs EAR)")
        axh.set_xlabel("EAR")
        axh.set_ylabel("Frequency")

        self.hist_canvas = FigureCanvasTkAgg(fig_hist, master=self.graph_frame)
        self.hist_canvas.draw()
        self.hist_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ----- Pie chart -----
        fig_pie = Figure(figsize=(4, 2.5), dpi=100)
        ax2 = fig_pie.add_subplot(111)

        times_list = [awake_sec, drowsy_sec]
        labels = ["Awake", "Drowsy"]
        colors = ["#4CAF50", "#E53935"]

        def make_autopct(time_values):
            def autopct(pct):
                total = sum(time_values)
                sec = pct / 100.0 * total
                return f"{pct:.1f}%\n({hms_str(sec)})"
            return autopct

        ax2.pie(
            times_list,
            labels=labels,
            autopct=make_autopct(times_list),
            colors=colors,
            startangle=90
        )
        ax2.set_title("Time Distribution (Awake vs Drowsy)")

        self.pie_canvas = FigureCanvasTkAgg(fig_pie, master=self.pie_frame)
        self.pie_canvas.draw()
        self.pie_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ----- Stats table figure -----
        fig_stats = Figure(figsize=(8.27, 11.69), dpi=100)
        ax_stats = fig_stats.add_subplot(111)
        ax_stats.axis("off")

        mode_label = "Hybrid (EAR+CNN)" if self.mode.get() == "HYBRID" else "EAR only"
        title = f"Driver Drowsiness Detection - Session Report ({mode_label})"
        dt_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ax_stats.text(0.5, 0.97, title, ha="center", va="top",
                      fontsize=14, fontweight="bold")
        ax_stats.text(0.02, 0.93, f"Date/Time: {dt_str}",
                      ha="left", va="top", fontsize=11)
        ax_stats.text(
            0.02, 0.90,
            f"EAR Threshold (final): {self.ear_threshold.get():.3f}    "
            f"Drowsy Duration Setting: {self.drowsy_seconds.get():.1f} s",
            ha="left", va="top", fontsize=11
        )

        time_data = [
            ["Total Monitoring Time", hms_str(total_sec)],
            ["Awake Time", hms_str(awake_sec)],
            ["Drowsy Time", hms_str(drowsy_sec)],
            ["Eyes-Closed Time (closed_flag)", hms_str(closed_sec)],
            ["Awake (%)", f"{awake_pct:.2f}%"],
            ["Drowsy (%)", f"{drowsy_pct:.2f}%"],
            ["Eyes-Closed (%)", f"{closed_pct:.2f}%"],
        ]
        tab1 = ax_stats.table(
            cellText=time_data,
            colLabels=["Session Metric", "Value"],
            loc="upper left",
            cellLoc="left",
            bbox=[0.02, 0.58, 0.96, 0.30]
        )
        tab1.auto_set_column_width(col=list(range(2)))
        tab1.scale(1.0, 1.2)

        ear_data = [
            ["Mean EAR", f"{mean_ear:.3f}"],
            ["Median EAR", f"{median_ear:.3f}"],
            ["Min EAR", f"{min_ear:.3f}"],
            ["Max EAR", f"{max_ear:.3f}"],
            ["Std EAR", f"{std_ear:.3f}"],
        ]
        tab2 = ax_stats.table(
            cellText=ear_data,
            colLabels=["EAR Metric", "Value"],
            loc="upper left",
            cellLoc="left",
            bbox=[0.02, 0.39, 0.96, 0.16]
        )
        tab2.auto_set_column_width(col=list(range(2)))
        tab2.scale(1.0, 1.2)

        event_data = [
            ["Number of Drowsy Events", str(num_drowsy)],
            ["Total Drowsy Event Time", hms_str(total_drowsy_event_time)],
            ["Avg Drowsy Event Duration", hms_str(avg_drowsy)],
            ["Longest Drowsy Event", hms_str(max_drowsy)],
            ["Number of Blinks", str(num_blinks)],
            ["Total Blink Time", hms_str(total_blink_time)],
            ["Blink Frequency", f"{blink_freq:.2f} blinks/min"],
        ]
        tab3 = ax_stats.table(
            cellText=event_data,
            colLabels=["Event Metric", "Value"],
            loc="upper left",
            cellLoc="left",
            bbox=[0.02, 0.10, 0.96, 0.25]
        )
        tab3.auto_set_column_width(col=list(range(2)))
        tab3.scale(1.0, 1.2)

        # CSV log
        csv_path = os.path.join(results_dir, f"session_log_{timestamp_str}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "ear", "status", "closed_flag"])
            for ts, ear_val, st, cf in zip(
                self.log_timestamps, self.log_ears, self.log_status, self.log_closed_flag
            ):
                writer.writerow([ts, ear_val if ear_val is not None else 0.0, st, int(cf)])

        # Log file path on stats page
        abs_csv_path = os.path.abspath(csv_path)
        ax_stats.text(
            0.02, 0.05,
            f"Log file: {abs_csv_path}",
            fontsize=10,
            ha="left", va="top"
        )

        fig_stats.tight_layout(rect=[0, 0, 1, 0.99])

        # Save PNGs
        pie_path = os.path.join(results_dir, f"piechart_{timestamp_str}.png")
        ear_path = os.path.join(results_dir, f"earplot_{timestamp_str}.png")
        hist_path = os.path.join(results_dir, f"ear_hist_{timestamp_str}.png")
        stats_path = os.path.join(results_dir, f"session_stats_{timestamp_str}.png")

        fig_pie.savefig(pie_path, bbox_inches="tight")
        fig_ear.savefig(ear_path, bbox_inches="tight")
        fig_hist.savefig(hist_path, bbox_inches="tight")
        fig_stats.savefig(stats_path, bbox_inches="tight")

        # PDF
        pdf_path = os.path.join(results_dir, f"session_report_{timestamp_str}.pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig_stats, bbox_inches="tight")   # stats + log path
            pdf.savefig(fig_pie, bbox_inches="tight")     # time distribution
            pdf.savefig(fig_ear, bbox_inches="tight")     # EAR over time
            pdf.savefig(fig_hist, bbox_inches="tight")    # EAR histogram

        print(f"[INFO] Saved PNGs, CSV, and PDF to: {os.path.abspath(results_dir)}")
        print(f"[INFO] Log file: {abs_csv_path}")


# -------------------- Main -------------------- #

if __name__ == "__main__":
    root = tk.Tk()
    app = HybridDrowsinessApp(root)
    root.mainloop()
