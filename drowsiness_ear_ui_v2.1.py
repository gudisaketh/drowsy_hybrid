"""
drowsiness_ear_ui_v2.py
-----------------------

Driver Drowsiness Detection (EAR-only, MediaPipe FaceMesh) with Tkinter UI.

Features:
- EAR-based drowsiness detection using MediaPipe FaceMesh.
- Tkinter UI:
    * EAR threshold slider
    * Drowsy duration (seconds) entry
    * START / STOP buttons
    * Live camera feed with EAR + status overlay
    * Scrollable window with:
        - Camera view
        - Awake vs Drowsy pie chart
        - EAR-over-time line graph
- Continuous beep alarm once drowsy condition is met.
- Per-session logging:
    * Per-frame timestamps, EAR, and status ("AWAKE" or "DROWSY")
    * Detailed session stats:
        - Total monitoring time
        - Awake / Drowsy / Eyes-closed times
        - Mean / median / min / max / std EAR
        - Drowsy events (count, total time, avg & longest duration)
        - Blink stats (short closures; count, total blink time, blink frequency)
- On STOP:
    * UI updates with pie chart + EAR line chart
    * Saves to `session_results/`:
        - piechart_<timestamp>.png
        - earplot_<timestamp>.png
        - ear_stats_<timestamp>.png  (EAR histogram + table)
        - session_log_<timestamp>.csv
        - session_report_<timestamp>.pdf (with table)
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


# -------------------- Theme detection (light/dark) -------------------- #

def detect_theme():
    """
    Try to detect macOS light/dark mode.
    Fallback: light theme on other OS or on error.
    """
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
        # Dark theme
        return {
            "bg": "#202124",
            "panel_bg": "#303134",
            "fg": "#FFFFFF",
            "accent": "#0B84FF",
            "btn_start": "#34C759",   # green
            "btn_stop": "#FF3B30",    # red
            "entry_bg": "#3C4043",
            "entry_fg": "#FFFFFF"
        }
    else:
        # Light theme
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


# -------------------- Audio: continuous beep wave -------------------- #

def create_beep_wave(frequency=880, duration=0.25, volume=0.5, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t) * volume
    audio = (tone * 32767).astype(np.int16)
    return sa.WaveObject(audio, 1, 2, sample_rate)


BEEP_WAVE = create_beep_wave()


# -------------------- EAR helpers (MediaPipe indices) -------------------- #

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


def calculate_ear(landmarks, indices):
    """
    Calculate Eye Aspect Ratio (EAR) from MediaPipe landmarks.
    landmarks: list of normalized landmarks (x, y)
    indices: list of 6 landmark indices for one eye
    """
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    ear = (A + B) / (2.0 * C + 1e-6)
    return ear


def seconds_to_hms(total_sec: float):
    """Convert seconds to (H, M, S) integers."""
    total_sec = int(round(total_sec))
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    return h, m, s


def hms_str(total_sec: float):
    """Return HH:MM:SS string for given seconds."""
    h, m, s = seconds_to_hms(total_sec)
    return f"{h:02d}:{m:02d}:{s:02d}"


# -------------------- Main Tkinter App -------------------- #

class DrowsinessEARApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Drowsiness Detection (EAR-only)")
        self.root.configure(bg=THEME["bg"])
        self.root.geometry("1100x750")

        # Scrollable canvas
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

        # Camera & detection state
        self.cap = None
        self.running = False
        self.alarm_flag = False
        self.drowsy_start_time = None

        self.ear_threshold = tk.DoubleVar(value=0.25)
        self.drowsy_seconds = tk.DoubleVar(value=4.0)
        self.status_text = tk.StringVar(value="STATUS: STOPPED")
        self.current_ear_display = tk.StringVar(value="EAR: 0.000")

        # Logs
        self.log_timestamps = []
        self.log_ears = []
        self.log_status = []
        self.log_closed_flag = []  # True if EAR < threshold at that frame

        # EAR smoothing
        self.ear_history = []
        self.ear_smooth_window = 5

        # MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Matplotlib canvases for UI
        self.pie_canvas = None
        self.graph_canvas = None

        self._build_ui()

    # -------------------- Scrolling handler -------------------- #

    def _on_mousewheel(self, event):
        if event.delta:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # -------------------- UI Layout -------------------- #

    def _build_ui(self):
        # Top control panel
        ctrl = tk.Frame(self.scroll_frame, bg=THEME["bg"])
        ctrl.pack(side=tk.TOP, fill=tk.X, pady=10)

        status_label = tk.Label(
            ctrl, textvariable=self.status_text,
            font=("Helvetica", 14, "bold"),
            fg=THEME["accent"], bg=THEME["bg"]
        )
        status_label.pack(side=tk.TOP, pady=(0, 5))

        sliders = tk.Frame(ctrl, bg=THEME["bg"])
        sliders.pack(side=tk.TOP)

        # EAR threshold slider
        tk.Label(
            sliders, text="EAR Threshold", fg=THEME["fg"],
            bg=THEME["bg"], font=("Helvetica", 10)
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

        # Drowsy duration (seconds)
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

        # Pie chart frame
        self.pie_frame = tk.Frame(self.scroll_frame, bg=THEME["bg"])
        self.pie_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(10, 5))

        # EAR graph frame
        self.graph_frame = tk.Frame(self.scroll_frame, bg=THEME["bg"])
        self.graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(5, 20))

    # -------------------- Camera / Monitoring -------------------- #

    def start_monitoring(self):
        if self.running:
            return

        # Read drowsy duration
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
        self.drowsy_start_time = None

        self.running = True
        self.status_text.set("STATUS: MONITORING")
        self.alarm_flag = False

        # Open camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 550)

        # Some cameras ignore these, but try anyway
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
        except Exception:
            pass

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

        # Generate charts + save outputs
        self._plot_and_save_results()

    def _update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self._update_frame)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        ear = None
        status = "AWAKE"
        closed_flag = False

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            left_ear = calculate_ear(face_landmarks, LEFT_EYE_INDICES)
            right_ear = calculate_ear(face_landmarks, RIGHT_EYE_INDICES)
            ear = (left_ear + right_ear) / 2.0

            # Smooth EAR
            self.ear_history.append(ear)
            if len(self.ear_history) > self.ear_smooth_window:
                self.ear_history.pop(0)
            ear = float(np.mean(self.ear_history))

            self.current_ear_display.set(f"EAR: {ear:.3f}")
            thresh = self.ear_threshold.get()
            dur_needed = self.drowsy_seconds.get()
            now = time.time()

            closed_flag = ear < thresh

            if closed_flag:
                if self.drowsy_start_time is None:
                    self.drowsy_start_time = now
                elif now - self.drowsy_start_time >= dur_needed:
                    status = "DROWSY"
                    self.status_text.set("STATUS: DROWSY")
                    self._start_alarm()
            else:
                status = "AWAKE"
                if self.running:
                    self.status_text.set("STATUS: MONITORING")
                self.drowsy_start_time = None
                self._stop_alarm()

            # Overlay EAR & status on frame
            cv2.putText(
                frame,
                f"EAR: {ear:.3f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0) if status == "AWAKE" else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                status,
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0) if status == "AWAKE" else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Log frame
        ts = time.time()
        self.log_timestamps.append(ts)
        self.log_ears.append(ear if ear is not None else 0.0)
        self.log_status.append(status)
        self.log_closed_flag.append(closed_flag)

        # Show frame in Tkinter
        rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_display)
        img = img.resize((900, 550), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)

        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk, bg=THEME["panel_bg"])

        self.root.after(10, self._update_frame)

    # -------------------- Alarm: continuous beeping -------------------- #

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
        self.alarm_flag = True
        t = threading.Thread(target=self._alarm_loop, daemon=True)
        t.start()

    def _stop_alarm(self):
        self.alarm_flag = False

    # -------------------- Chart & metrics helpers -------------------- #

    def _clear_old_charts(self):
        if self.pie_canvas is not None:
            self.pie_canvas.get_tk_widget().destroy()
            self.pie_canvas = None
        if self.graph_canvas is not None:
            self.graph_canvas.get_tk_widget().destroy()
            self.graph_canvas = None

    def _compute_time_distribution(self):
        """
        Compute total time spent awake vs drowsy using log_status.
        Returns:
            awake_sec, drowsy_sec, total_sec
        """
        if len(self.log_timestamps) < 2:
            return 0.0, 0.0, 0.0

        awake_time = 0.0
        drowsy_time = 0.0

        for i in range(1, len(self.log_timestamps)):
            dt = self.log_timestamps[i] - self.log_timestamps[i - 1]
            prev_state = self.log_status[i - 1]
            if prev_state == "AWAKE" or prev_state == "MONITORING":
                awake_time += dt
            elif prev_state == "DROWSY":
                drowsy_time += dt

        total = awake_time + drowsy_time
        return awake_time, drowsy_time, total

    def _segment_events(self, mask, min_duration=0.0):
        """
        Segment contiguous True intervals in 'mask' given timestamps.
        mask: list[bool]
        min_duration: ignore events shorter than this duration (seconds)
        Returns list of (start_time, end_time, duration)
        """
        events = []
        if len(self.log_timestamps) < 2 or len(mask) != len(self.log_timestamps):
            return events

        in_event = False
        start_ts = None

        for i in range(len(mask)):
            flag = mask[i]
            ts = self.log_timestamps[i]

            if flag and not in_event:
                in_event = True
                start_ts = ts
            elif not flag and in_event:
                end_ts = ts
                dur = end_ts - start_ts
                if dur >= min_duration:
                    events.append((start_ts, end_ts, dur))
                in_event = False
                start_ts = None

        # If ended while still in event
        if in_event and start_ts is not None:
            end_ts = self.log_timestamps[-1]
            dur = end_ts - start_ts
            if dur >= min_duration:
                events.append((start_ts, end_ts, dur))

        return events

    # -------------------- Plotting + Saving -------------------- #

    def _plot_and_save_results(self):
        if len(self.log_ears) < 5 or len(self.log_timestamps) < 2:
            print("[INFO] Not enough data to generate results.")
            return

        self._clear_old_charts()

        # Ensure results dir
        results_dir = "session_results"
        os.makedirs(results_dir, exist_ok=True)

        # Timestamp for filenames
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Normalize time axis
        t0 = self.log_timestamps[0]
        times = [t - t0 for t in self.log_timestamps]
        ears = self.log_ears
        statuses = self.log_status
        closed_flags = self.log_closed_flag

        # Time distribution (awake vs drowsy)
        awake_sec, drowsy_sec, total_sec = self._compute_time_distribution()
        if total_sec <= 0:
            total_sec = 1e-6

        # Eyes-closed time (EAR < threshold)
        closed_time = 0.0
        for i in range(1, len(times)):
            dt = self.log_timestamps[i] - self.log_timestamps[i - 1]
            if closed_flags[i - 1]:
                closed_time += dt

        # EAR stats
        ear_array = np.array(ears, dtype=float)
        mean_ear = float(np.mean(ear_array))
        median_ear = float(np.median(ear_array))
        min_ear = float(np.min(ear_array))
        max_ear = float(np.max(ear_array))
        std_ear = float(np.std(ear_array))

        # Drowsy events: contiguous intervals where status == "DROWSY"
        drowsy_mask = [st == "DROWSY" for st in statuses]
        drowsy_events = self._segment_events(drowsy_mask, min_duration=0.0)
        num_drowsy_events = len(drowsy_events)
        total_drowsy_event_time = sum(ev[2] for ev in drowsy_events)
        avg_drowsy_event = (total_drowsy_event_time / num_drowsy_events) if num_drowsy_events > 0 else 0.0
        longest_drowsy_event = max((ev[2] for ev in drowsy_events), default=0.0)

        # Blinks: intervals where eye is closed (EAR < threshold) but not labeled DROWSY
        # i.e., closed_flag True AND status != "DROWSY"
        blink_mask = [cf and (st != "DROWSY") for cf, st in zip(closed_flags, statuses)]
        drowsy_setting = self.drowsy_seconds.get()
        # We treat blinks as short closures (< drowsy_setting)
        blink_events_all = self._segment_events(blink_mask, min_duration=0.0)
        blink_events = [ev for ev in blink_events_all if ev[2] < drowsy_setting]

        num_blinks = len(blink_events)
        total_blink_time = sum(ev[2] for ev in blink_events)
        blink_freq_per_min = 0.0
        if total_sec > 0:
            blink_freq_per_min = num_blinks / (total_sec / 60.0)

        # ---------- EAR over time figure (for UI + PNG) ----------
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

        # ---------- Pie chart figure (Awake vs Drowsy) ----------
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

        # ---------- EAR stats figure (histogram + small table) ----------
        fig_stats = Figure(figsize=(6, 3), dpi=100)
        ax_stats = fig_stats.add_subplot(121)
        ax_table = fig_stats.add_subplot(122)
        ax_table.axis("off")

        # Histogram of EAR
        ax_stats.hist(ear_array, bins=20, edgecolor="black")
        ax_stats.axvline(self.ear_threshold.get(), color="red", linestyle="--", label="Threshold")
        ax_stats.set_xlabel("EAR")
        ax_stats.set_ylabel("Frequency")
        ax_stats.set_title("EAR Distribution")
        ax_stats.legend()

        # Short table of key stats (to avoid overlap)
        table_data = [
            ["Mean EAR", f"{mean_ear:.3f}"],
            ["Median EAR", f"{median_ear:.3f}"],
            ["Min EAR", f"{min_ear:.3f}"],
            ["Max EAR", f"{max_ear:.3f}"],
            ["Std EAR", f"{std_ear:.3f}"],
            ["Drowsy Events", f"{num_drowsy_events}"],
            ["Blinks", f"{num_blinks}"],
        ]
        table = ax_table.table(
            cellText=table_data,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.3)
        ax_table.set_title("Session Summary", fontsize=10, pad=8)

        # ---------- Save PNGs ----------
        pie_path = os.path.join(results_dir, f"piechart_{timestamp_str}.png")
        ear_path = os.path.join(results_dir, f"earplot_{timestamp_str}.png")
        stats_path = os.path.join(results_dir, f"ear_stats_{timestamp_str}.png")
        fig_pie.savefig(pie_path, bbox_inches="tight")
        fig_ear.savefig(ear_path, bbox_inches="tight")
        fig_stats.savefig(stats_path, bbox_inches="tight")

        # ---------- Save CSV log ----------
        csv_path = os.path.join(results_dir, f"session_log_{timestamp_str}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "ear", "status", "closed_flag"])
            for ts, ear_val, st, cf in zip(
                self.log_timestamps, self.log_ears, self.log_status, self.log_closed_flag
            ):
                writer.writerow([ts, ear_val if ear_val is not None else 0.0, st, int(cf)])

        # ---------- Save PDF report (with table) ----------
        pdf_path = os.path.join(results_dir, f"session_report_{timestamp_str}.pdf")
        with PdfPages(pdf_path) as pdf:
            # Page 1: Text summary
            fig_summary = Figure(figsize=(8.27, 11.69), dpi=100)  # A4
            ax_sum = fig_summary.add_subplot(111)
            ax_sum.axis("off")

            lines = []
            lines.append("Driver Drowsiness Detection - Session Report")
            lines.append("")
            lines.append(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")
            lines.append(f"EAR Threshold: {self.ear_threshold.get():.3f}")
            lines.append(f"Drowsy Duration Setting: {self.drowsy_seconds.get():.1f} s")
            lines.append("")
            lines.append(f"Total Monitoring Time: {hms_str(total_sec)}")
            lines.append(f"Awake Time: {hms_str(awake_sec)}")
            lines.append(f"Drowsy Time: {hms_str(drowsy_sec)}")
            lines.append(f"Eyes-Closed Time (EAR < threshold): {hms_str(closed_time)}")
            lines.append("")
            awake_pct = (awake_sec / total_sec) * 100.0 if total_sec > 0 else 0.0
            drowsy_pct = (drowsy_sec / total_sec) * 100.0 if total_sec > 0 else 0.0
            closed_pct = (closed_time / total_sec) * 100.0 if total_sec > 0 else 0.0
            lines.append(f"Awake: {awake_pct:.2f}%")
            lines.append(f"Drowsy: {drowsy_pct:.2f}%")
            lines.append(f"Eyes-Closed (any closure): {closed_pct:.2f}%")
            lines.append("")
            lines.append("EAR Statistics:")
            lines.append(f"  Mean EAR:   {mean_ear:.3f}")
            lines.append(f"  Median EAR: {median_ear:.3f}")
            lines.append(f"  Min EAR:    {min_ear:.3f}")
            lines.append(f"  Max EAR:    {max_ear:.3f}")
            lines.append(f"  Std EAR:    {std_ear:.3f}")
            lines.append("")
            lines.append("Drowsy Event Statistics (long closures meeting drowsy duration):")
            lines.append(f"  Number of Drowsy Events: {num_drowsy_events}")
            lines.append(f"  Total Drowsy Event Time: {total_drowsy_event_time:.2f} s")
            lines.append(f"  Avg Drowsy Event Duration: {avg_drowsy_event:.2f} s")
            lines.append(f"  Longest Drowsy Event: {longest_drowsy_event:.2f} s")
            lines.append("")
            lines.append("Blink Statistics (short closures < drowsy duration):")
            lines.append(f"  Number of Blinks: {num_blinks}")
            lines.append(f"  Total Blink Time: {total_blink_time:.2f} s")
            lines.append(f"  Blink Frequency: {blink_freq_per_min:.2f} blinks/min")
            lines.append("")
            lines.append("Log file:")
            lines.append(os.path.abspath(csv_path))

            text = "\n".join(lines)
            ax_sum.text(
                0.05, 0.95, text,
                va="top", ha="left", fontsize=11
            )

            pdf.savefig(fig_summary)

            # Page 2: Table + histogram
            # Reuse fig_stats (hist + small table)
            pdf.savefig(fig_stats)

            # Page 3: pie chart + EAR-over-time
            pdf.savefig(fig_pie)
            pdf.savefig(fig_ear)

        print(f"[INFO] Saved PNGs, CSV, and PDF to: {os.path.abspath(results_dir)}")


# -------------------- Main -------------------- #

if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessEARApp(root)
    root.mainloop()
