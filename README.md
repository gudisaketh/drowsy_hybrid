# 🚗 Hybrid Driver Drowsiness Detection (EAR + MediaPipe FaceMesh + CNN)

A real-time driver drowsiness detection system combining:

- **Eye Aspect Ratio (EAR)** via MediaPipe FaceMesh  
- **Fine-tuned MobileNetV2 CNN** for eye-state classification  
- **Hybrid fusion logic** (EAR + CNN) for robust detection  
- **Auto EAR calibration**, **blink detection**, **single-eye fallback**  
- **Tkinter UI**, **session analytics**, and **auto-generated PDF reporting**

Designed to perform reliably under eyeglasses, glare, occlusion, low light, and natural variations in eye shape.

---

# ⭐ Quick Start — Run the System Immediately

A **pre-trained CNN model** is already included:

```
models/eye_mobilenet_finetuned_v2.h5
```

Meaning **you do NOT need to train anything** to run the main system.

---

# ▶️ 1. Clone the repository

```bash
git clone https://github.com/gudisaketh/drowsy_hybrid.git
cd drowsy_hybrid
```

---

# ▶️ 2. Create & activate environment

```bash
python3 -m venv drowsy_env
source drowsy_env/bin/activate      # macOS / Linux
drowsy_env\Scripts\activate         # Windows
```

---

# ▶️ 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ 4. Start the Hybrid EAR+CNN Application

```bash
python drowsiness_hybrid_ui_combo_auto.py
```

This opens the **Tkinter UI** with:

- Live webcam feed  
- EAR calculation  
- CNN eye-state prediction  
- **Hybrid Awake / Blink / Drowsy decision**  
- Session logging & PDF generation  

🟢 **This is the only script needed to run the system.**

---

# 📘 What Each File Does

## 🎯 Main Applications
| File | Description |
|------|-------------|
| **drowsiness_hybrid_ui_combo_auto.py** | ⭐ Main Hybrid System (EAR + CNN + Auto Calibration + UI + Reports) |
| drowsiness_ear_ui_v2.1.py | EAR-only legacy version |

---

## 🧠 Training Scripts (optional)

Users who want to train their own models can use:

| File | Description |
|------|-------------|
| train_mobilenet_eye_cnn_v2.py | Trains MobileNetV2 on open/closed eyes |
| finetune_mobilenet_eye_cnn_v2.py | Fine-tunes (unfreezes last 75 layers) |

---

## 📊 Evaluation Scripts

| File | Description |
|------|-------------|
| evaluate_cnn_model.py | CNN accuracy, confusion matrix, ROC |
| evaluate_hybrid_system_template.py | Evaluate EAR+CNN hybrid with ground-truth labels |

---

## 🧩 Model Files

| File | Description |
|------|-------------|
| models/eye_mobilenet_v2.h5 | Base MobileNet pretrained model |
| **models/eye_mobilenet_finetuned_v2.h5** | ⭐ Final fine-tuned model used in the Hybrid system |

---

## 📦 Supporting Folders
| Folder | Purpose |
|--------|----------|
| sample_dataset/ | Tiny open/closed dataset for reference |
| dataset_structure.md | Explains how to download/prepare MRL dataset |
| session_results/ | Auto-saved session statistics, logs, charts, PDFs |
| images/ | Architecture diagrams, training plots, UI previews |

---

# 🗂 Execution Order (Very Clear)

## ✔ If you only want to RUN the system (most users)

1. Keep the pre-trained model in `models/`  
2. Install dependencies  
3. Run the main script:

```bash
python drowsiness_hybrid_ui_combo_auto.py
```

That's it.

---

## ✔ If you want to TRAIN your own model (optional)

1. Download the MRL Eye Dataset  
2. Follow folder structure in `dataset_structure.md`  
3. Train base model:

```bash
python train_mobilenet_eye_cnn_v2.py
```

4. Fine-tune:

```bash
python finetune_mobilenet_eye_cnn_v2.py
```

5. Replace the `.h5` model in `models/` (optional)  
6. Run UI normally:

```bash
python drowsiness_hybrid_ui_combo_auto.py
```

---

## ✔ If you want to EVALUATE the Hybrid System

1. Export frames from video  
2. Create CSV with labels  
3. Update script paths  
4. Run:

```bash
python evaluate_hybrid_system_template.py
```

---

# 📊 Dataset

This project uses the **MRL Eye Dataset** (476 MB):

🔗 https://www.kaggle.com/datasets/tauilabdelilah/mrl-eye-dataset

Not included due to size.  
A **sample dataset** is included to show proper structure.

More details → `dataset_structure.md`

---

# 🧠 Model Architecture

### MobileNetV2 (Fine-Tuned)
- Input: 160×160 RGB eye crops  
- Stage 1: Train top layers  
- Stage 2: Unfreeze last 75 layers  
- Output: open/closed  
- Final performance: **~95–96% accuracy**

Diagrams located in:
```
images/architecture/
```

---

# 🖥 UI Preview

Located in:
```
images/ui/UI.png
```

The UI displays:

- EAR  
- CNN probability  
- Hybrid decision  
- Sliding window closed ratio  
- EAR plot, histogram, pie chart  
- Stats table & PDF export  

---

# 📈 Training Results

- training_plot_v2.png  
- finetune_plot_v2.png  

---

# 🔍 Known Limitations

- Glass glare may reduce CNN accuracy  
- No yawn (MAR) detection  
- No head-pose tracking  
- Requires frontal face  
- Webcam quality affects EAR precision  

---

# 🚀 Future Improvements

- Add yawn detection  
- Add head pose estimation  
- IR camera support  
- Vision Transformer models  
- Multi-sensor fusion  

---

# 📄 License

MIT License — free to use, modify, distribute.

---

# ✨ Author

**Saketh Gudi**

