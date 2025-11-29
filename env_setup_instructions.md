# 🛠️ Environment Setup Instructions

This document explains how to set up a clean and fully compatible Python environment for running the **Hybrid EAR + CNN Driver Drowsiness Detection System**.

The project was developed and tested using:

- **Python 3.10.x**
- **TensorFlow 2.12.0**
- **MediaPipe 0.10.9**
- **OpenCV 4.8.1**
- **macOS (M1/M2/M3) and Windows 11**

Follow the steps below to ensure reproducibility.

---

# 1️⃣ Install Python 3.10

## macOS
Download from the official site:  
https://www.python.org/downloads/release/python-3100/

Or via Homebrew:

```bash
brew install python@3.10
brew link python@3.10 --force
```

## Windows
Download the installer:  
https://www.python.org/downloads/release/python-3100/

⚠️ Make sure to enable:  
**“Add Python to PATH”**

## Linux

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

---

# 2️⃣ Create a Virtual Environment

A clean environment prevents version conflicts.

## macOS / Linux

```bash
python3.10 -m venv drowsy_env
source drowsy_env/bin/activate
```

## Windows (PowerShell)

```powershell
python -m venv drowsy_env
drowsy_env\Scripts\activate
```

Once activated, your terminal should show:

```
(drowsy_env) $
```

---

# 3️⃣ Upgrade pip (recommended)

```bash
pip install --upgrade pip
```

---

# 4️⃣ Install Dependencies

Install everything using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

This installs exact versions used during development:

- TensorFlow 2.12.0  
- OpenCV 4.8.1.78  
- MediaPipe 0.10.9  
- numpy 1.23.5  
- h5py 3.8.0  
- scikit-learn, pandas, matplotlib, Pillow  
- protobuf 3.20.3 (required by MediaPipe)

---

# 5️⃣ Verify Installation

Run:

```bash
python -c "import cv2, mediapipe, tensorflow as tf; print('OK')"
```

Expected output:

```
OK
```

If TensorFlow gives a warning about Apple M-series GPU, it is safe to ignore.

---

# 6️⃣ Run the Application

To start the hybrid drowsiness detection system:

```bash
python drowsiness_hybrid_ui_combo_auto.py
```

To run the EAR-only version:

```bash
python drowsiness_ear_ui_v2.1.py
```

---

# 7️⃣ Run Training or Evaluation Scripts (Optional)

### Train base CNN:

```bash
python train_mobilenet_eye_cnn_v2.py
```

### Fine-tune CNN:

```bash
python finetune_mobilenet_eye_cnn_v2.py
```

### Evaluate CNN:

```bash
python evaluate_cnn_model.py
```

### Evaluate Hybrid System:

```bash
python evaluate_hybrid_system_template.py
```

---

# 8️⃣ Deactivate Environment (Optional)

```bash
deactivate
```

---

# 🎉 Environment Setup Complete

Your system is now fully configured to run:

- The Hybrid EAR + CNN UI  
- Model training  
- Model evaluation  
- PDF session analytics  

If you encounter installation issues, feel free to open an Issue on GitHub.
