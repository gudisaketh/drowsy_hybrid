🚗 Hybrid Driver Drowsiness Detection Using EAR, MediaPipe FaceMesh & CNN

A real-time hybrid driver drowsiness detection system combining:
	•	Eye Aspect Ratio (EAR) via MediaPipe FaceMesh
	•	Deep-learning eye-state CNN (MobileNetV2 fine-tuned)
	•	Hybrid fusion logic for high reliability
	•	Auto EAR calibration, single-eye fallback, glasses support,
	•	Full Tkinter UI, session analytics, and PDF reporting

Designed for robust real-world performance with eyeglasses, glare, occlusion, and variable lighting.

⸻

📌 Key Features
	•	✅ Hybrid EAR + CNN fusion model
	•	✅ Auto EAR threshold calibration
	•	✅ Single-eye fallback (works even with occlusion)
	•	✅ Glare & glasses-friendly detection
	•	✅ Real-time UI with EAR curve, blink detection, CNN probabilities
	•	✅ Full analytics: histogram, pie chart, session stats
	•	✅ Automatic PDF session report generation
	•	✅ Training + fine-tuning scripts included
	•	✅ Evaluation scripts for CNN & hybrid model

⸻

📂 Project Structure

drowsy_hybrid/
│
├── drowsiness_hybrid_ui_combo_auto.py      # Main Hybrid EAR+CNN App (Auto Calibration)
├── drowsiness_ear_ui_v2.1.py               # EAR-only app (legacy)
│
├── train_mobilenet_eye_cnn_v2.py           # Train MobileNetV2 on open/closed eyes
├── finetune_mobilenet_eye_cnn_v2.py        # Fine-tune last 75 layers
├── evaluate_cnn_model.py                   # CNN evaluation (accuracy, CM, ROC)
├── evaluate_hybrid_system_template.py       # Template for hybrid evaluation
│
├── models/
│   ├── eye_mobilenet_v2.h5                 # Base feature extractor model
│   └── eye_mobilenet_finetuned_v2.h5       # Fully fine-tuned final model
│
├── images/
│   ├── system_architecture.png
│   ├── process_flow.png
│   ├── cnn_architecture.png
│   ├── UI.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── training_plot_v2.png
│   └── finetune_plot_v2.png
│
├── dataset_structure.md                    # Instructions for dataset setup
├── sample_dataset/                         # Small open/closed sample set
│
├── session_results/                        # Generated logs, charts, PDF reports
├── evaluation_results/                     # Model evaluation outputs
│
└── README.md


⸻

📦 Installation

🔧 Option 1 — Create virtual environment (recommended)

python3 -m venv drowsy_env
source drowsy_env/bin/activate    # macOS / Linux
drowsy_env\Scripts\activate       # Windows

Install dependencies:

pip install -r requirements.txt


⸻

🔧 Option 2 — Install manually

pip install tensorflow==2.12.0
pip install opencv-python==4.8.1.78
pip install mediapipe==0.10.9
pip install pillow matplotlib numpy pandas scikit-learn simpleaudio
pip install protobuf==3.20.3 six==1.16.0 h5py==3.8.0

💡 Uses exact package versions used during development for compatibility.

⸻

📊 Dataset

We use the MRL Eye Dataset (476 MB):

🔗 https://www.kaggle.com/datasets/tauilabdelilah/mrl-eye-dataset

Due to size limits, it is NOT included in the repository.

A small sample_dataset/ is included to show correct structure:

sample_dataset/
└── train/
     ├── open/
     └── closed/

Full instructions:
📄 dataset_structure.md

⸻

🧠 Model Information

MobileNetV2 Architecture (Fine-Tuned)
	•	Input: 160×160×3 RGB eye crops
	•	Stage 1: Train top layers only
	•	Stage 2: Unfreeze last 75 layers for fine-tuning
	•	Output: Open (0) / Closed (1)
	•	Final performance: ~95–96% accuracy

⸻

▶️ Running the Hybrid Application

python drowsiness_hybrid_ui_combo_auto.py

Includes:
	•	Auto EAR calibration
	•	Hybrid EAR + CNN fusion
	•	Real-time detection
	•	Alerts
	•	PDF reporting
	•	EAR chart, histogram, pie chart

⸻

🧪 Evaluate the Hybrid Model

python evaluate_hybrid_system_template.py

You must provide:
	•	evaluation frames folder
	•	CSV with labels
	•	update paths inside script

⸻

🖼 System Diagrams

System Architecture

Process Flow

CNN Architecture


⸻

📈 Model Training Results

Base Training

Fine-Tuning


⸻

🖥 UI Preview


⸻

🚀 UI Features
	•	Live EAR
	•	CNN open/closed probability
	•	Hybrid decision:
	•	Awake
	•	Blink
	•	Drowsy
	•	Session stats:
	•	EAR time-series graph
	•	EAR histogram
	•	Awake/drowsy time distribution
	•	Stats table
	•	CSV
	•	PDF report

⸻

🔍 Known Limitations
	•	Strong glare on eyeglasses can affect predictions
	•	No mouth/yawn detection
	•	No head-pose monitoring
	•	Best with frontal face position
	•	Webcam quality affects EAR precision

⸻

📌 Future Improvements
	•	Add yawn (MAR) detection
	•	Add head pose estimation
	•	Use IR-based camera
	•	Replace CNN with ViT or EfficientNet
	•	Add multi-sensor fusion (steering wheel, HRV)

⸻

📝 License

MIT License — free to use, modify, and distribute.

⸻

✨ Author

Saketh Gudi
