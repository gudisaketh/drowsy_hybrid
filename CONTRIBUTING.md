# 🤝 Contributing to Hybrid EAR + CNN Driver Drowsiness Detection

Thank you for your interest in contributing!  
This project welcomes improvements, bug fixes, new features, model training enhancements, and evaluation work.

Please follow the guidelines below to keep contributions consistent and high quality.

---

# 1️⃣ Reporting Issues

If you discover a:

- ❗ Bug  
- 📌 Incorrect model prediction  
- 🔧 Environment or installation problem  
- 🚀 Feature suggestion  

Please open a GitHub Issue with:

1. A clear title  
2. Steps to reproduce  
3. Screenshots (if UI-related)  
4. System details (OS, Python version, package versions)

---

# 2️⃣ Development Environment Setup

Create a clean environment (Python 3.10 recommended):

```bash
python3.10 -m venv drowsy_env
source drowsy_env/bin/activate   # macOS / Linux
drowsy_env\Scripts\activate      # Windows
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

# 3️⃣ Code Style Guidelines

Please follow these rules when contributing:

### Naming
- Use `snake_case` for functions and variables  
- Use `CamelCase` for classes  
- Keep filenames descriptive (e.g., `evaluate_cnn_model.py`)

### Formatting
- Use **4-space indentation**
- Use **PEP8** guidelines when possible
- Add clear comments for logic-heavy sections

### File Structure
Keep new files inside these folders when appropriate:

```
models/               # Store trained model files (.h5)
evaluation_results/   # Confusion matrix, ROC, metrics, text reports
session_results/      # UI-generated session logs & PDF reports
sample_dataset/       # Mini dataset examples
```

---

# 4️⃣ Submitting Pull Requests (PR)

To submit a PR:

1. Fork the repository  
2. Create a new branch:  
   ```bash
   git checkout -b feature/my-feature
   ```
3. Make changes  
4. Add tests or sample inputs if needed  
5. Commit with a clear message:

   ```
   git commit -m "Fix EAR threshold instability for glasses"
   ```

6. Push to your fork:  
   ```bash
   git push origin feature/my-feature
   ```
7. Open a Pull Request describing:
   - What changed  
   - Why it helps  
   - Screenshots / results when applicable  

All PRs are reviewed before merging.

---

# 5️⃣ Contributing to the ML Model

To train the eye-state CNN:

```bash
python train_mobilenet_eye_cnn_v2.py
```

Fine-tune the model:

```bash
python finetune_mobilenet_eye_cnn_v2.py
```

### Guidelines
- Do **not** commit large datasets
- Keep large `.h5` files inside `/models/`
- Always save evaluation outputs to `/evaluation_results/`
- Document your training hyperparameters in the PR

---

# 6️⃣ Evaluating the Hybrid Drowsiness System

To run hybrid evaluation:

```bash
python evaluate_hybrid_system_template.py
```

Contributors may create their own labeled test frames & CSV.

---

# 7️⃣ Where to Add New Features

| Feature Type                | Folder / File                  |
|-----------------------------|--------------------------------|
| UI improvements             | `drowsiness_hybrid_ui_combo_auto.py` |
| EAR logic enhancements      | Same UI file or helper modules |
| CNN improvements            | `train_mobilenet_eye_cnn_v2.py` |
| Evaluation / metrics        | `evaluate_*` scripts           |
| Diagrams / images           | `images/` folder               |

---

# 8️⃣ Code of Conduct

- Be respectful and constructive  
- Do not upload personal data or identifiable videos  
- Keep large datasets outside the repo  

---

# 🙏 Thank You!

Your contributions help make this project more accurate, stable, and useful for real-world safety applications.

If you have questions, feel free to open an Issue or contact the maintainer.
