# Dataset Structure (MRL Eye Dataset + Custom Samples)

This project uses the **MRL Eye Dataset** from Kaggle:

🔗 https://www.kaggle.com/datasets/tauilabdelilah/mrl-eye-dataset

The full dataset is **NOT included** in this repository due to size (~476 MB).  
Instead, a **small sample dataset** is included for demonstration in:

```
sample_dataset/
```

---

## Expected Full Dataset Layout

After downloading and organizing, the full training dataset should be arranged as:

```
dataset/
│
├── train/
│   ├── open/
│   │   ├── img_00001.jpg
│   │   ├── img_00002.jpg
│   │   └── ...
│   └── closed/
│       ├── img_01001.jpg
│       ├── img_01002.jpg
│       └── ...
│
└── test/
    ├── open/
    │   ├── img_05001.jpg
    │   └── ...
    └── closed/
        ├── img_06001.jpg
        └── ...
```

---

## How to Prepare the Dataset

1. Download the dataset from Kaggle.
2. Extract it.
3. Sort images into:
   - `open/` → eyes open
   - `closed/` → eyes closed
4. Place folders into:

```
dataset/train/open
dataset/train/closed
dataset/test/open
dataset/test/closed
```

---

## Sample Dataset Provided

The repository includes a small example dataset:

```
sample_dataset/
│
├── train/
│   ├── open/
│   └── closed/
│
└── test/
    ├── open/
    └── closed/
```

This sample ONLY demonstrates expected structure and should **not be used for actual training.**

---
