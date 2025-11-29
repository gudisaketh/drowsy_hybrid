"""
Fine-Tune MobileNetV2 Eye Classifier - Version 2 (Works with your saved model)
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os

# -------------------------------
# SETTINGS
# -------------------------------
IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 15

DATASET_PATH = "/Users/sakethgudi/drowsy_hybrid/dataset"

print("Loading base model eye_mobilenet_v2.h5 ...")
model = load_model("eye_mobilenet_v2.h5")

print("\n✔ Model loaded successfully.")
print(f"Total layers in model: {len(model.layers)}")

# -------------------------------
# UNFREEZE LAST 75 LAYERS
# -------------------------------
UNFREEZE_COUNT = 75
print(f"\nUnfreezing last {UNFREEZE_COUNT} layers ...")

for layer in model.layers[-UNFREEZE_COUNT:]:
    layer.trainable = True

# Compile after unfreezing
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# DATA GENERATORS
# -------------------------------
print("\nLoading dataset...")

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.25,
    height_shift_range=0.25,
    brightness_range=(0.5, 1.5),
    shear_range=0.15,
    zoom_range=0.20,
    horizontal_flip=True,
)

train_gen = datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# -------------------------------
# CALLBACKS
# -------------------------------
checkpoint = ModelCheckpoint(
    "eye_mobilenet_finetuned_v2_best.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    min_lr=1e-6
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

# -------------------------------
# TRAINING
# -------------------------------
print("\nStarting fine-tuning...\n")

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[checkpoint, lr_scheduler, early_stop]
)

# -------------------------------
# SAVE FINAL MODEL
# -------------------------------
model.save("eye_mobilenet_finetuned_v2.h5")
print("\n✔ Saved final model: eye_mobilenet_finetuned_v2.h5")

# -------------------------------
# SAVE ACCURACY PLOT
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label="train_acc")
plt.plot(history.history['val_accuracy'], label="val_acc")
plt.title("Fine-Tuned Accuracy")
plt.legend()
plt.savefig("finetune_plot_v2.png")

print("✔ Saved plot: finetune_plot_v2.png")
