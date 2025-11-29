"""
Train MobileNetV2 Eye Classifier - Version 2 (High Accuracy)
Achieves ~85-90% before fine-tuning.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# -------------------------------
# CONFIG
# -------------------------------
IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 20

DATASET_PATH = "/Users/sakethgudi/drowsy_hybrid/dataset"

# -------------------------------
# Data Generators (Heavy Augmentations)
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.25,
    height_shift_range=0.25,
    brightness_range=(0.5, 1.5),
    shear_range=0.2,
    zoom_range=0.20,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    subset="training"
)

val_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
    subset="validation"
)

# -------------------------------
# Model Definition
# -------------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # Freeze for first phase

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# LR Scheduler
# -------------------------------
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.4,
    patience=3,
    min_lr=1e-6
)

# -------------------------------
# Training
# -------------------------------
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[lr_scheduler]
)

# -------------------------------
# Save model
# -------------------------------
model.save("eye_mobilenet_v2.h5")
print("\nSaved model as eye_mobilenet_v2.h5")

# -------------------------------
# Save plot
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label="train_acc")
plt.plot(history.history['val_accuracy'], label="val_acc")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.savefig("training_plot_v2.png")
print("Saved training_plot_v2.png")
