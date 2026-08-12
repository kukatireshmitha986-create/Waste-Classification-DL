import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# ==========================================
# PROJECT SETTINGS
# ==========================================

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_DIR = "DATASET/TRAIN"
TEST_DIR = "DATASET/TEST"

# ==========================================
# CHECK DATASET
# ==========================================

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

print("Dataset folders found successfully!")

# ==========================================
# DATA PREPROCESSING
# ==========================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

# ==========================================
# TRAINING DATA
# ==========================================

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

# ==========================================
# VALIDATION DATA
# ==========================================

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=True
)

# ==========================================
# TEST DATA
# ==========================================

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ==========================================
# DISPLAY DATASET INFORMATION
# ==========================================

print("\n==========================================")
print("DATASET INFORMATION")
print("==========================================")

print("Training images:", train_generator.samples)
print("Validation images:", validation_generator.samples)
print("Test images:", test_generator.samples)

print("\nClass labels:")
print(train_generator.class_indices)

# ==========================================
# BUILD CNN MODEL
# ==========================================

model = models.Sequential([
    
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    # First Convolution Block
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Second Convolution Block
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Third Convolution Block
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Flatten
    layers.Flatten(),

    # Fully Connected Layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Dropout
    layers.Dropout(0.5),

    # Output Layer
    layers.Dense(
        1,
        activation="sigmoid"
    )
])

# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# DISPLAY MODEL
# ==========================================

print("\n==========================================")
print("CNN MODEL")
print("==========================================")

model.summary()

# ==========================================
# TRAIN MODEL
# ==========================================

print("\n==========================================")
print("STARTING TRAINING")
print("==========================================")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# ==========================================
# EVALUATE MODEL
# ==========================================

print("\n==========================================")
print("EVALUATING MODEL")
print("==========================================")

test_loss, test_accuracy = model.evaluate(
    test_generator
)

print("\n==========================================")
print("TEST RESULTS")
print("==========================================")

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# ==========================================
# SAVE MODEL
# ==========================================

model.save("waste_classifier.keras")

print("\nModel saved successfully!")
print("File: waste_classifier.keras")

# ==========================================
# SAVE ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    "accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# SAVE LOSS GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    "loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n==========================================")
print("PROJECT FILES CREATED")
print("==========================================")

print("✓ waste_classifier.keras")
print("✓ accuracy.png")
print("✓ loss.png")

print("\nTraining completed successfully!")