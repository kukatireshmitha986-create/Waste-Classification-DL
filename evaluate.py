import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ==========================================
# SETTINGS
# ==========================================

IMG_SIZE = 128
BATCH_SIZE = 32

TEST_DIR = "DATASET/TEST"

MODEL_PATH = "waste_classifier.keras"

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ==========================================
# LOAD TEST DATA
# ==========================================

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ==========================================
# CLASS NAMES
# ==========================================

class_names = list(test_generator.class_indices.keys())

print("\nClass names:")
print(class_names)

# ==========================================
# MODEL EVALUATION
# ==========================================

test_loss, test_accuracy = model.evaluate(
    test_generator
)

print("\n==========================================")
print("MODEL EVALUATION")
print("==========================================")

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# ==========================================
# PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

predictions = model.predict(
    test_generator,
    verbose=1
)

# Convert probabilities into class predictions
predicted_classes = (
    predictions.ravel() >= 0.5
).astype(int)

true_classes = test_generator.classes

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names
    )
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(cm)

# ==========================================
# DISPLAY CONFUSION MATRIX
# ==========================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot()

plt.title("Waste Classification Confusion Matrix")

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion matrix saved as:")
print("confusion_matrix.png")

print("\nEvaluation completed successfully!")