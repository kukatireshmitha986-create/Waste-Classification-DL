import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("waste_classifier.keras")

print("Model loaded successfully!")

# Note:
# Training history is not stored inside the saved model,
# so this file is only used to verify the saved model.

print("\nModel Summary:")
model.summary()