import tensorflow as tf
from PIL import Image
import numpy as np

# --- Config ---
MODEL_PATH = "models/fundus_validator.keras"
IMAGE_PATH = r"C:\Users\hamza\Documents\Projects\Research Projects\my data\Proccesed dataset\test images\test_010.jpg"  # ⬅️ Change this
IMAGE_SIZE = (224, 224)

# --- Load Model ---
model = tf.keras.models.load_model(MODEL_PATH)

# --- Load & Preprocess Image ---
img = Image.open(IMAGE_PATH).convert("RGB")
img = img.resize(IMAGE_SIZE)
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

# --- Predict ---
prediction = model.predict(img_array)[0][0]

print(f"Prediction Score: {prediction:.4f}")
if prediction >= 0.5:
    print("✅ This is likely a FUNDUS image.")
else:
    print("❌ This is likely NOT a fundus image.")
