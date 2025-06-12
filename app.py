from flask import Flask, request, render_template, redirect, url_for, send_from_directory, render_template_string, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter
import pydicom
import io
import os
import time
import tensorflow as tf
import numpy as np
import keras
from tensorflow.keras.models import load_model
from keras_unet_collection import models, losses
from keras.optimizers import Adam

# --- Config ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Load fundus validator model once ---
FUNDUS_MODEL_PATH = "models/fundus_validator.keras"
classifier_model = tf.keras.models.load_model(FUNDUS_MODEL_PATH)

# --- Utility Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_dicom(file_path):
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
    except Exception:
        return False

def preprocess_image_for_validation(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    
    if ext == 'dcm' or is_dicom(file_path):
        dcm = pydicom.dcmread(file_path)
        arr = dcm.pixel_array
        arr = ((arr - arr.min()) / (np.ptp(arr) + 1e-6) * 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
    else:
        img = Image.open(file_path).convert("RGB")
    
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def is_fundus_image(file_path):
    processed_img = preprocess_image_for_validation(file_path)
    prediction = classifier_model.predict(processed_img)[0][0]
    return prediction >= 0.5

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if not allowed_file(file.filename):
        return "Unsupported file type", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        if not is_fundus_image(file_path):
            return render_template_string('''
                <h3 style="color:red;">Invalid image. Please upload a valid fundus scan.</h3>
                <a href="/">← Back to Upload</a>
            ''')

        print("✅ Fundus image validated successfully.")

        ext = filename.rsplit('.', 1)[1].lower()
        if ext == 'dcm' or is_dicom(file_path):
            print("Uploaded file is a DICOM image")
            return f"Successfully uploaded and validated DICOM fundus image: {filename}"
        else:
            img = Image.open(file_path)
            img.verify()
            print("Uploaded file is a standard image")
            return f"Successfully uploaded and validated image file: {filename}"

    except Exception as e:
        return f"Error during validation: {e}", 400

@app.route('/validate-image', methods=['POST'])
def validate_image():
    if 'file' not in request.files:
        print("🚫 No file in request.")
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        print("🚫 No filename.")
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400

    if not allowed_file(file.filename):
        print("🚫 Unsupported file type:", file.filename)
        return jsonify({'status': 'error', 'message': 'Unsupported file type'}), 400

    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{filename}")
    file.save(temp_path)

    try:
        print(f"🔍 Validating fundus image: {filename}")
        is_valid = is_fundus_image(temp_path)
        os.remove(temp_path)

        if is_valid:
            print("✅ Fundus image accepted.")
            return jsonify({'status': 'ok'})
        else:
            print("❌ Not a valid fundus image.")
            return jsonify({'status': 'invalid', 'message': 'This image is not a valid fundus scan'}), 200

    except Exception as e:
        print("❌ Exception during validation:", e)
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- Run ---
if __name__ == "__main__":
    app.run(debug=True)
