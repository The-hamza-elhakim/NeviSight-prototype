from flask import Flask, request, render_template, render_template_string, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
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
import json
import base64
from datetime import datetime

# --- Config ---
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}
METADATA_FILE = 'metadata_records.json'

app = Flask(__name__)

# --- Model Registry ---
MODEL_REGISTRY_PATH = os.path.join(app.root_path, 'models', 'model_info.json')
with open(MODEL_REGISTRY_PATH, 'r') as f:
    MODEL_REGISTRY = json.load(f)

MODEL_CACHE = {}

# --- Load fundus validator model once ---
FUNDUS_MODEL_PATH = "models/fundus_validator.keras"
classifier_model = tf.keras.models.load_model(FUNDUS_MODEL_PATH)

# --- Utility Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_dicom_bytes(file_bytes):
    try:
        pydicom.dcmread(io.BytesIO(file_bytes), stop_before_pixels=True)
        return True
    except Exception:
        return False

def preprocess_image_for_validation(file_bytes, ext):
    if ext == 'dcm' or is_dicom_bytes(file_bytes):
        dcm = pydicom.dcmread(io.BytesIO(file_bytes))
        arr = dcm.pixel_array
        arr = ((arr - arr.min()) / (np.ptp(arr) + 1e-6) * 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
    else:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def is_fundus_image(file_bytes, ext):
    processed_img = preprocess_image_for_validation(file_bytes, ext)
    prediction = classifier_model.predict(processed_img)[0][0]
    return prediction >= 0.5

def extract_dicom_metadata(file_bytes):
    dcm = pydicom.dcmread(io.BytesIO(file_bytes))
    metadata = {
        "mrn": getattr(dcm, "PatientID", None),
        "age": getattr(dcm, "PatientAge", None),
        "sex": getattr(dcm, "PatientSex", None),
        "study_date": getattr(dcm, "StudyDate", None),
        "eye": getattr(dcm, "Laterality", None) or getattr(dcm, "ImageLaterality", None)
    }
    return metadata

def append_metadata_record(record):
    # Ensure file exists
    if not os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'w') as f:
            json.dump([], f)

    with open(METADATA_FILE, 'r+') as f:
        data = json.load(f)
        data.append(record)
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-models', methods=['GET'])
def get_models():
    model_list = [
        {'id': model_id, 'name': meta['name']}
        for model_id, meta in MODEL_REGISTRY.items()
    ]
    return jsonify(model_list)

@app.route('/preload-model/<model_id>', methods=['GET'])
def preload_model(model_id):
    if model_id not in MODEL_REGISTRY:
        return jsonify({'status': 'error', 'message': 'Invalid model ID'}), 400

    if model_id not in MODEL_CACHE:
        print(f"🧠 Preloading model '{model_id}'...")
        model_path = os.path.join(app.root_path, 'models', MODEL_REGISTRY[model_id]['file'])
        MODEL_CACHE[model_id] = tf.keras.models.load_model(
            model_path,
            custom_objects={'iou_seg': losses.iou_seg, 'dice': losses.dice}
        )
    else:
        print(f"⚡ Model '{model_id}' already cached.")

    return jsonify({'status': 'ok', 'message': f'Model {model_id} ready.'})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if not allowed_file(file.filename):
        return "Unsupported file type", 400

    model_id = request.form.get('model', 'unet')
    model_config = MODEL_REGISTRY.get(model_id)
    if not model_config:
        return f"Invalid model ID: {model_id}", 400

    input_size = tuple(model_config['input_size'])
    ext = file.filename.rsplit('.', 1)[1].lower()
    file_bytes = file.read()

    try:
        if not is_fundus_image(file_bytes, ext):
            return render_template_string('''
                <h3 style="color:red;">Invalid image. Please upload a valid fundus scan.</h3>
                <a href="/">← Back to Upload</a>
            ''')

        print(f"✅ Fundus image validated. Using model: {model_id}")

        # --- Collect metadata ---
        if ext == 'dcm':
            metadata = extract_dicom_metadata(file_bytes)
            metadata["source"] = "DICOM"
        else:
            metadata = {
                "mrn": request.form.get('mrn'),
                "age": request.form.get('age'),
                "sex": request.form.get('sex'),
                "study_date": request.form.get('study_date'),
                "eye": request.form.get('eye'),
                "source": "Manual"
            }

        metadata["uploaded_at"] = datetime.utcnow().isoformat()

        append_metadata_record(metadata)
        print("📝 Metadata saved:", metadata)

        # --- Process image ---
        if ext == 'dcm' or is_dicom_bytes(file_bytes):
            dcm = pydicom.dcmread(io.BytesIO(file_bytes))
            arr = dcm.pixel_array
            arr = ((arr - arr.min()) / (np.ptp(arr) + 1e-6) * 255).astype(np.uint8)
            img = Image.fromarray(arr).convert("RGB")
        else:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

        img_resized = img.resize(input_size)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if model_id not in MODEL_CACHE:
            print(f"🧠 Loading model '{model_id}' from disk...")
            model_path = os.path.join(app.root_path, 'models', model_config['file'])
            MODEL_CACHE[model_id] = tf.keras.models.load_model(
                model_path,
                custom_objects={'iou_seg': losses.iou_seg, 'dice': losses.dice}
            )
        else:
            print(f"⚡ Using cached model: {model_id}")

        seg_model = MODEL_CACHE[model_id]
        prediction = seg_model.predict(img_array)
        binary_mask = (prediction > 0.5).astype(np.uint8)

        output_img = Image.fromarray(binary_mask.squeeze() * 255).convert("L")
        output_resized = output_img.resize((2048, 2048))

        orig_buf = io.BytesIO()
        img.save(orig_buf, format='PNG')
        orig_base64 = base64.b64encode(orig_buf.getvalue()).decode('utf-8')

        proc_buf = io.BytesIO()
        output_resized.save(proc_buf, format='PNG')
        proc_base64 = base64.b64encode(proc_buf.getvalue()).decode('utf-8')

        return render_template_string('''
            <h3>Segmentation complete using model: {{ model_id }}</h3>
            <p>Original Image:</p>
            <img src="data:image/png;base64,{{ orig_base64 }}" width="400"><br><br>
            <p>Processed Output:</p>
            <img src="data:image/png;base64,{{ proc_base64 }}" width="400"><br><br>
            <a href="/">← Back to Upload</a>
        ''', model_id=model_id, orig_base64=orig_base64, proc_base64=proc_base64)

    except Exception as e:
        return f"Error during processing: {e}", 500


# --- Run ---
if __name__ == "__main__":
    app.run(debug=True)
