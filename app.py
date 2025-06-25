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
import json

# --- Config ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Model Registry ---
MODEL_REGISTRY_PATH = os.path.join(app.root_path, 'models', 'model_info.json')
with open(MODEL_REGISTRY_PATH, 'r') as f:
    MODEL_REGISTRY = json.load(f)

MODEL_CACHE = {}


@app.route('/get-models', methods=['GET'])
def get_models():
    model_list = [
        {'id': model_id, 'name': meta['name']}
        for model_id, meta in MODEL_REGISTRY.items()
    ]
    return jsonify(model_list)

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

    # --- Get selected model ID ---
    model_id = request.form.get('model', 'unet')
    model_config = MODEL_REGISTRY.get(model_id)
    if not model_config:
        return f"Invalid model ID: {model_id}", 400

    model_path = os.path.join(app.root_path, 'models', model_config['file'])
    input_size = tuple(model_config['input_size'])

    # --- Save file ---
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # --- Validate it's a fundus image ---
        if not is_fundus_image(file_path):
            return render_template_string('''
                <h3 style="color:red;">Invalid image. Please upload a valid fundus scan.</h3>
                <a href="/">← Back to Upload</a>
            ''')

        print(f"✅ Fundus image validated. Using model: {model_id}")

        # --- Open + preprocess image for segmentation ---
        img = Image.open(file_path).convert("RGB")
        img_resized = img.resize(input_size)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # --- Load segmentation model ---
        if model_id not in MODEL_CACHE:
            print(f"🧠 Loading model '{model_id}' from disk...")
            MODEL_CACHE[model_id] = tf.keras.models.load_model(
                model_path,
                custom_objects={'iou_seg': losses.iou_seg, 'dice': losses.dice}
            )
        else:
            print(f"⚡ Using cached model: {model_id}")

        seg_model = MODEL_CACHE[model_id]


        # --- Run segmentation ---
        prediction = seg_model.predict(img_array)
        binary_mask = (prediction > 0.5).astype(np.uint8)

        print("📈 Output shape:", binary_mask.shape)  # Should be (1, H, W, 1) or similar

        # --- (Optional for now) Save processed output ---
        output_img = Image.fromarray(binary_mask.squeeze() * 255).convert("L")
        output_resized = output_img.resize((2048, 2048))
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{filename}")
        output_resized.save(output_path)

        return render_template_string('''
        <h3>Segmentation complete using model: {{ model_id }}</h3>
        <p>Original Image:</p>
        <img src="/show_image/{{ filename }}" width="400"><br><br>
        <p>Processed Output:</p>
        <img src="/show_image/{{ processed_filename }}" width="400"><br><br>
        <a href="/">← Back to Upload</a>
    ''', model_id=model_id, filename=filename, processed_filename=f"processed_{filename}")


    except Exception as e:
        return f"Error during processing: {e}", 500

@app.route('/show_image/<filename>')
def show_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

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
