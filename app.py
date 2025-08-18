from flask import Flask, request, render_template, render_template_string, jsonify, make_response
from werkzeug.utils import secure_filename
from my_image_processing_library import draw_mask_contour_on_image
from PIL import Image
import pydicom
import io
import os
import pdfkit
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from keras_unet_collection import models, losses
import json
import base64
import cv2
from datetime import datetime
from uuid import uuid4
import platform

if platform.system().lower() != "windows":
    from weasyprint import HTML, CSS


# --- Config ---
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}
METADATA_FILE = 'metadata_records.json'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 megabytes

PDF_CACHE = {}

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

def preprocess_image_for_validation(file_bytes, _ext_unused=None):
    img = pil_from_bytes(file_bytes)
    img = img.resize((224, 224))
    arr = (np.array(img).astype(np.float32) / 255.0)[None, ...]
    return arr


def is_fundus_image(file_bytes, ext):
    processed_img = preprocess_image_for_validation(file_bytes, ext)
    prediction = classifier_model.predict(processed_img)[0][0]
    is_valid = prediction >= 0.5
    return is_valid, processed_img


def extract_dicom_metadata(file_bytes):
    dcm = pydicom.dcmread(io.BytesIO(file_bytes))
    # Get raw study date string
    raw_date = getattr(dcm, "StudyDate", None)

    # Format DICOM date YYYYMMDD -> YYYY-MM-DD
    formatted_date = None
    if raw_date and len(raw_date) == 8:
        formatted_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"

    metadata = {
        "mrn": getattr(dcm, "PatientID", None),
        "age": getattr(dcm, "PatientAge", None),
        "sex": getattr(dcm, "PatientSex", None),
        "study_date": formatted_date,
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

def _norm_age(v):
    # DICOM often gives '046Y', '012M', etc. Keep just the number.
    if isinstance(v, str) and v and v[-1] in ("Y", "M", "W", "D"):
        return v[:-1]
    return v if v not in ("", None) else "N/A"

def pil_from_bytes(file_bytes):
    """
    Returns a PIL.Image in RGB, handling both DICOM and regular image files.
    Applies the same normalization path for DICOM (min-max to 0–255, uint8).
    """
    if is_dicom_bytes(file_bytes):
        dcm = pydicom.dcmread(io.BytesIO(file_bytes))
        arr = dcm.pixel_array.astype(np.float32)
        # Min-max normalize to 0–1 to handle varied DICOM ranges
        rng = np.ptp(arr)
        if rng == 0:
            arr = np.zeros_like(arr, dtype=np.uint8)
        else:
            arr = ((arr - arr.min()) / (rng + 1e-6) * 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    # PNG/JPG path
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


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
        print(f"Preloading model... '{model_id}'...")
        model_path = os.path.join(app.root_path, 'models', MODEL_REGISTRY[model_id]['file'])
        MODEL_CACHE[model_id] = tf.keras.models.load_model(
            model_path,
            custom_objects={'iou_seg': losses.iou_seg, 'dice': losses.dice}
        )
    else:
        print(f"⚡ Model '{model_id}' already cached.")

    return jsonify({'status': 'ok', 'message': f'Model {model_id} ready.'})

@app.route('/validate-fundus', methods=['POST'])
def validate_fundus():
    if 'file' not in request.files:
        return jsonify({'valid': False, 'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'valid': False, 'error': 'No selected file'}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower()
    file_bytes = file.read()

    print(f"🔍 Validating file: {file.filename} ({ext}), size: {len(file_bytes)} bytes")

    try:
        # Call is_fundus_image once, keep output
        is_valid, _ = is_fundus_image(file_bytes, ext)
        print(f"✅ Fundus validation result: {is_valid}")
        return jsonify({'valid': bool(is_valid)})
    except Exception as e:
        print(f"❌ Validation error: {str(e)}")
        return jsonify({'valid': False, 'error': str(e)}), 500



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

    safe_base = secure_filename(file.filename).rsplit(".", 1)[0]

    try:
        is_valid, preprocessed_array = is_fundus_image(file_bytes, ext)
        if not is_valid:
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
                "gender": request.form.get('gender'),
                "study_date": request.form.get('study_date'),
                "eye": request.form.get('eye'),
                "source": "Manual"
            }

        # --- Normalize keys/values (single source of truth) ---
        metadata = {
            **metadata,
            "mrn": metadata.get("mrn") or "N/A",
            "sex": (metadata.get("sex") or metadata.get("gender") or "N/A"),
            "age": _norm_age(metadata.get("age")),
            "eye": metadata.get("eye") or "N/A",
        }

        patient_id = metadata["mrn"]
        age = metadata["age"]
        sex = metadata["sex"]
        eye = metadata["eye"]


        metadata["uploaded_at"] = datetime.utcnow().isoformat()

        append_metadata_record(metadata)
        print("📝 Metadata saved:", metadata)

        # --- Process image ---
        img = pil_from_bytes(file_bytes)

        # Build model input
        img_resized = img.resize(input_size)
        img_array = (np.array(img_resized).astype(np.float32) / 255.0)[None, ...]

        if model_id not in MODEL_CACHE:
            print(f"Loading model... '{model_id}' from disk...")
            model_path = os.path.join(app.root_path, 'models', model_config['file'])
            MODEL_CACHE[model_id] = tf.keras.models.load_model(
                model_path,
                custom_objects={'iou_seg': losses.iou_seg, 'dice': losses.dice}
            )
        else:
            print(f"⚡ Using cached model: {model_id}")

        seg_model = MODEL_CACHE[model_id]

        start_time = time.time()
        prediction = seg_model.predict(img_array)
        end_time = time.time()

        prediction_duration = end_time - start_time
        prediction_time_str = f"{prediction_duration:.2f} seconds" 
        binary_mask = (prediction > 0.5).astype(np.uint8)

        # Basic lesion detection logic
        lesion_detected = np.sum(binary_mask) > 0


        # Resize original to match the processed display dimensions (2048x2048)
        resized_orig = img.resize((2048, 2048))

        # Optional: crop black border if needed using same logic later (or keep as-is)
        resized_orig = resized_orig.convert("RGB")

        # Encode
        orig_buf = io.BytesIO()
        resized_orig.save(orig_buf, format='PNG')
        orig_base64 = base64.b64encode(orig_buf.getvalue()).decode('utf-8')


        # Prepare image for contour overlay
        overlay_input_img = img.resize(input_size)
        overlay_np = np.array(overlay_input_img) / 255.0

        # Make sure shape is (H, W, 3)
        if overlay_np.ndim == 2:
            overlay_np = np.stack([overlay_np]*3, axis=-1)
        elif overlay_np.shape[-1] != 3:
            overlay_np = overlay_np[:, :, :3]

        # Get binary mask resized to model input size
        mask_np = binary_mask.squeeze()

        # Wrap into batch format for draw_contours_on_images (expects lists)
        temp_folder = "static/temp_contours"
        os.makedirs(temp_folder, exist_ok=True)

        # Dummy GT
        dummy_gt = np.zeros_like(mask_np)

        # Upscale original image to 2048x2048 for display
        display_img = img.resize((2048, 2048))
        display_arr = np.array(display_img)

        # Resize binary mask to 2048x2048
        mask_resized = Image.fromarray(mask_np * 255).resize((2048, 2048))
        mask_np_resized = (np.array(mask_resized) > 127).astype(np.uint8)

        # Overlay contour in memory
        contour_image = draw_mask_contour_on_image(display_arr, mask_np_resized)

        # Convert to base64
        proc_buf = io.BytesIO()
        contour_image.save(proc_buf, format='PNG')
        proc_base64 = base64.b64encode(proc_buf.getvalue()).decode("utf-8")

        lesion_detected = np.sum(binary_mask) > 0
        soft_mask = prediction.squeeze()
        conf_region = soft_mask[soft_mask > 0.5]
        avg_confidence = float(np.mean(conf_region)) if conf_region.size > 0 else 0.0

        cache_id = str(uuid4())
        PDF_CACHE[cache_id] = {
            "orig_base64": orig_base64,
            "proc_base64": proc_base64,
            "safe_base": safe_base,
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "prediction_time": prediction_time_str,
            "model_name": model_config["name"],
            "lesion_detected": lesion_detected,
            "eye": eye,
            "avg_confidence": avg_confidence,
        }

        return render_template(
            "result.html",
            orig_base64=orig_base64,
            proc_base64=proc_base64,
            cache_id=cache_id,
            model_name=model_config["name"],
            prediction_time=prediction_time_str,
            lesion_detected=lesion_detected,
            eye=eye,
        )

    
    except Exception as e:
        return f"Error during processing: {e}", 500

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    cache_id = request.form.get('cache_id')
    clinician_notes = request.form.get("clinician_notes", "").strip()
    data = PDF_CACHE.pop(cache_id, None)

    if not data:
        return "Error: PDF data expired or not found.", 400

    html = render_template(
        "report_template.html",
        patient_id=data["patient_id"],
        age=data["age"],
        sex=data["sex"],
        eye=data["eye"],
        exam_date=datetime.utcnow().strftime("%Y-%m-%d"),
        exam_time=datetime.utcnow().strftime("%H:%M:%S"),
        orig_base64=data["orig_base64"],
        proc_base64=data["proc_base64"],
        model_name="Attention U-Net v1.0",
        processing_time=data.get("prediction_time", "N/A"),
        lesion_detected=data["lesion_detected"],
        avg_confidence=data["avg_confidence"],
        clinician_notes=clinician_notes,
        year=datetime.utcnow().year
    )

    
    import pdfkit
    wkhtml_path = os.getenv("WKHTMLTOPDF_PATH", r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
    config = pdfkit.configuration(wkhtmltopdf=wkhtml_path) if os.path.exists(wkhtml_path) else None
    pdf_bytes = pdfkit.from_string(html, False, configuration=config)

    response = make_response(pdf_bytes)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=report_{data["safe_base"]}.pdf'
    return response



# --- Run ---
if __name__ == "__main__":
    app.run(debug=True)
