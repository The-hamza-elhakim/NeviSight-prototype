from flask import Flask, request, render_template, redirect, url_for, send_from_directory
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

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_dicom(file_path):
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
    except Exception:
        return False

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

    ext = filename.rsplit('.', 1)[1].lower()

    if ext == 'dcm' or is_dicom(file_path):
        # Just validate it’s a real DICOM — you could parse metadata later
        print("Uploaded file is a DICOM image")
        return f"Successfully uploaded DICOM file: {filename}"
    else:
        # Handle as regular image (png/jpg/jpeg)
        try:
            img = Image.open(file_path)
            img.verify()  # Check if valid image
            print("Uploaded file is a standard image")
            return f"Successfully uploaded image file: {filename}"
        except Exception as e:
            return f"Error opening image: {e}", 400

if __name__ == "__main__":
    app.run(debug=True)
