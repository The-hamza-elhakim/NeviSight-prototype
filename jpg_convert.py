import os
import pydicom
import numpy as np
from PIL import Image

def convert_dicom_to_jpg_only(input_dcm_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dcm_dir):
        if file.lower().endswith('.dcm'):
            dicom_path = os.path.join(input_dcm_dir, file)
            try:
                dcm = pydicom.dcmread(dicom_path)
                arr = dcm.pixel_array

                # Normalize to 0–255 and convert to uint8
                arr_norm = ((arr - arr.min()) / (np.ptp(arr) + 1e-6) * 255).astype(np.uint8)

                img = Image.fromarray(arr_norm).convert("RGB")
                out_name = f"dcm_{file.replace('.dcm', '.jpg')}"
                img.save(os.path.join(output_dir, out_name))
                print(f"✔ Converted: {file} → {out_name}")
            except Exception as e:
                print(f"⚠️ Skipped {file} due to error: {e}")


# --- Example usage ---
convert_dicom_to_jpg_only(
    input_dcm_dir=r"C:\Users\hamza\Documents\Projects\Research Projects\my data\Original dataset\DICOM",
    output_dir=r"C:\Users\hamza\Documents\Projects\Research Projects\my data\Proccesed dataset\vaild"
)
