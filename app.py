import torch
from ultralytics import YOLO
import cv2
import os
import numpy as np
from PIL import Image  # For TIFF conversion

MODEL_PATH = "yolov8n.pt"  # or your custom model
model = YOLO(MODEL_PATH)

IGNORE_ABOVE_LINE = 200  # y-coord above which to ignore objects
INTENSITY_THRESHOLD = 0.3

def convert_tiff_to_jpeg(tiff_path, output_dir):
    """Convert a TIFF image to JPEG format."""
    image = Image.open(tiff_path)
    filename = os.path.splitext(os.path.basename(tiff_path))[0] + ".jpg"
    jpeg_path = os.path.join(output_dir, filename)
    image.convert("RGB").save(jpeg_path, "JPEG")
    return jpeg_path

def process_local_images(directory):
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    output_dir = os.path.join(directory, "annotated_results")
    os.makedirs(output_dir, exist_ok=True)

    # Convert TIFF images before processing
    for filename in os.listdir(directory):
        if filename.lower().endswith('.tiff'):
            tiff_path = os.path.join(directory, filename)
            jpeg_path = convert_tiff_to_jpeg(tiff_path, directory)
            print(f"Converted {filename} to {jpeg_path}")

    # Process images (including converted TIFFs)
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            print(f"\nProcessing {filename}...")

            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                print(f"Failed to load {filename}. Skipping.")
                continue

            # Convert to grayscale for thermal processing
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            
            # Further processing as in your original code...
            # (detect_and_analyze, annotations, etc.)

if __name__ == "__main__":
    image_directory = "Privacy-Solution/test_thermal_data/test_images_16_bit"
    process_local_images(image_directory)
