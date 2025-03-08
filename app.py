import torch
from ultralytics import YOLO
import cv2
import os
import numpy as np
from PIL import Image

MODEL_PATH = "yolov8n.pt"  # or your custom model
model = YOLO(MODEL_PATH)

IGNORE_ABOVE_LINE = 200  # y-coord above which to ignore objects
INTENSITY_THRESHOLD = 0.3

# Function to convert TIFF to JPEG
def convert_tiff_to_jpeg(tiff_path, output_dir):
    try:
        img = Image.open(tiff_path)
        jpeg_path = os.path.join(output_dir, os.path.splitext(os.path.basename(tiff_path))[0] + ".jpg")
        img.convert("RGB").save(jpeg_path, "JPEG")
        return jpeg_path  # Return new JPEG path
    except Exception as e:
        print(f"Error converting {tiff_path} to JPEG: {e}")
        return None

# Function to check if an image is 8-bit
def is_8bit_image(image_path):
    try:
        img = Image.open(image_path)
        return img.mode in ["L", "P", "RGB"]  # L = 8-bit grayscale, P = 8-bit palette, RGB = standard color
    except Exception as e:
        print(f"Error checking image bit depth: {e}")
        return False

# Map YOLO's default class IDs to your desired labeling scheme
def map_class_id(yolo_class_id):
    if yolo_class_id == 0:
        return 1  # Person
    elif yolo_class_id == 1:
        return 2  # Bicycle
    elif yolo_class_id in [2, 3, 5]:
        return 3  # Car or other vehicle
    else:
        return 0  # Unknown/unmapped

def detect_and_analyze(model, bgr_image, gray_image, intensity_threshold=0.3):
    results = model.predict(bgr_image)
    boxes = results[0].boxes

    detections = boxes.data.cpu().numpy() if len(boxes) > 0 else []
    valid_detections = []

    for box in detections:
        if len(box) != 6:
            continue

        x_min, y_min, x_max, y_max, yolo_conf, yolo_class_id = box

        if y_max < IGNORE_ABOVE_LINE:
            continue

        left, top, right, bottom = map(int, [x_min, y_min, x_max, y_max])
        h, w = gray_image.shape
        left = max(0, left)
        right = min(w, right)
        top = max(0, top)
        bottom = min(h, bottom)
        if right <= left or bottom <= top:
            continue

        roi = gray_image[top:bottom, left:right]
        if roi.size == 0:
            continue

        max_intensity = roi.max()
        thermal_conf = max_intensity / 255.0

        if thermal_conf >= intensity_threshold:
            mapped_id = map_class_id(int(yolo_class_id))
            valid_detections.append((int(x_min), int(y_min), int(x_max), int(y_max), float(thermal_conf), mapped_id))

    return valid_detections

def process_local_images(directory):
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    output_dir = os.path.join(directory, "annotated_results")
    os.makedirs(output_dir, exist_ok=True)

    txt_output_path = os.path.join(output_dir, "results.txt")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Convert TIFF to JPEG if needed
        if filename.lower().endswith('.tiff') or filename.lower().endswith('.tif'):
            print(f"Converting {filename} from TIFF to JPEG...")
            converted_path = convert_tiff_to_jpeg(file_path, directory)
            if not converted_path:
                continue
            file_path = converted_path  # Use new JPEG file
            filename = os.path.basename(converted_path)

        # Check if the image is 8-bit before processing
        if not is_8bit_image(file_path):
            print(f"Skipping {filename}: Not an 8-bit image.")
            continue

        # Process only 8-bit images
        print(f"\nProcessing {filename}...")

        image_bgr = cv2.imread(file_path)
        if image_bgr is None:
            print(f"Failed to load {filename}. Skipping.")
            continue

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray_eq = clahe.apply(gray)  # Enhanced contrast

        image_preprocessed = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

        detections = detect_and_analyze(model, image_preprocessed, gray_eq, INTENSITY_THRESHOLD)

        if detections:
            with open(txt_output_path, 'a') as f:
                for (x_min, y_min, x_max, y_max, conf, class_id) in detections:
                    line = f"{filename} {class_id} {conf:.2f} {x_min} {y_min} {x_max} {y_max}\n"
                    f.write(line)

                    cv2.rectangle(image_preprocessed, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(image_preprocessed, f"ID={class_id}, conf={conf:.2f}", (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        else:
            print("No objects detected.")

        out_path = os.path.join(output_dir, f"annotated_{filename}")
        cv2.imwrite(out_path, image_preprocessed)
        print(f"Annotated image saved to: {out_path}")

    cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    image_directory = r"C:\Users\aditb\OneDrive\Desktop\Western AIC Competition\Privacy-Solution\test_thermal_data\test_images_8_bit"
    process_local_images(image_directory)
