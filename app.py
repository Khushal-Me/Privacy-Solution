import torch
from ultralytics import YOLO
import cv2
import os

# Load the trained YOLO model
MODEL_PATH = r"C:\Users\aditb\OneDrive\Desktop\Western AIC Competition\runs\detect\train11\weights\best.pt"
model = YOLO(MODEL_PATH)

# Specify directories (Updated Paths)
image_directory = r"C:\Users\aditb\OneDrive\Desktop\Western AIC Competition\yolo_dataset\test_images"
output_directory = r"C:\Users\aditb\OneDrive\Desktop\Western AIC Competition\yolo_dataset\runs\detect"

def detect_objects(model, image_path, conf_threshold=0.3):
    results = model(image_path)
    boxes = results[0].boxes
    detections = boxes.data.cpu().numpy()

    valid_detections = []
    for box in detections:
        if len(box) != 6:
            print(f"Skipping invalid box: {box}")
            continue
        
        x_min, y_min, x_max, y_max, conf, class_id = box
        if conf >= conf_threshold:
            valid_detections.append((x_min, y_min, x_max, y_max, conf, int(class_id)))
            print(f"Detected: Class {int(class_id)}, Conf {conf:.2f}, "
                  f"Box ({x_min}, {y_min}, {x_max}, {y_max})")
    
    return valid_detections

def draw_detections(image_path, detections, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}. Skipping.")
        return

    for (x_min, y_min, x_max, y_max, conf, class_id) in detections:
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        label = f"Class {class_id}: {conf:.2f}"
        cv2.putText(image, label, (int(x_min), int(y_min) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(output_path, image)
    print(f"✅ Detection results saved to {output_path}")

def process_local_images(directory, output_dir):
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            output_path = os.path.join(output_dir, filename)

            print(f"Processing {filename}...")

            detections = detect_objects(model, image_path)
            if detections:
                draw_detections(image_path, detections, output_path)
            else:
                print(f"No objects detected in {filename}.")

# Run detection on test images
process_local_images(image_directory, output_directory)