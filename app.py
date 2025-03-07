import torch
from ultralytics import YOLO
import cv2
import os

# Load the YOLOv8 model (replace with your model path if different)
MODEL_PATH = "yolov8n.pt"  # Pre-trained model, or use your custom-trained model
model = YOLO(MODEL_PATH)

def detect_objects(model, image_path, conf_threshold=0.3):
    """
    Runs object detection on an image using the YOLO model.

    Parameters:
        model: The loaded YOLOv8 model.
        image_path (str): Path to the image file.
        conf_threshold (float): Minimum confidence score for detections.

    Returns:
        List of detections: [(x_min, y_min, x_max, y_max, confidence, class_id)].
    """
    # Run inference on the image
    results = model(image_path)
    boxes = results[0].boxes
    detections = boxes.data.cpu().numpy()

    valid_detections = []
    for box in detections:
        if len(box) != 6:  # Ensure box has expected format
            print(f"Skipping invalid box: {box}")
            continue
        
        x_min, y_min, x_max, y_max, conf, class_id = box
        if conf >= conf_threshold:  # Filter by confidence
            valid_detections.append((x_min, y_min, x_max, y_max, conf, int(class_id)))
            print(f"Detected: Class {int(class_id)}, Conf {conf:.2f}, "
                  f"Box ({x_min}, {y_min}, {x_max}, {y_max})")
    
    return valid_detections

def process_local_images(directory):
    """
    Process all images in a local directory.

    Parameters:
        directory (str): Path to the folder containing thermal images.
    """
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    # Loop through files in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter for image files
            image_path = os.path.join(directory, filename)
            print(f"Processing {filename}...")
            
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load {filename}. Skipping.")
                continue
            
            # Run object detection
            detections = detect_objects(model, image_path)
            if detections:
                for det in detections:
                    x_min, y_min, x_max, y_max, conf, class_id = det
                    print(f"Class ID {class_id}: Conf {conf:.2f}, "
                          f"Box ({x_min}, {y_min}, {x_max}, {y_max})")
            else:
                print(f"No objects detected in {filename}.")

# Specify your local directory here
image_directory = "test_thermal_data/test_images_8_bit/"  # Update this to match your folder path

# Process the images
process_local_images(image_directory)