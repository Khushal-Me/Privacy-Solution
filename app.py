import torch
from ultralytics import YOLO
import cv2
import os
import numpy as np

MODEL_PATH = "yolov8n.pt"  # Replace with your custom YOLO if needed
model = YOLO(MODEL_PATH)

# Set this to the pixel row above which you want to ignore objects
IGNORE_ABOVE_LINE = 200  # Example: y=200

def identify_object_by_heat(gray_image, box):
    x_min, y_min, x_max, y_max = map(int, box)

    # Clip ROI
    h, w = gray_image.shape
    left = max(0, x_min)
    right = min(w, x_max)
    top = max(0, y_min)
    bottom = min(h, y_max)

    if right <= left or bottom <= top:
        return "Unknown"

    roi = gray_image[top:bottom, left:right]
    _, _, _, maxLoc = cv2.minMaxLoc(roi)  # (minVal, maxVal, minLoc, maxLoc)
    max_x, max_y = maxLoc

    box_width = right - left
    box_height = bottom - top
    aspect_ratio = box_height / float(box_width + 1e-6)  # avoid /0

    if aspect_ratio > 1.2:
        # Likely a person
        region_top = box_height / 3
        region_middle = 2 * box_height / 3
        if max_y < region_top:
            return "Human (hot head)"
        elif max_y < region_middle:
            return "Human (hot torso)"
        else:
            return "Human (lower body warm)"
    else:
        # Likely a car
        region_top = box_height / 3
        region_bottom = 2 * box_height / 3
        if max_y < region_top:
            return "Car (engine near top?)"
        elif max_y > region_bottom:
            return "Car (exhaust/rotors near bottom?)"
        else:
            return "Car (engine area)"

def detect_and_analyze(model, bgr_image, gray_image, intensity_threshold=0.3):
    results = model.predict(bgr_image)
    boxes = results[0].boxes

    detections = boxes.data.cpu().numpy() if len(boxes) > 0 else []
    valid_detections = []

    for box in detections:
        # box = [x_min, y_min, x_max, y_max, yolo_conf, class_id]
        if len(box) != 6:
            continue

        x_min, y_min, x_max, y_max, _, _ = box

        # ---------- IGNORE if bounding box is entirely above our ignore line -----------
        if y_max < IGNORE_ABOVE_LINE:
            # This bounding box is above the line; skip it
            continue

        # Clip & check ROI
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

        # Use max thermal intensity as the confidence
        max_intensity = roi.max()
        new_conf = max_intensity / 255.0

        # Keep only if above intensity_threshold
        if new_conf >= intensity_threshold:
            object_type = identify_object_by_heat(gray_image, (x_min, y_min, x_max, y_max))
            valid_detections.append((x_min, y_min, x_max, y_max, object_type))

    return valid_detections

def process_local_images(directory):
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    output_dir = os.path.join(directory, "annotated_results")
    os.makedirs(output_dir, exist_ok=True)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            print(f"\nProcessing {filename}...")

            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                print(f"Failed to load {filename}. Skipping.")
                continue

            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            gray_eq = clahe.apply(gray)
            image_preprocessed = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

            detections = detect_and_analyze(
                model=model,
                bgr_image=image_preprocessed,
                gray_image=gray_eq,
                intensity_threshold=0.3
            )

            if not detections:
                print("No objects detected.")
            else:
                for (x_min, y_min, x_max, y_max, obj_type) in detections:
                    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                    # Draw bounding box
                    cv2.rectangle(
                        image_preprocessed,
                        (x_min, y_min),
                        (x_max, y_max),
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        image_preprocessed,
                        obj_type,
                        (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )

            cv2.imshow("Detections", image_preprocessed)
            cv2.waitKey(0)

            out_path = os.path.join(output_dir, f"annotated_{filename}")
            cv2.imwrite(out_path, image_preprocessed)
            print(f"Annotated image saved to: {out_path}")

    cv2.destroyAllWindows()

# Update to your thermal image directory
image_directory = "Privacy-Solution/test_thermal_data/test_images_8_bit"
process_local_images(image_directory)
