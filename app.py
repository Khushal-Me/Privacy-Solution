import torch
from ultralytics import YOLO
import cv2
import os
import numpy as np

MODEL_PATH = "yolov8n.pt"  # or your custom model
model = YOLO(MODEL_PATH)

IGNORE_ABOVE_LINE = 200  # y-coord above which to ignore objects
INTENSITY_THRESHOLD = 0.3

# Map YOLO's default class IDs to your desired labeling scheme
# YOLOv8’s default classes are:
#   0=person, 1=bicycle, 2=car, 3=motorcycle, 4=airplane, 5=bus, ...
# Modify to suit your actual dataset and desired IDs.
def map_class_id(yolo_class_id):
    if yolo_class_id == 0:
        return 1  # Person
    elif yolo_class_id == 1:
        return 2  # Bicycle
    elif yolo_class_id == 2 or yolo_class_id == 3 or yolo_class_id == 5:
        return 3  # Car or other vehicle
    else:
        return 0  # Unknown/unmapped

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
    _, _, _, maxLoc = cv2.minMaxLoc(roi)
    max_x, max_y = maxLoc

    box_width = right - left
    box_height = bottom - top
    aspect_ratio = box_height / float(box_width + 1e-6)

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
        # Likely a car (heuristic)
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
        # box format: [x_min, y_min, x_max, y_max, yolo_conf, yolo_class]
        if len(box) != 6:
            continue

        x_min, y_min, x_max, y_max, yolo_conf, yolo_class_id = box

        # IGNORE if bounding box is entirely above our ignore line
        if y_max < IGNORE_ABOVE_LINE:
            continue

        # Clip & check ROI for max thermal intensity
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

        # Use max thermal intensity as the new "confidence"
        max_intensity = roi.max()
        thermal_conf = max_intensity / 255.0

        # Keep only if above your chosen threshold
        if thermal_conf >= intensity_threshold:
            # Convert YOLO class to your own ID system
            mapped_id = map_class_id(int(yolo_class_id))

            # Optionally: Instead of YOLO's class, you could call identify_object_by_heat
            # obj_type = identify_object_by_heat(gray_image, (x_min, y_min, x_max, y_max))

            # Save detection
            valid_detections.append((
                int(x_min),
                int(y_min),
                int(x_max),
                int(y_max),
                float(thermal_conf),
                mapped_id
            ))

    return valid_detections

def process_local_images(directory):
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    output_dir = os.path.join(directory, "annotated_results")
    os.makedirs(output_dir, exist_ok=True)

    # You can store your results here:
    txt_output_path = os.path.join(output_dir, "results.txt")

    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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
            gray_eq = clahe.apply(gray)  # Enhanced contrast

            # Convert back to BGR for YOLO detection
            image_preprocessed = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

            detections = detect_and_analyze(
                model=model,
                bgr_image=image_preprocessed,
                gray_image=gray_eq,
                intensity_threshold=INTENSITY_THRESHOLD
            )

            # For each valid detection, append to the text file
            if detections:
                with open(txt_output_path, 'a') as f:
                    for (x_min, y_min, x_max, y_max, conf, class_id) in detections:
                        # Write a single line in the desired format
                        # e.g. "thermal_001.png 1 0.95 50 200 60 250"
                        line = f"{filename} {class_id} {conf:.2f} {x_min} {y_min} {x_max} {y_max}\n"
                        f.write(line)

                        # Also draw bounding boxes for visualization
                        cv2.rectangle(
                            image_preprocessed,
                            (x_min, y_min),
                            (x_max, y_max),
                            (0, 255, 0), 2
                        )
                        cv2.putText(
                            image_preprocessed,
                            f"ID={class_id}, conf={conf:.2f}",
                            (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1
                        )
            else:
                print("No objects detected.")

            # Show the detections on screen
            cv2.imshow("Detections", image_preprocessed)
            cv2.waitKey(0)

            # Save annotated image
            out_path = os.path.join(output_dir, f"annotated_{filename}")
            cv2.imwrite(out_path, image_preprocessed)
            print(f"Annotated image saved to: {out_path}")

    cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    image_directory = "Privacy-Solution/test_thermal_data/test_images_8_bit"
    process_local_images(image_directory)
