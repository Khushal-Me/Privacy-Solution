import torch
from ultralytics import YOLO
import cv2
import os

# Replace with your custom thermal or YOLO model path
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)

def detect_objects(model, image, conf_threshold=0.3):
    """
    Runs object detection on a preprocessed image using the YOLO model.

    Parameters:
        model: The loaded YOLOv8 model.
        image (numpy.array): Preprocessed image (BGR).
        conf_threshold (float): Minimum confidence score for detections.

    Returns:
        List of detections: [(x_min, y_min, x_max, y_max, confidence, class_id)].
    """
    results = model.predict(image)
    boxes = results[0].boxes

    # Convert detections to NumPy
    detections = boxes.data.cpu().numpy() if len(boxes) > 0 else []

    valid_detections = []
    for box in detections:
        if len(box) != 6:
            continue

        x_min, y_min, x_max, y_max, conf, class_id = box
        if conf >= conf_threshold:
            valid_detections.append((x_min, y_min, x_max, y_max, conf, int(class_id)))

    return valid_detections


def process_local_images(directory):
    """
    Process all images in a local directory, applying CLAHE to improve
    contrast in thermal images before detection, then draw bounding
    boxes for visualization.
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    # Setup CLAHE parameters
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Create an output folder to save annotated images
    output_dir = os.path.join(directory, "annotated_results")
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            print(f"\nProcessing {filename}...")

            # Load the image (in color)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load {filename}. Skipping.")
                continue

            # Convert to grayscale if dealing with thermal
            # (some cameras store 3-channel 'false color'; adjust as needed)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply CLAHE to enhance contrast
            gray_eq = clahe.apply(gray)

            # Convert back to BGR for YOLO if needed
            image_preprocessed = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

            # Detect objects
            detections = detect_objects(model, image_preprocessed, conf_threshold=0.3)

            # Draw bounding boxes on image_preprocessed
            for (x_min, y_min, x_max, y_max, conf, class_id) in detections:
                # Convert floats to ints for drawing
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

                # Draw rectangle
                cv2.rectangle(
                    image_preprocessed,
                    (x_min, y_min),
                    (x_max, y_max),
                    (0, 255, 0),  # (B, G, R) color
                    thickness=2
                )

                # Create label text: "class_id: conf"
                label = f"{class_id}: {conf:.2f}"
                cv2.putText(
                    image_preprocessed,
                    label,
                    (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    thickness=1
                )

            # If you want to show the image (blocking window)
            # Remove these lines if running headless
            cv2.imshow("Detections", image_preprocessed)
            cv2.waitKey(0)  # Press any key to close the window

            # Save the image with drawn boxes
            out_path = os.path.join(output_dir, f"annotated_{filename}")
            cv2.imwrite(out_path, image_preprocessed)
            print(f"Annotated image saved to: {out_path}")

    # Close any OpenCV windows if they remain open
    cv2.destroyAllWindows()


# Update this to your thermal image folder path
image_directory = "Privacy-Solution/test_thermal_data/test_images_8_bit"
process_local_images(image_directory)
