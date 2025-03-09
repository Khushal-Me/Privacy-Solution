# Privacy Solution 🔒

An intelligent privacy-preserving solution for thermal imaging detection using YOLOv8 and thermal intensity analysis.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)

## 🔥 Overview

Privacy Solution is a specialized detection system that combines computer vision and thermal imaging to identify objects while preserving privacy. The system uses YOLOv8 for object detection and analyzes thermal intensity in grayscale images to validate detections while minimizing privacy concerns in surveillance applications.

### Key Features

- 🚶‍♂️ Privacy-focused object detection for people, vehicles, and bicycles
- 🌡️ Thermal intensity validation to reduce false positives
- 🔄 TIFF to JPEG conversion support for thermal imagery
- 🛑 Spatial filtering to ignore regions above a specified y-coordinate
- 📊 Customizable detection confidence thresholds
- 🖼️ Automated annotation of detected objects

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Khushal-Me/Privacy-Solution.git
cd Privacy-Solution

# Install dependencies
pip install torch ultralytics opencv-python numpy pillow
```

## 📋 Requirements

- Python 3.8+
- Ultralytics YOLOv8
- OpenCV
- NumPy
- PIL (Pillow)

## 🚀 Usage

### Basic Usage:

```bash
python app.py --image_folder test_thermal_data/test_images_8_bit --results_folder test_thermal_data/test_images_8_bit/annotated_results
```

### Options:

- `--image_folder`: Directory containing thermal images (default: "test_thermal_data/test_images_8_bit")
- `--results_folder`: Directory to save annotated images and detection results (default: "test_thermal_data/test_images_8_bit/annotated_results")

### Evaluation:

```bash
cd evaluation_script
./run_exec.sh
```

## 🧠 How It Works

1. **Image Preprocessing**:
   - Converts TIFF images to JPEG if needed
   - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhanced contrast
   - Checks for 8-bit image compatibility

2. **Detection Pipeline**:
   - YOLOv8 identifies potential objects in the image
   - Thermal intensity analysis confirms valid detections
   - Class mapping converts YOLOv8 classes to custom ID scheme:
     - Person: ID 1
     - Bicycle: ID 2
     - Vehicles (car, truck, bus): ID 3

3. **Privacy Features**:
   - Ignores detections above a specified y-coordinate (customizable)
   - Uses thermal intensity threshold to validate true detections
   - Works with grayscale thermal imagery instead of RGB video

4. **Output**:
   - Annotated images with bounding boxes and class information
   - Text file with detection results in format: `filename class_id confidence x_min y_min x_max y_max`

## 📂 Project Structure

```
.
├── LICENSE                     # MIT License
├── app.py                      # Main application script
├── evaluation_script/          # Scripts for evaluating model performance
│   ├── evaluation.py           # Evaluation metrics calculation
│   └── run_exec.sh             # Shell script to run evaluations
├── runs/                       # Training runs and model weights
├── test_thermal_data/          # Test data for thermal imaging
│   ├── test_images_16_bit/     # 16-bit TIFF thermal images
│   └── test_images_8_bit/      # 8-bit JPEG thermal images
└── yolov8n.pt                  # Pre-trained YOLOv8 nano model
```

## ⚙️ Configuration

Key parameters that can be modified in `app.py`:

```python
MODEL_PATH = "yolov8n.pt"      # Path to YOLOv8 model
IGNORE_ABOVE_LINE = 200        # Y-coordinate threshold to ignore detections
INTENSITY_THRESHOLD = 0.3      # Minimum thermal intensity for valid detections
```

## 🔬 Training Custom Models

Custom YOLOv8 models can be trained on thermal data and used by updating the `MODEL_PATH` variable:

```python
MODEL_PATH = "runs/detect/train7/weights/best.pt"  # Path to your custom trained model
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Contact

For questions or collaboration, please reach out to:
Khushal @ khushaldemehta@gmail.com
