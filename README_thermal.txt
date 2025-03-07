# Thermal Dataset

## Overview
This dataset consists of thermal images categorized into two main types based on bit-depth:
1. **8-bit Dataset**
2. **16-bit Dataset**

Each dataset contains images and corresponding label files organized into training and validation sets.

## Dataset Structure
The dataset is structured as follows:

```
Thermal_Dataset/
│── 8_bit_dataset/
│   │── train_images_8_bit/
│   │── val_images_8_bit/
│   │── train_labels_8_bit.txt
│   │── val_labels_8_bit.txt
│
│── 16-bit/
│   │── train_images_16_bit/
│   │── val_images_16_bit/
│   │── train_labels_16_bit.txt
│   │── val_labels_16_bit.txt
```

### Description of Files and Folders
- `8-bit/` and `16-bit/`: Contain respective datasets categorized by bit-depth.
- `train_images/`: Folder containing training images.
- `val_images/`: Folder containing validation images.
- `train_labels.txt`: A single text file containing labels for all training images.
- `val_labels.txt`: A single text file containing labels for all validation images.

## Label File Format
Each label file (`train_labels.txt` and `val_labels.txt`) contains annotations in the following format:

```
image_file class_id x_min x_max y_min y_max
```

### Explanation of Label Fields:
- **image_file**: The name of the image file (e.g., `thermal_001.png`).
- **class_id**: The class label of the object in the image.
- **x_min**: The minimum x-coordinate (left boundary) of the bounding box.
- **x_max**: The maximum x-coordinate (right boundary) of the bounding box.
- **y_min**: The minimum y-coordinate (top boundary) of the bounding box.
- **y_max**: The maximum y-coordinate (bottom boundary) of the bounding box.

### available classes/labels:
person (class_id = 1)
bicycle/bike (class_id = 2)
vehicle/car (class_id = 3)

Each line represents one object annotation within an image. If an image has multiple objects, it will have multiple corresponding lines in the label file.

## Usage
This dataset can be used for object detection tasks, including training deep learning models for thermal image analysis. Users should ensure they load the correct bit-depth dataset based on their requirements.

