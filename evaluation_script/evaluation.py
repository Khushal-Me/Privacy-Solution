import numpy as np
import argparse
import os

def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def average_precision(recalls, precisions):
    """Compute AP by integrating the precision-recall curve."""
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap


def mean_average_precision(predictions, ground_truths, iou_threshold=0.5):
    """Compute mean Average Precision (mAP) at a given IoU threshold."""
    aps = []
    classes = set([gt[1] for gt in ground_truths])

    for cls in classes:
        preds = [p for p in predictions if p[1] == cls]
        gts = [g for g in ground_truths if g[1] == cls]

        preds.sort(key=lambda x: x[2], reverse=True)  # Sort by confidence

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        gt_used = {}

        for i, pred in enumerate(preds):
            img_gts = [g for g in gts if g[0] == pred[0]]
            best_iou = 0
            best_gt_idx = -1

            for j, gt in enumerate(img_gts):
                iou = compute_iou(pred[3:], gt[2:])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold and (pred[0], best_gt_idx) not in gt_used:
                tp[i] = 1
                gt_used[(pred[0], best_gt_idx)] = True
            else:
                fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / len(gts) if len(gts) > 0 else np.zeros(len(tp))
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap = average_precision(recalls, precisions)
        aps.append(ap)

    return np.mean(aps) if aps else 0


def read_detections(file_path):
    """Read detections from file."""
    detections = []
    file_path = os.path.join(file_path, 'results.txt')
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            image_id = parts[0]
            class_id = parts[1]
            confidence = float(parts[2])
            bbox = list(map(float, parts[3:]))
            detections.append([image_id, class_id, confidence] + bbox)
    return detections


def read_ground_truths(file_path):
    """Read ground truths from file."""
    ground_truths = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            image_id = parts[0]
            class_id = parts[1]
            bbox = list(map(float, parts[2:]))
            ground_truths.append([image_id, class_id] + bbox)
    return ground_truths


def main():
    parser = argparse.ArgumentParser(description="Compute mAP@0.5 for object detection results.")
    parser.add_argument("detection_path", type=str, help="Path to detections file.")
    parser.add_argument("label_path", type=str, help="Path to ground truths file.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold (default: 0.5).")
    args = parser.parse_args()

    detections = read_detections(args.detection_path)
    ground_truths = read_ground_truths(args.label_path)

    map_score = mean_average_precision(detections, ground_truths, iou_threshold=args.iou)
    print(f"mAP@{args.iou}: {map_score:.4f}")


if __name__ == "__main__":
    main()
