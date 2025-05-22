# src/utils.py
import numpy as np
import cv2

def non_max_suppression(boxes, scores, iou_threshold):
    """
    Applies Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.

    Args:
        boxes (numpy.array): Array of bounding boxes, shape (N, 4), format [x1, y1, x2, y2].
        scores (numpy.array): Array of confidence scores for each box, shape (N,).
        iou_threshold (float): IoU threshold for suppressing boxes.

    Returns:
        list: Indices of the boxes to keep.
    """
    if len(boxes) == 0:
        return []

    # Convert boxes to (x1, y1, x2, y2) if they are (x, y, w, h)
    # Assuming input `boxes` are already [x1, y1, x2, y2] for this implementation.
    # If they are [x_center, y_center, w, h] or [x1, y1, w, h], convert them first.

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Calculate the area of the bounding boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1) # +1 to include the boundary pixel

    # Sort the bounding boxes by their confidence scores in descending order
    idxs = np.argsort(scores)[::-1]

    pick = [] # List to store the indices of the picked boxes

    while len(idxs) > 0:
        # Grab the last index in the idxs list (which is the box with the highest score)
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of the bounding box
        # and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the overlapping region
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the Intersection over Union (IoU)
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have an IoU greater
        # than the provided threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > iou_threshold)[0])))

    return pick

def draw_bounding_boxes(image, boxes, scores, class_labels=None, color=(0, 255, 0), thickness=2):
    """
    Draws bounding boxes and (optionally) scores/labels on an image.

    Args:
        image (numpy.array): The image to draw on (OpenCV BGR format).
        boxes (list or numpy.array): List of boxes, each [x1, y1, x2, y2].
        scores (list or numpy.array): Confidence scores for each box.
        class_labels (list or numpy.array, optional): Class labels for each box.
        color (tuple, optional): BGR color for the boxes.
        thickness (int, optional): Thickness of the box lines.
    """
    img_copy = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box) # Ensure coordinates are integers for drawing
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)

        text = f"{scores[i]:.2f}"
        if class_labels is not None and i < len(class_labels):
            text = f"{class_labels[i]}: {text}"
        
        # Position text above the bounding box
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 15 
        cv2.putText(img_copy, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness // 2 + 1)
    return img_copy