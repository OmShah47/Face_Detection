# src/predict.py
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from . import config # To get TARGET_SIZE
from . import data_loader # For preprocess_image if needed (though we might do it manually)
from . import utils # For NMS and drawing

# --- Inference Parameters ---
MODEL_PATH = None # Will be set below, or you can hardcode path to your .keras model
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to consider a detection as a face
NMS_IOU_THRESHOLD = 0.3     # IoU threshold for Non-Max Suppression

def find_latest_model(models_dir):
    """Finds the latest .keras model file in a directory based on filename timestamp or 'best'."""
    keras_files = [f for f in os.listdir(models_dir) if f.endswith(".keras")]
    if not keras_files:
        return None

    # Prioritize 'best' model if it exists
    best_models = [f for f in keras_files if "best" in f.lower()]
    if best_models:
        # If multiple 'best' models, pick the most recent one by modification time
        best_models_paths = [os.path.join(models_dir, f) for f in best_models]
        latest_best_model = max(best_models_paths, key=os.path.getmtime)
        print(f"Found 'best' model: {latest_best_model}")
        return latest_best_model

    # If no 'best' model, pick the most recent 'final' model by timestamp in filename
    # This assumes a filename format like '..._YYYYMMDD-HHMMSS_...'
    # Or simply the most recently modified file
    model_paths = [os.path.join(models_dir, f) for f in keras_files]
    latest_model = max(model_paths, key=os.path.getmtime)
    print(f"Found latest model by modification time: {latest_model}")
    return latest_model


def preprocess_patch(patch, target_size):
    """Preprocesses a single image patch for model input."""
    resized_patch = cv2.resize(patch, (target_size[1], target_size[0])) # cv2 uses (w,h)
    normalized_patch = resized_patch.astype(np.float32) / 255.0
    # Expand dimensions to create a batch of 1
    batched_patch = np.expand_dims(normalized_patch, axis=0)
    return batched_patch

def predict_on_image(image_path, model, target_size, confidence_thresh, nms_iou_thresh):
    """
    Performs face detection on a single image.
    This version uses a simple multi-scale approach without exhaustive sliding windows.
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    img_h, img_w = original_image.shape[:2]
    
    detected_boxes = []
    detected_scores = []

    # --- Multi-Scale Patch Generation (Simplified) ---
    # We'll test the image at a few different scales.
    # A more robust system uses sliding windows across these scales.
    scales = [1.0, 0.75, 0.5] # Test at original size, 75%, 50%
    # For each scale, we'll divide the image into overlapping patches
    # matching the model's TARGET_SIZE aspect ratio.

    for scale in scales:
        scaled_h, scaled_w = int(img_h * scale), int(img_w * scale)
        if scaled_h < target_size[0] or scaled_w < target_size[1]:
            continue # Skip if scaled image is smaller than target patch size

        scaled_image = cv2.resize(original_image, (scaled_w, scaled_h))
        
        # Stride for sliding window (can be target_size[0]//2 for 50% overlap)
        stride_y = target_size[0] // 2
        stride_x = target_size[1] // 2

        for y_start in range(0, scaled_h - target_size[0] + 1, stride_y):
            for x_start in range(0, scaled_w - target_size[1] + 1, stride_x):
                patch = scaled_image[y_start : y_start + target_size[0], 
                                     x_start : x_start + target_size[1]]
                
                if patch.shape[0] != target_size[0] or patch.shape[1] != target_size[1]:
                    continue # Should not happen with correct loop bounds

                # Preprocess patch and predict
                processed_patch = preprocess_patch(patch, target_size)
                class_pred, bbox_pred = model.predict(processed_patch, verbose=0) # verbose=0 for less output

                confidence = class_pred[0][0]

                if confidence > confidence_thresh:
                    # Bbox_pred is [norm_x, norm_y, norm_w, norm_h] relative to the patch
                    norm_bx, norm_by, norm_bw, norm_bh = bbox_pred[0]

                    # Convert normalized patch coordinates to coordinates in the *scaled_image*
                    box_x_patch = norm_bx * target_size[1]
                    box_y_patch = norm_by * target_size[0]
                    box_w_patch = norm_bw * target_size[1]
                    box_h_patch = norm_bh * target_size[0]

                    # Convert patch bbox to coordinates in the *original image*
                    # x1_orig = (x_start_in_scaled_image + box_x_in_patch) / scale_of_scaled_image
                    x1_orig = (x_start + box_x_patch) / scale
                    y1_orig = (y_start + box_y_patch) / scale
                    w_orig = box_w_patch / scale
                    h_orig = box_h_patch / scale
                    
                    x2_orig = x1_orig + w_orig
                    y2_orig = y1_orig + h_orig
                    
                    detected_boxes.append([x1_orig, y1_orig, x2_orig, y2_orig])
                    detected_scores.append(confidence)

    if not detected_boxes:
        print("No faces detected.")
        cv2.imshow("Detections", original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    detected_boxes = np.array(detected_boxes)
    detected_scores = np.array(detected_scores)

    # Apply Non-Max Suppression
    print(f"Applying NMS to {len(detected_boxes)} raw detections...")
    keep_indices = utils.non_max_suppression(detected_boxes, detected_scores, nms_iou_thresh)
    
    final_boxes = detected_boxes[keep_indices]
    final_scores = detected_scores[keep_indices]
    print(f"Kept {len(final_boxes)} detections after NMS.")

    # Draw final bounding boxes
    output_image = utils.draw_bounding_boxes(original_image, final_boxes, final_scores)

    cv2.imshow("Detections", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally save the output image
    # output_filename = os.path.join("predictions", os.path.basename(image_path).replace(".", "_pred."))
    # cv2.imwrite(output_filename, output_image)
    # print(f"Saved prediction to {output_filename}")


if __name__ == '__main__':
    # --- Setup ---
    # Try to find the latest trained model automatically
    MODEL_PATH = find_latest_model(config.MODELS_TRAINED_DIR)
    
    if MODEL_PATH is None:
        print(f"Error: No model found in {config.MODELS_TRAINED_DIR}. Please train a model first or set MODEL_PATH manually.")
        exit()
    
    print(f"Loading model: {MODEL_PATH}")
    # When loading a model with custom objects or custom metrics not built into Keras by default,
    # you might need to provide a `custom_objects` dictionary to `load_model`.
    # Our current model uses standard layers and losses, so it should be fine.
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("If you have custom metrics like Precision/Recall with specific names, you might need to pass them in custom_objects.")
        print("Example: model = load_model(MODEL_PATH, custom_objects={'precision': tf.keras.metrics.Precision, 'recall': tf.keras.metrics.Recall})")
        exit()
        
    model.summary() # Print model summary to confirm it's loaded

    # --- Provide an image path for testing ---
    # You need to change this to an actual image path on your system
    # For example, an image from the WIDER_val set.
    # test_image_path = "path/to/your/test_image.jpg"
    # Example: find an image in your WIDER_val set
    example_val_image_dir = os.path.join(config.VAL_IMAGES_DIR, "0--Parade") # Adjust subfolder if needed
    if os.path.exists(example_val_image_dir) and os.listdir(example_val_image_dir):
         test_image_path = os.path.join(example_val_image_dir, os.listdir(example_val_image_dir)[0])
    else:
        test_image_path = None # Set a default if no image found
        print(f"Warning: Could not find a sample image in {example_val_image_dir}. Please set test_image_path manually.")

    if test_image_path and os.path.exists(test_image_path):
        print(f"Predicting on image: {test_image_path}")
        predict_on_image(
            test_image_path, 
            model, 
            config.TARGET_SIZE, 
            CONFIDENCE_THRESHOLD, 
            NMS_IOU_THRESHOLD
        )
    else:
        if test_image_path:
            print(f"Error: Test image not found at {test_image_path}")
        else:
            print("Error: test_image_path is not set. Please provide a valid image path for testing.")