# src/data_loader.py
import os
import cv2
import numpy as np
import tensorflow as tf
# This import is for when data_loader is imported as part of the 'src' package
from . import config as package_config # Renamed to avoid conflict if you run directly

def parse_wider_face_annotations(annot_file_path, images_dir_path):
    """
    Parses WIDER FACE annotation file.
    Returns a list of dictionaries, where each dictionary contains:
    'image_path': full path to the image
    'bboxes': list of [x1, y1, w, h] for valid faces
    """
    data = []
    with open(annot_file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    file_count = 0
    while i < len(lines):
        image_relative_path = lines[i].strip()
        image_full_path = os.path.join(images_dir_path, image_relative_path)
        i += 1
        file_count +=1

        num_bboxes_line = lines[i].strip()
        try:
            num_bboxes = int(num_bboxes_line)
        except ValueError:
            print(f"Warning: Could not parse num_bboxes: '{num_bboxes_line}' for image {image_relative_path}. Skipping file entry.")
            while i < len(lines) and not lines[i].strip().endswith(('.jpg', '.png', '.jpeg')):
                i += 1
            if i == len(lines):
                break
            continue
        i += 1

        bboxes = []
        if num_bboxes == 0:
            if i < len(lines) and lines[i].strip().count(' ') > 1:
                 i +=1
        else:
            for _ in range(num_bboxes):
                if i >= len(lines):
                    print(f"Warning: Unexpected end of file while reading bboxes for {image_relative_path}")
                    break
                bbox_info = lines[i].strip().split()
                try:
                    x1, y1, w, h = [int(x) for x in bbox_info[:4]]
                    invalid_flag = int(bbox_info[7])
                except (ValueError, IndexError):
                    print(f"Warning: Malformed bbox line '{lines[i].strip()}' for {image_relative_path}. Skipping bbox.")
                    i += 1
                    continue

                if w > 0 and h > 0 and invalid_flag == 0:
                    bboxes.append([x1, y1, w, h])
                i += 1
        
        if os.path.exists(image_full_path):
            data.append({'image_path': image_full_path, 'bboxes': bboxes})
            
    print(f"Parsed {len(data)} image entries from {file_count} file names in annotation.")
    return data

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Boxes are [x1, y1, w, h].
    """
    x1_1, y1_1, w1, h1 = box1
    x2_1, y2_1, w2, h2 = box2

    x1_2, y1_2 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x2_1 + w2, y2_1 + h2

    xi1 = max(x1_1, x2_1)
    yi1 = max(y1_1, y2_1)
    xi2 = min(x1_2, x2_2)
    yi2 = min(y1_2, y2_2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0 # Return float
    
    iou = inter_area / union_area
    return iou

def generate_training_samples(image_path, gt_bboxes, target_size, neg_iou_thresh, num_neg_to_generate=1):
    """
    Generates positive and negative training samples from a single image.
    """
    samples = []
    image = cv2.imread(image_path)
    if image is None:
        return samples
    
    img_h, img_w = image.shape[:2]

    # --- Generate Positive Samples ---
    for gt_bbox in gt_bboxes:
        x, y, w, h = gt_bbox
        scale_factor = 1.2
        crop_w = int(w * scale_factor)
        crop_h = int(h * scale_factor)
        crop_x = max(0, x - (crop_w - w) // 2)
        crop_y = max(0, y - (crop_h - h) // 2)
        
        crop_x2 = min(img_w, crop_x + crop_w)
        crop_y2 = min(img_h, crop_y + crop_h)
        # Update crop_w, crop_h based on actual cropped dimensions
        current_crop_w = crop_x2 - crop_x 
        current_crop_h = crop_y2 - crop_y

        if current_crop_w <=0 or current_crop_h <=0:
            continue

        positive_crop = image[crop_y:crop_y2, crop_x:crop_x2]
        if positive_crop.size == 0:
            continue
        
        resized_crop = cv2.resize(positive_crop, (target_size[1], target_size[0]))

        norm_bbox_x = (x - crop_x) / current_crop_w if current_crop_w > 0 else 0
        norm_bbox_y = (y - crop_y) / current_crop_h if current_crop_h > 0 else 0
        norm_bbox_w = w / current_crop_w if current_crop_w > 0 else 0
        norm_bbox_h = h / current_crop_h if current_crop_h > 0 else 0
        
        norm_bbox = [
            np.clip(norm_bbox_x, 0.0, 1.0),
            np.clip(norm_bbox_y, 0.0, 1.0),
            np.clip(norm_bbox_w, 0.0, 1.0),
            np.clip(norm_bbox_h, 0.0, 1.0)
        ]
        samples.append((resized_crop, 1, norm_bbox))

    # --- Generate Negative Samples ---
    generated_neg_count = 0
    attempts = 0
    
    while generated_neg_count < num_neg_to_generate and attempts < num_neg_to_generate * 10: # Increased attempts
        attempts += 1
        min_crop_dim = min(target_size[0], target_size[1])
        # Ensure max_crop_dim is at least min_crop_dim
        max_crop_dim_w = max(min_crop_dim, img_w // 2)
        max_crop_dim_h = max(min_crop_dim, img_h // 2)

        if max_crop_dim_w < min_crop_dim or max_crop_dim_h < min_crop_dim :
            break 

        # Ensure randint upper bound is greater than lower bound
        # Ensure lower bound is not 0 if upper bound is 0 or 1
        actual_min_w = min(min_crop_dim, max_crop_dim_w)
        actual_min_h = min(min_crop_dim, max_crop_dim_h)
        
        neg_crop_w = np.random.randint(actual_min_w, max(actual_min_w + 1, max_crop_dim_w + 1))
        neg_crop_h = np.random.randint(actual_min_h, max(actual_min_h + 1, max_crop_dim_h + 1))


        if img_w - neg_crop_w <= 0 or img_h - neg_crop_h <= 0:
            continue

        neg_x = np.random.randint(0, max(1, img_w - neg_crop_w +1)) # +1 to include upper bound if img_w-neg_crop_w is 0
        neg_y = np.random.randint(0, max(1, img_h - neg_crop_h +1))
        neg_crop_box = [neg_x, neg_y, neg_crop_w, neg_crop_h]

        is_valid_negative = True
        if not gt_bboxes:
            pass
        else:
            for gt_bbox in gt_bboxes:
                iou = calculate_iou(neg_crop_box, gt_bbox)
                if iou > neg_iou_thresh:
                    is_valid_negative = False
                    break
        
        if is_valid_negative:
            negative_crop = image[neg_y:neg_y + neg_crop_h, neg_x:neg_x + neg_crop_w]
            if negative_crop.size == 0:
                continue
            
            resized_neg_crop = cv2.resize(negative_crop, (target_size[1], target_size[0]))
            samples.append((resized_neg_crop, 0, [0.0, 0.0, 0.0, 0.0]))
            generated_neg_count += 1
            
    return samples

def preprocess_image(image_array):
    """ Normalize image to [0,1] """
    image_array = tf.cast(image_array, tf.float32)
    image_array = image_array / 255.0
    return image_array

# --- Data Augmentation (Simple Example) ---
def augment_data(image, bbox):
    """Applies simple augmentation: random horizontal flip.
    Ensures bbox remains a tf.Tensor.
    """
    # image and bbox are expected to be tf.Tensors here
    
    # Random horizontal flip
    # tf.random.uniform(()) returns a scalar tensor, shape ()
    if tf.random.uniform(shape=(), minval=0.0, maxval=1.0) > 0.5: # Explicit shape and range
        image = tf.image.flip_left_right(image)
        
        # bbox is [x_tl, y_tl, w, h]
        # x_new_tl = 1.0 - old_x_tl - old_w
        flipped_x_tl = 1.0 - bbox[0] - bbox[2]
        
        # Reconstruct the bbox tensor using tf.stack
        # This ensures the output is a tensor of the same rank (1D) and shape (4,)
        bbox_list = [flipped_x_tl, bbox[1], bbox[2], bbox[3]]
        bbox = tf.stack(bbox_list)
        
    # bbox is returned, either original tensor or the new stacked tensor
    return image, bbox


def create_dataset(annotations_list, target_size, batch_size, neg_iou_thresh, 
                   negatives_per_positive_ratio, is_training=True, shuffle_buffer_size=10000):
    """
    Creates a tf.data.Dataset from the parsed annotations.
    """
    all_image_paths = []
    all_gt_bboxes_for_paths = [] 

    for item in annotations_list:
        all_image_paths.append(item['image_path'])
        all_gt_bboxes_for_paths.append(item['bboxes'])
    
    print(f"Processing {len(all_image_paths)} images to generate samples for this dataset split...")

    def generator():
        num_positives_generated = 0
        num_total_samples = 0
        for img_path, gt_bboxes_for_img in zip(all_image_paths, all_gt_bboxes_for_paths):
            num_neg_for_this_image = int(len(gt_bboxes_for_img) * negatives_per_positive_ratio)
            if not gt_bboxes_for_img: 
                num_neg_for_this_image = max(1, int(negatives_per_positive_ratio)) 

            samples_from_img = generate_training_samples(
                img_path, 
                gt_bboxes_for_img, 
                target_size, 
                neg_iou_thresh,
                num_neg_to_generate=num_neg_for_this_image
            )
            for crop, label, bbox_target in samples_from_img:
                num_total_samples +=1
                if label == 1:
                    num_positives_generated +=1
                yield crop, (tf.constant([label], dtype=tf.float32), tf.constant(bbox_target, dtype=tf.float32))
        print(f"Total positive samples generated by generator for this split: {num_positives_generated}")
        print(f"Total samples (pos+neg) generated by generator for this split: {num_total_samples}")


    output_signature = (
        tf.TensorSpec(shape=(target_size[0], target_size[1], 3), dtype=tf.uint8), 
        (tf.TensorSpec(shape=(1,), dtype=tf.float32),
         tf.TensorSpec(shape=(4,), dtype=tf.float32))
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    def map_fn(image_crop, targets):
        class_label, bbox_target = targets 
        image_crop = preprocess_image(image_crop) 
        
        if is_training:
            image_crop, bbox_target = augment_data(image_crop, bbox_target)
            bbox_target = tf.clip_by_value(bbox_target, clip_value_min=0.0, clip_value_max=1.0)

        return image_crop, (class_label, bbox_target)

    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE) 

    return dataset


# --- Example Usage (for testing this script directly) ---
if __name__ == '__main__':
    import sys
    current_script_path = os.path.abspath(__file__)
    src_directory = os.path.dirname(current_script_path)
    project_root = os.path.dirname(src_directory) 
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src import config 

    print("Testing data_loader.py...")
    
    print("Parsing annotations (this might take a moment for the full dataset)...")
    train_annotations = parse_wider_face_annotations(config.TRAIN_ANNOT_FILE, config.TRAIN_IMAGES_DIR)
    
    # To test with a small subset for direct execution:
    # test_subset_annotations = train_annotations[:config.MAX_TRAIN_IMAGES if config.MAX_TRAIN_IMAGES else 100]
    test_subset_annotations = train_annotations[:100] # Fixed small subset for direct test
    
    print(f"Using {len(test_subset_annotations)} image entries for dataset creation test.")

    if not test_subset_annotations:
        print("No annotations found or parsed. Please check paths and parsing logic.")
    else:
        estimated_num_samples = len(test_subset_annotations) * (1 + config.NEGATIVES_PER_POSITIVE_RATIO) * 1.5 
        buffer_size = min(int(estimated_num_samples), 20000) 
        print(f"Estimated shuffle buffer size for test: {buffer_size}")
        
        train_dataset = create_dataset(
            test_subset_annotations, # Use the small subset for direct test
            target_size=config.TARGET_SIZE,
            batch_size=config.BATCH_SIZE,
            neg_iou_thresh=config.NEGATIVE_IOU_THRESHOLD,
            negatives_per_positive_ratio=config.NEGATIVES_PER_POSITIVE_RATIO,
            is_training=True,
            shuffle_buffer_size=buffer_size
        )

        print("Dataset created for test. Taking one batch to inspect...")
        num_pos_in_batch = 0
        num_neg_in_batch = 0
        for images, targets in train_dataset.take(1): 
            class_labels, bbox_targets = targets
            print("Images batch shape:", images.shape) # (BATCH_SIZE, H, W, C)
            print("Class labels batch shape:", class_labels.shape) # (BATCH_SIZE, 1)
            print("BBox targets batch shape:", bbox_targets.shape) # (BATCH_SIZE, 4)
            
            for lbl in class_labels.numpy():
                if lbl[0] == 1: num_pos_in_batch +=1
                else: num_neg_in_batch +=1
            print(f"Positive samples in batch: {num_pos_in_batch}, Negative: {num_neg_in_batch}")

            print("\nSample from batch (first image):")
            print("Image min/max pixel values:", tf.reduce_min(images[0]).numpy(), tf.reduce_max(images[0]).numpy())
            print("Class label:", class_labels[0].numpy())
            print("BBox target:", bbox_targets[0].numpy())

            import matplotlib.pyplot as plt
            plt.imshow(images[0].numpy()) 
            
            if class_labels[0].numpy()[0] == 1:
                bbox = bbox_targets[0].numpy()
                h_img, w_img = config.TARGET_SIZE
                x1 = int(bbox[0] * w_img); y1 = int(bbox[1] * h_img)
                bw = int(bbox[2] * w_img); bh = int(bbox[3] * h_img)
                img_to_show = images[0].numpy().copy()
                cv2.rectangle(img_to_show, (x1,y1), (x1+bw, y1+bh), (0,1,0), 1) 
                plt.imshow(img_to_show)
            plt.title(f"Sample from Batch. Label: {class_labels[0].numpy()[0]}")
            plt.show()
            break