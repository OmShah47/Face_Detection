# src/config.py
import os

# --- Project Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
WIDER_FACE_DIR = os.path.join(DATA_DIR, 'WIDER_FACE')

TRAIN_IMAGES_DIR = os.path.join(WIDER_FACE_DIR, 'WIDER_train/images/')
TRAIN_ANNOT_FILE = os.path.join(WIDER_FACE_DIR, 'wider_face_split/wider_face_train_bbx_gt.txt')

VAL_IMAGES_DIR = os.path.join(WIDER_FACE_DIR, 'WIDER_val/images/')
VAL_ANNOT_FILE = os.path.join(WIDER_FACE_DIR, 'wider_face_split/wider_face_val_bbx_gt.txt')

MODELS_TRAINED_DIR = os.path.join(BASE_DIR, 'models_trained')
os.makedirs(MODELS_TRAINED_DIR, exist_ok=True)

# --- Data Parameters ---
# Keep TARGET_SIZE relatively small for faster processing
TARGET_SIZE = (96, 96) # Reduced from (128, 128) - significantly less data per image
NUM_CLASSES = 1

# --- Data Limiting Parameters (CRITICAL FOR CPU/iGPU) ---
# Max number of IMAGE FILES (not individual samples) to use from WIDER FACE.
# Start VERY small to ensure epochs complete quickly.
MAX_TRAIN_IMAGES = 100   # Start with a very small number of image files
MAX_VAL_IMAGES = 20      # Also very small for validation

# --- Training Parameters ---
BATCH_SIZE = 16 # Smaller batch size is generally better for CPU/limited memory
EPOCHS = 15     # You can run more epochs if each epoch is fast
LEARNING_RATE = 1e-4 # Keep as is for now, can adjust later

# --- Negative Sample Generation (CRITICAL FOR CPU/iGPU) ---
# Reduce the number of negative samples to speed up data generation and reduce total samples.
NEGATIVE_IOU_THRESHOLD = 0.2 # Keep as is
NEGATIVES_PER_POSITIVE_RATIO = 0.25 # Generate significantly fewer negatives.
                                   # For 4 positive samples, it will try to find 1 negative.

# --- Bounding Box Normalization ---
# Target bounding box coordinates are normalized relative to the patch size.