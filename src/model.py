# src/model.py
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     BatchNormalization, ReLU, Dropout, Input)
from tensorflow.keras.models import Model
from . import config as package_config # Use package_config for default TARGET_SIZE

def build_face_detector_model(input_shape=(package_config.TARGET_SIZE[0], package_config.TARGET_SIZE[1], 3)):
    """
    Builds the original custom CNN model for face detection.
    """
    inputs = Input(shape=input_shape, name="input_image")

    # --- Shared Convolutional Base ---
    # Block 1
    x = Conv2D(32, (3, 3), padding="same", name="conv1_1")(inputs)
    x = BatchNormalization(name="bn1_1")(x)
    x = ReLU(name="relu1_1")(x)
    x = Conv2D(32, (3, 3), padding="same", name="conv1_2")(x)
    x = BatchNormalization(name="bn1_2")(x)
    x = ReLU(name="relu1_2")(x)
    x = MaxPooling2D((2, 2), name="pool1")(x)
    x = Dropout(0.25, name="dropout1")(x)

    # Block 2
    x = Conv2D(64, (3, 3), padding="same", name="conv2_1")(x)
    x = BatchNormalization(name="bn2_1")(x)
    x = ReLU(name="relu2_1")(x)
    x = Conv2D(64, (3, 3), padding="same", name="conv2_2")(x)
    x = BatchNormalization(name="bn2_2")(x)
    x = ReLU(name="relu2_2")(x)
    x = MaxPooling2D((2, 2), name="pool2")(x)
    x = Dropout(0.25, name="dropout2")(x)

    # Block 3
    x = Conv2D(128, (3, 3), padding="same", name="conv3_1")(x)
    x = BatchNormalization(name="bn3_1")(x)
    x = ReLU(name="relu3_1")(x)
    x = Conv2D(128, (3, 3), padding="same", name="conv3_2")(x)
    x = BatchNormalization(name="bn3_2")(x)
    x = ReLU(name="relu3_2")(x)
    x = MaxPooling2D((2, 2), name="pool3")(x)
    x = Dropout(0.25, name="dropout3")(x)

    # Flatten for Dense layers
    base_output = Flatten(name="flatten")(x)

    # --- Head 1: Classification (Face / No-Face) ---
    class_head = Dense(128, name="class_fc1")(base_output)
    class_head = BatchNormalization(name="class_bn1")(class_head)
    class_head = ReLU(name="class_relu1")(class_head)
    class_head = Dropout(0.5, name="class_dropout1")(class_head)
    class_output = Dense(1, activation="sigmoid", name="class_output")(class_head)

    # --- Head 2: Bounding Box Regression ---
    bbox_head = Dense(128, name="bbox_fc1")(base_output)
    bbox_head = BatchNormalization(name="bbox_bn1")(bbox_head)
    bbox_head = ReLU(name="bbox_relu1")(bbox_head)
    bbox_head = Dropout(0.5, name="bbox_dropout1")(bbox_head)
    bbox_output = Dense(4, activation="sigmoid", name="bbox_output")(bbox_head)

    model = Model(
        inputs=inputs,
        outputs=[class_output, bbox_output],
        name="face_detector_custom_cnn" # New name to distinguish
    )
    return model

# --- Example Usage (for testing this script directly) ---
if __name__ == '__main__':
    import sys
    import os
    current_script_path = os.path.abspath(__file__)
    src_directory = os.path.dirname(current_script_path)
    project_root = os.path.dirname(src_directory)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src import config # For direct script execution config

    model = build_face_detector_model(
        input_shape=(config.TARGET_SIZE[0], config.TARGET_SIZE[1], 3)
    )
    model.summary()

    try:
        output_path = os.path.join(config.BASE_DIR, "custom_cnn_model_architecture.png")
        tf.keras.utils.plot_model(model, to_file=output_path, show_shapes=True, show_layer_names=True)
        print(f"Model architecture plotted to {output_path}")
    except Exception as e:
        print(f"Could not plot model (ensure pydot and graphviz are installed): {e}")