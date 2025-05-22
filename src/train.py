# src/train.py
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
import datetime

from . import config
from . import data_loader
from . import model as model_builder # This will now use your custom CNN from model.py

def build_and_compile_custom_cnn_model(): # Renamed for clarity
    """Builds and compiles the custom CNN face detection model."""
    # This now calls your original model builder
    face_detector = model_builder.build_face_detector_model(
        input_shape=(config.TARGET_SIZE[0], config.TARGET_SIZE[1], 3)
    )

    losses = {
        "class_output": tf.keras.losses.BinaryCrossentropy(from_logits=False),
        "bbox_output": tf.keras.losses.MeanSquaredError()
    }
    loss_weights = {"class_output": 1.0, "bbox_output": 1.0}
    metrics = {
        "class_output": ["accuracy", tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
        "bbox_output": ["mse", tf.keras.metrics.MeanAbsoluteError(name='mae')]
    }

    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    face_detector.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
    print("Custom CNN Model compiled successfully.")
    face_detector.summary()
    return face_detector

def train_custom_cnn_with_limited_data(): # Renamed for clarity
    print("Starting training process for CUSTOM CNN with LIMITED DATA...")

    # 1. Load and prepare data (using subsets defined in config)
    print("Loading ALL training annotations initially...")
    all_train_annotations = data_loader.parse_wider_face_annotations(
        config.TRAIN_ANNOT_FILE, config.TRAIN_IMAGES_DIR
    )
    
    # Use subset based on config.MAX_TRAIN_IMAGES
    if config.MAX_TRAIN_IMAGES is not None and len(all_train_annotations) > config.MAX_TRAIN_IMAGES:
        train_annotations = all_train_annotations[:config.MAX_TRAIN_IMAGES]
        print(f"Using a SUBSET of {len(train_annotations)} (out of {len(all_train_annotations)}) image entries for training.")
    else:
        train_annotations = all_train_annotations
        print(f"Using FULL {len(train_annotations)} image entries for training (or MAX_TRAIN_IMAGES is not restrictive).")

    if not train_annotations:
        print("No training annotations to use. Exiting.")
        return

    # Estimate shuffle buffer size based on the actual number of annotations being used
    estimated_train_samples = len(train_annotations) * (1 + config.NEGATIVES_PER_POSITIVE_RATIO) * 1.5 
    train_shuffle_buffer = min(int(estimated_train_samples), 30000) # Cap buffer size
    print(f"Using shuffle buffer size for training: {train_shuffle_buffer}")

    train_dataset = data_loader.create_dataset(
        train_annotations, # Pass the (potentially sliced) list
        target_size=config.TARGET_SIZE, batch_size=config.BATCH_SIZE,
        neg_iou_thresh=config.NEGATIVE_IOU_THRESHOLD,
        negatives_per_positive_ratio=config.NEGATIVES_PER_POSITIVE_RATIO,
        is_training=True, shuffle_buffer_size=train_shuffle_buffer
    )
    print("Training dataset created.")

    # --- VALIDATION DATA (using subset defined in config) ---
    print("Loading ALL validation annotations initially...")
    all_val_annotations = data_loader.parse_wider_face_annotations(
        config.VAL_ANNOT_FILE, config.VAL_IMAGES_DIR
    )

    # Use subset based on config.MAX_VAL_IMAGES
    if config.MAX_VAL_IMAGES is not None and len(all_val_annotations) > config.MAX_VAL_IMAGES:
        val_annotations = all_val_annotations[:config.MAX_VAL_IMAGES]
        print(f"Using a SUBSET of {len(val_annotations)} (out of {len(all_val_annotations)}) image entries for validation.")
    else:
        val_annotations = all_val_annotations
        print(f"Using FULL {len(val_annotations)} image entries for validation (or MAX_VAL_IMAGES is not restrictive).")
    
    val_dataset = None
    if val_annotations:
        val_neg_ratio = config.NEGATIVES_PER_POSITIVE_RATIO * 0.5 # Can adjust for validation
        estimated_val_samples = len(val_annotations) * (1 + val_neg_ratio) * 1.5
        val_shuffle_buffer = min(int(estimated_val_samples), 10000) # Smaller cap for val

        val_dataset = data_loader.create_dataset(
            val_annotations, # Pass the (potentially sliced) list
            target_size=config.TARGET_SIZE, batch_size=config.BATCH_SIZE,
            neg_iou_thresh=config.NEGATIVE_IOU_THRESHOLD,
            negatives_per_positive_ratio=val_neg_ratio,
            is_training=False, shuffle_buffer_size=val_shuffle_buffer
        )
        print("Validation dataset created.")
    else:
        print("No validation annotations to use or validation set creation skipped.")
    
    # 2. Build and compile the custom CNN model
    face_detector_model = build_and_compile_custom_cnn_model()

    # 3. Define Callbacks
    log_dir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_custom_cnn_limited_data"
    log_dir = os.path.join(config.BASE_DIR, "logs", "fit", log_dir_name)
    
    best_model_filepath = os.path.join(config.MODELS_TRAINED_DIR, f"custom_cnn_best_{log_dir_name}.keras")
    
    monitor_metric = 'val_loss' if val_dataset else 'loss'
    
    model_checkpoint_callback = ModelCheckpoint(
        filepath=best_model_filepath,
        save_weights_only=False, monitor=monitor_metric, mode='min', save_best_only=True, verbose=1
    )
    reduce_lr_callback = ReduceLROnPlateau(
        monitor=monitor_metric, factor=0.2, patience=3, min_lr=1e-6, verbose=1
    )
    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric, patience=7, verbose=1, restore_best_weights=True
    )
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks_list = [
        model_checkpoint_callback,
        reduce_lr_callback,
        early_stopping_callback,
        tensorboard_callback
    ]
    
    # 4. Train the model
    print(f"Starting model training for up to {config.EPOCHS} epochs...")
    print(f"Monitoring: {monitor_metric}")
    print(f"Batch size: {config.BATCH_SIZE}, Target size: {config.TARGET_SIZE}")
    
    history = face_detector_model.fit(
        train_dataset,
        epochs=config.EPOCHS,
        callbacks=callbacks_list,
        validation_data=val_dataset
    )

    print("Training finished.")

    final_model_path = os.path.join(config.MODELS_TRAINED_DIR, f"custom_cnn_final_{log_dir_name}.keras")
    face_detector_model.save(final_model_path)
    print(f"Final model saved to {final_model_path} (best weights might have been restored by EarlyStopping).")

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            print(len(gpus), "Physical GPUs available.")
        except RuntimeError as e: print(e)
    else:
        print("No GPU found, TensorFlow will use CPU. Training the custom CNN will be slow.")
    
    train_custom_cnn_with_limited_data()