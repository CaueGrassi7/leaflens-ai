"""
Training script for the LeafLens AI plant disease detection model.

This script trains a universal MobileNetV2-based model that can detect
diseases across multiple plant species. It dynamically loads all available
classes from the training data directory and saves both the model and
class mappings for use by the backend API.
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras import layers

# --- CONFIGURATION ---
IMG_SIZE = 256
BATCH_SIZE = 32
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'ml' / 'data' / 'raw'
MODELS_DIR = PROJECT_ROOT / 'ml' / 'models'


def load_data() -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Load training and validation datasets from the data directory.

    Dynamically discovers all plant disease classes from subdirectories
    in the training data folder. Applies data augmentation and optimization
    for efficient training.

    Returns:
        Tuple containing:
            - Training dataset (tf.data.Dataset)
            - Validation dataset (tf.data.Dataset)
            - List of class names (List[str])

    Raises:
        FileNotFoundError: If the data directory or required subdirectories
            do not exist.
        ValueError: If no classes are found in the training directory.
    """
    print(f"📂 Loading all plant cultures from: {DATA_DIR}")

    train_dir = DATA_DIR / 'train'
    val_dir = DATA_DIR / 'val'

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    # Load dataset dynamically (reads all folders found)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        shuffle=True,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        shuffle=True,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    if not class_names:
        raise ValueError("No classes found in training directory")

    print(f"✅ Total Classes Detected: {len(class_names)}")
    print(f"📋 Example: {class_names[:5]}...")  # Show first 5 for verification

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(num_classes: int) -> tf.keras.Model:
    """
    Build a MobileNetV2-based model for plant disease classification.

    Uses transfer learning with MobileNetV2 as the base model, adding
    data augmentation layers and a custom classification head.

    Args:
        num_classes: Number of output classes for the classification task.

    Returns:
        Compiled Keras model ready for training.
    """
    print(f"🏗️ Adapting MobileNetV2 for {num_classes} classes...")

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ])

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def save_artifacts(
    model: tf.keras.Model,
    class_names: List[str],
    version: str
) -> None:
    """
    Save the trained model and class mappings to disk.

    Saves both the Keras model file and a JSON file containing the
    class names. This is crucial for the backend to dynamically load
    the correct classes for the universal model.

    Args:
        model: Trained Keras model to save.
        class_names: List of class names corresponding to model outputs.
        version: Version identifier for the model (e.g., "universal_v1").

    Raises:
        OSError: If the models directory cannot be created or files
            cannot be written.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save Model
    model_path = MODELS_DIR / f"plant_disease_model_v{version}.keras"
    try:
        model.save(model_path)
        print(f"💾 Model saved: {model_path}")
    except Exception as e:
        raise OSError(f"Failed to save model to {model_path}: {e}")

    # 2. Save Classes (Critical for Universal Backend)
    classes_path = MODELS_DIR / f"classes_v{version}.json"
    try:
        with open(classes_path, 'w') as f:
            json.dump(class_names, f)
        print(f"📝 Classes saved: {classes_path}")
    except Exception as e:
        raise OSError(f"Failed to save classes to {classes_path}: {e}")


def main(epochs: int, version: str) -> None:
    """
    Main training function that orchestrates the entire training process.

    Loads data, builds the model, trains it, and saves all artifacts.

    Args:
        epochs: Number of training epochs.
        version: Version identifier for the model (e.g., "universal_v1").
    """
    train_ds, val_ds, class_names = load_data()
    model = build_model(len(class_names))

    print(f"🏋️ Training 'Universal' model for {epochs} epochs...")
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        verbose=1
    )

    save_artifacts(model, class_names, version)
    print("✨ Training Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a universal plant disease detection model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="universal_v1",
        help="Model version identifier (default: universal_v1)"
    )
    args = parser.parse_args()
    main(args.epochs, args.version)
