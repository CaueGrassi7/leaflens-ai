import os
import json
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
import matplotlib.pyplot as plt

# --- CONFIGURAÇÕES ---
IMG_SIZE = 256
BATCH_SIZE = 32
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'ml' / 'data' / 'raw'
MODELS_DIR = PROJECT_ROOT / 'ml' / 'models'

def load_data():
    print(f"📂 Carregando TODAS as culturas de: {DATA_DIR}")
    
    train_dir = DATA_DIR / 'train'
    val_dir = DATA_DIR / 'val'

    # Carrega dataset dinamicamente (lê todas as pastas que encontrar)
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
    print(f"✅ Total de Classes Detectadas: {len(class_names)}")
    print(f"📋 Exemplo: {class_names[:5]}...") # Mostra as 5 primeiras para conferir

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

def build_model(num_classes):
    print(f"🏗️ Adaptando MobileNetV2 para {num_classes} classes...")
    
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

def save_artifacts(model, class_names, version):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Salvar Modelo
    model_path = MODELS_DIR / f"plant_disease_model_v{version}.keras"
    model.save(model_path)
    print(f"💾 Modelo salvo: {model_path}")

    # 2. SALVAR AS CLASSES (Crucial para o Backend Universal)
    classes_path = MODELS_DIR / f"classes_v{version}.json"
    with open(classes_path, 'w') as f:
        json.dump(class_names, f)
    print(f"📝 Classes salvas: {classes_path}")

def main(epochs, version):
    train_ds, val_ds, class_names = load_data()
    model = build_model(len(class_names))
    
    print(f"🏋️ Treinando modelo 'Generalista' por {epochs} épocas...")
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        verbose=1
    )
    
    save_artifacts(model, class_names, version)
    print("✨ Treinamento Completo!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--version", type=str, default="universal_v1")
    args = parser.parse_args()
    main(args.epochs, args.version)
