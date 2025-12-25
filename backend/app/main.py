from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
from pathlib import Path

app = FastAPI()

# --- CONFIGURATION ---
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL LOADING ---
# Get project root: from backend/app/main.py, go up 2 levels
BASE_DIR = Path(__file__).parent  # backend/app
PROJECT_ROOT = BASE_DIR.parent.parent  # leaflens-ai root
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "plant_disease_model_v1.keras"
MODEL_PATH = str(MODEL_PATH.resolve())

print(f"Loading model from: {MODEL_PATH}")

try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    MODEL = None

# --- CLASS NAMES (Tomato Dataset) ---
# Ordered alphabetically as TensorFlow reads them (Uppercase first, then lowercase)
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "LeafLens AI API",
        "version": "1.0.0",
        "description": "API para detecção de doenças em folhas de tomate usando Machine Learning",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "model_loaded": MODEL is not None
    }

@app.get("/health")
async def ping():
    return {"status": "healthy"}

def read_file_as_image(data) -> np.ndarray:
    try:
        image = np.array(Image.open(BytesIO(data)))
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    image = read_file_as_image(await file.read())
    
    # Preprocessing: Expand dims (256,256,3) -> (1,256,256,3)
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)
    
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = float(np.max(predictions[0]))
    
    return {
        "class": predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
