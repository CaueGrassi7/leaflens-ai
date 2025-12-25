"""
LeafLens AI Backend API.

FastAPI application for plant disease detection using a universal
MobileNetV2-based machine learning model.
"""
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf
import keras  # Import standalone keras for better model loading compatibility
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Monkey patch Dense layer to handle quantization_config compatibility
# This must be done BEFORE any model loading attempts
_original_dense_from_config = keras.layers.Dense.from_config

def _patched_dense_from_config(cls, config):
    """Patched from_config that removes quantization_config before deserialization."""
    if isinstance(config, dict):
        config = config.copy()
        # Remove quantization_config from both top level and nested 'config' dict
        config.pop('quantization_config', None)
        if 'config' in config and isinstance(config['config'], dict):
            config['config'] = config['config'].copy()
            config['config'].pop('quantization_config', None)
    return _original_dense_from_config(config)

# Apply monkey patch to both keras.layers.Dense and keras.src.layers.core.dense.Dense
keras.layers.Dense.from_config = classmethod(_patched_dense_from_config)
try:
    from keras.src.layers.core import dense as dense_module
    dense_module.Dense.from_config = classmethod(_patched_dense_from_config)
except (ImportError, AttributeError):
    pass

# Also patch tf.keras.layers.Dense if available
try:
    _original_tf_dense_from_config = tf.keras.layers.Dense.from_config
    
    def _patched_tf_dense_from_config(cls, config):
        """Patched from_config for tf.keras that removes quantization_config."""
        if isinstance(config, dict):
            config = config.copy()
            config.pop('quantization_config', None)
            if 'config' in config and isinstance(config['config'], dict):
                config['config'] = config['config'].copy()
                config['config'].pop('quantization_config', None)
        return _original_tf_dense_from_config(config)
    
    tf.keras.layers.Dense.from_config = classmethod(_patched_tf_dense_from_config)
except (AttributeError, TypeError):
    pass

# Custom Dense layer as fallback
class CompatibleDense(keras.layers.Dense):
    """Dense layer that accepts quantization_config but ignores it for backward compatibility."""
    def __init__(self, *args, quantization_config=None, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        if isinstance(config, dict):
            config = config.copy()
            config.pop('quantization_config', None)
            if 'config' in config and isinstance(config['config'], dict):
                config['config'] = config['config'].copy()
                config['config'].pop('quantization_config', None)
        return super().from_config(config)

# Register custom Dense globally as additional fallback
CUSTOM_OBJECTS = {
    'Dense': CompatibleDense,
    'keras.layers.Dense': CompatibleDense,
    'keras.src.layers.core.dense.Dense': CompatibleDense,
}
keras.utils.get_custom_objects().update(CUSTOM_OBJECTS)
tf.keras.utils.get_custom_objects().update(CUSTOM_OBJECTS)

from backend.app.core.config import settings
from backend.app.schemas.response import (
    EndpointsInfo,
    HealthResponse,
    LowConfidenceResponse,
    PredictionResponse,
    RootResponse,
    StatusInfo,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LeafLens AI API",
    version=settings.API_VERSION,
    description="API for plant disease detection across multiple cultures (Universal Model)"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).parent  # backend/app
PROJECT_ROOT = BASE_DIR.parent.parent  # leaflens-ai root
MODELS_DIR = PROJECT_ROOT / "ml" / "models"

MODEL_PATH = MODELS_DIR / f"plant_disease_model_v{settings.MODEL_VERSION}.keras"
CLASSES_PATH = MODELS_DIR / f"classes_v{settings.MODEL_VERSION}.json"

logger.info("--- SYSTEM STARTUP ---")
logger.info(f"📂 Project Root: {PROJECT_ROOT}")
logger.info(f"🧠 Loading Model Version: {settings.MODEL_VERSION}")

# --- LOAD MODEL ---
MODEL: tf.keras.Model | None = None
try:
    def _clean_quantization_config(config_dict):
        """Recursively remove quantization_config from layer configs."""
        if isinstance(config_dict, dict):
            config_dict = config_dict.copy()
            config_dict.pop('quantization_config', None)
            # Recursively clean nested configs
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    config_dict[key] = _clean_quantization_config(value)
                elif isinstance(value, list):
                    config_dict[key] = [
                        _clean_quantization_config(item) if isinstance(item, dict) else item
                        for item in value
                    ]
        return config_dict
    
    def _load_model_with(loader_name: str, loader_fn, allow_safe_mode: bool) -> tf.keras.Model:
        """Attempt to load the model with the given loader, handling safe_mode availability."""
        kwargs = {
            "filepath": str(MODEL_PATH.resolve()),
            "compile": False,
            "custom_objects": CUSTOM_OBJECTS,
        }
        if allow_safe_mode:
            kwargs["safe_mode"] = False  # Allow custom objects and ignore quantization_config

        try:
            return loader_fn(**kwargs)
        except (TypeError, ValueError) as error:
            # Handle quantization_config errors or safe_mode not supported
            error_msg = str(error)
            if 'quantization_config' in error_msg.lower() or 'safe_mode' in error_msg.lower():
                if allow_safe_mode:
                    logger.warning(f"{loader_name} error (may be quantization_config): {error_msg}")
                    kwargs.pop("safe_mode", None)
                    try:
                        return loader_fn(**kwargs)
                    except Exception as e2:
                        logger.warning(f"Retry without safe_mode also failed: {e2}")
                        raise
            raise

    # Try using standalone keras first (better compatibility with newer Keras 3.x features)
    try:
        MODEL = _load_model_with("keras", keras.models.load_model, allow_safe_mode=True)
        logger.info(f"✅ Model loaded successfully using keras from: {MODEL_PATH.name}")
    except Exception as e:
        # Fallback: try using tf.keras
        logger.warning(f"Failed to load with keras, trying tf.keras: {e}")
        try:
            MODEL = _load_model_with("tf.keras", tf.keras.models.load_model, allow_safe_mode=True)
            logger.info(f"✅ Model loaded successfully using tf.keras from: {MODEL_PATH.name}")
        except Exception as e2:
            # Last resort: try default loading without compile flag
            logger.warning(f"Failed to load with compile=False, trying default load: {e2}")
            MODEL = tf.keras.models.load_model(
                str(MODEL_PATH.resolve()),
                custom_objects=CUSTOM_OBJECTS,
                safe_mode=False
            )
            logger.info(f"✅ Model loaded successfully (default) from: {MODEL_PATH.name}")
except FileNotFoundError as e:
    logger.error(f"❌ Model file not found: {MODEL_PATH}")
    logger.error("Did you run the training script? Ensure the model file exists.")
    MODEL = None
except Exception as e:
    logger.error(f"❌ Critical error loading model: {e}", exc_info=True)
    logger.error("This might be due to TensorFlow/Keras version mismatch.")
    logger.error("Ensure TensorFlow >= 2.18.0 and Keras >= 3.3.0 are installed.")
    logger.error("The model may have been saved with a newer Keras version.")
    logger.error("Consider re-saving the model with the current TensorFlow/Keras version.")
    MODEL = None

# --- LOAD CLASS NAMES DYNAMICALLY ---
# Instead of hardcoding, we read the JSON generated by the training script
CLASS_NAMES: list[str] = []
try:
    with open(CLASSES_PATH.resolve(), "r") as f:
        CLASS_NAMES = json.load(f)
    logger.info(f"✅ Loaded {len(CLASS_NAMES)} classes from JSON.")
    logger.info(f"   Example: {CLASS_NAMES[:3]}...")
except FileNotFoundError:
    logger.error(f"❌ Classes file not found: {CLASSES_PATH}")
    logger.error("Did you run the training script? Ensure 'classes_vX.json' exists.")
    CLASS_NAMES = []
except json.JSONDecodeError as e:
    logger.error(f"❌ Invalid JSON in classes file: {e}")
    CLASS_NAMES = []
except Exception as e:
    logger.error(f"❌ Critical error loading classes JSON: {e}", exc_info=True)
    CLASS_NAMES = []


@app.get("/", response_model=RootResponse)
async def root() -> RootResponse:
    """
    Root endpoint with API information.

    Returns:
        RootResponse: API metadata including version, endpoints, and status.
    """
    return RootResponse(
        name="LeafLens AI API",
        version=settings.API_VERSION,
        model_version=settings.MODEL_VERSION,
        description="API for plant disease detection across multiple cultures (Universal Model)",
        endpoints=EndpointsInfo(
            health="/health",
            predict="/predict",
            docs="/docs",
            redoc="/redoc"
        ),
        status=StatusInfo(
            model_loaded=MODEL is not None,
            classes_loaded=len(CLASS_NAMES) > 0
        )
    )


@app.get("/health", response_model=HealthResponse)
async def ping() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        HealthResponse: System health status and class count if healthy.
    """
    if MODEL is None or not CLASS_NAMES:
        return HealthResponse(
            status="unhealthy",
            reason="Model or classes not loaded"
        )
    return HealthResponse(
        status="healthy",
        classes_count=len(CLASS_NAMES)
    )


def read_file_as_image(data: bytes) -> np.ndarray:
    """
    Convert bytes data to a numpy array image.

    Args:
        data: Image file bytes.

    Returns:
        numpy.ndarray: Image as numpy array.

    Raises:
        HTTPException: If the image cannot be decoded.
    """
    try:
        image = np.array(Image.open(BytesIO(data)))
        return image
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        raise HTTPException(
            status_code=400,
            detail="Invalid image file. Please ensure the file is a valid image format."
        )


@app.post(
    "/predict",
    response_model=Union[PredictionResponse, LowConfidenceResponse]
)
async def predict(
    file: UploadFile = File(...)
) -> Union[PredictionResponse, LowConfidenceResponse]:
    """
    Predict plant disease from an uploaded image.

    Args:
        file: Image file to analyze.

    Returns:
        Union[PredictionResponse, LowConfidenceResponse]:
            - PredictionResponse if confidence >= threshold
            - LowConfidenceResponse if confidence < threshold

    Raises:
        HTTPException: If model or classes are not loaded, or if image
            processing fails.
    """
    # Fail fast if system is not ready
    if MODEL is None:
        logger.error("Prediction attempted but model is not loaded")
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded. Please check server logs."
        )
    if not CLASS_NAMES:
        logger.error("Prediction attempted but class names are not loaded")
        raise HTTPException(
            status_code=500,
            detail="Class names not loaded. Please check server logs."
        )

    try:
        image_data = await file.read()
        image = read_file_as_image(image_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading uploaded file: {e}")
        raise HTTPException(
            status_code=400,
            detail="Failed to process uploaded file."
        )

    # Preprocessing: Expand dims (256,256,3) -> (1,256,256,3)
    img_batch = np.expand_dims(image, 0)

    # Inference
    try:
        predictions = MODEL.predict(img_batch, verbose=0)
    except Exception as e:
        logger.error(f"Model prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Model prediction failed. Please try again."
        )

    # Decoding result
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = float(np.max(predictions[0]))

    # Check confidence threshold for "Open World" problem
    if confidence < settings.CONFIDENCE_THRESHOLD:
        return LowConfidenceResponse(
            class_name="Unidentified",
            confidence=confidence,
            low_confidence=True,
            message=(
                f"Model confidence ({confidence:.0%}) was too low. "
                "The image may not be of a known plant."
            )
        )

    # Return standard prediction for high confidence
    return PredictionResponse(
        class_name=predicted_class,
        confidence=confidence
    )


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
