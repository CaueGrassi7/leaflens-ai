"""
Configuration constants for the LeafLens AI backend API.

This module contains all configuration constants used throughout the application,
including model settings, API settings, and CORS configuration.
"""
from typing import List


class Settings:
    """Application settings and configuration constants."""

    # Model Configuration
    CONFIDENCE_THRESHOLD: float = 0.70  # 70% minimum confidence for predictions
    MODEL_VERSION: str = "universal_v1"

    # API Configuration
    API_VERSION: str = "2.0.0"

    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
    ]


# Export settings instance for easy access
settings = Settings()

