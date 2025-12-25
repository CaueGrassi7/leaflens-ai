"""
Pydantic response models for API endpoints.

This module defines the structure and validation for all API responses,
ensuring type safety and consistent response formats.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict, model_serializer


class PredictionResponse(BaseModel):
    """Standard prediction response with class and confidence."""

    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(..., alias="class", description="Predicted disease class")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Serialize model using alias for 'class' field."""
        return {
            "class": self.class_name,
            "confidence": self.confidence
        }


class LowConfidenceResponse(PredictionResponse):
    """Response for predictions below the confidence threshold."""

    low_confidence: bool = Field(
        True,
        description="Flag indicating low confidence prediction"
    )
    message: str = Field(
        ...,
        description="Human-readable message explaining the low confidence"
    )


class HealthResponse(BaseModel):
    """Health check endpoint response."""

    status: str = Field(..., description="Health status: 'healthy' or 'unhealthy'")
    classes_count: Optional[int] = Field(
        None,
        description="Number of classes loaded (only present when healthy)"
    )
    reason: Optional[str] = Field(
        None,
        description="Reason for unhealthy status (only present when unhealthy)"
    )


class EndpointsInfo(BaseModel):
    """Information about available API endpoints."""

    health: str = Field(..., description="Health check endpoint path")
    predict: str = Field(..., description="Prediction endpoint path")
    docs: str = Field(..., description="API documentation endpoint path")
    redoc: str = Field(..., description="ReDoc documentation endpoint path")


class StatusInfo(BaseModel):
    """System status information."""

    model_loaded: bool = Field(..., description="Whether the model is loaded")
    classes_loaded: bool = Field(..., description="Whether classes are loaded")


class RootResponse(BaseModel):
    """Root endpoint response with API information."""

    name: str = Field(..., description="API name")
    version: str = Field(..., description="API version")
    model_version: str = Field(..., description="Model version identifier")
    description: str = Field(..., description="API description")
    endpoints: EndpointsInfo = Field(..., description="Available endpoints")
    status: StatusInfo = Field(..., description="System status")

