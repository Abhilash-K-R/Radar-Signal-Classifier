"""
FastAPI REST API for Radar Signal Classification

Provides HTTP endpoints for:
- /predict — Classify a signal (generate or upload)
- /health — Health check
- /models — List available models and their metadata

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import io
import numpy as np
import pandas as pd
from typing import Optional
from enum import Enum

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure src is importable
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC_DIR)

try:
    from src.config import SAMPLE_RATE, N_SAMPLES, RESULTS_DIR
    from src.predictor import RadarPredictor, generate_signal
except ImportError:
    from config import SAMPLE_RATE, N_SAMPLES, RESULTS_DIR
    from predictor import RadarPredictor, generate_signal


# ─── App Setup ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Radar Signal Classifier API",
    description=(
        "ML-powered API for classifying synthetic radar signals. "
        "Supports sine, square, chirp, and noisy signal types using "
        "Random Forest (95.6% accuracy) or CNN (100% on clean data)."
    ),
    version="2.0.0",
    contact={"name": "Abhilash K R", "url": "https://github.com/Abhilash-K-R"},
)

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded predictors (loaded on first request)
_predictors = {}


def get_predictor(model_type: str) -> RadarPredictor:
    """Get or create a predictor for the given model type."""
    if model_type not in _predictors:
        _predictors[model_type] = RadarPredictor(model_type=model_type)
    return _predictors[model_type]


# ─── Request/Response Models ──────────────────────────────────────────────

class SignalType(str, Enum):
    sine = "sine"
    square = "square"
    chirp = "chirp"
    noisy = "noisy"


class ModelType(str, Enum):
    rf = "rf"
    cnn = "cnn"


class GenerateAndPredictRequest(BaseModel):
    """Generate a synthetic signal and classify it."""
    signal_type: SignalType = Field(..., description="Type of signal to generate")
    frequency: float = Field(5.0, ge=0.1, le=100.0, description="Base frequency in Hz")
    amplitude: float = Field(1.0, ge=0.01, le=10.0, description="Signal amplitude")
    model: ModelType = Field(ModelType.rf, description="Model to use for prediction")
    f1: Optional[float] = Field(50.0, ge=1.0, le=200.0, description="End frequency for chirp signals")
    noise_level: Optional[float] = Field(1.5, ge=0.0, le=10.0, description="Noise level for noisy signals")


class PredictFromArrayRequest(BaseModel):
    """Classify a signal from raw sample values."""
    signal: list[float] = Field(..., description=f"Signal samples (must be {N_SAMPLES} values)")
    model: ModelType = Field(ModelType.rf, description="Model to use for prediction")


class PredictionResponse(BaseModel):
    """Prediction result."""
    label: str
    confidence: float
    probabilities: dict[str, float]
    model_type: str
    signal_info: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    models_available: dict[str, bool]
    sample_rate: int
    signal_length: int


class ModelInfo(BaseModel):
    name: str
    type: str
    accuracy_clean: str
    accuracy_noisy_5db: str
    robustness: str
    features: str


# ─── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/", tags=["root"])
async def root():
    """API welcome page."""
    return {
        "message": "🛰️ Radar Signal Classifier API",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "POST /predict/generate": "Generate a signal and classify it",
            "POST /predict/array": "Classify from raw signal array",
            "POST /predict/upload": "Upload CSV and classify",
            "GET /health": "Health check",
            "GET /models": "List available models",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Check API health and model availability."""
    rf_available = os.path.exists(os.path.join(RESULTS_DIR, "best_model.pkl"))
    cnn_available = os.path.exists(os.path.join(RESULTS_DIR, "cnn_model.pth"))

    return HealthResponse(
        status="healthy",
        models_available={"rf": rf_available, "cnn": cnn_available},
        sample_rate=SAMPLE_RATE,
        signal_length=N_SAMPLES,
    )


@app.get("/models", response_model=list[ModelInfo], tags=["system"])
async def list_models():
    """List available models and their characteristics."""
    return [
        ModelInfo(
            name="Random Forest",
            type="rf",
            accuracy_clean="95.63%",
            accuracy_noisy_5db="63.00%",
            robustness="High — graceful degradation",
            features="8 hand-engineered (mean, std, RMS, energy, peak-to-peak, dominant freq, max, min)",
        ),
        ModelInfo(
            name="CNN (1D Convolutional)",
            type="cnn",
            accuracy_clean="100%",
            accuracy_noisy_5db="36.75%",
            robustness="Low — sudden collapse under noise",
            features="Learned automatically from raw 500-sample signals",
        ),
    ]


@app.post("/predict/generate", response_model=PredictionResponse, tags=["prediction"])
async def predict_from_generated(request: GenerateAndPredictRequest):
    """Generate a synthetic signal and classify it."""
    try:
        kwargs = {}
        if request.signal_type == SignalType.chirp:
            kwargs["f0"] = request.frequency
            kwargs["f1"] = request.f1
        elif request.signal_type == SignalType.noisy:
            kwargs["noise_level"] = request.noise_level

        signal = generate_signal(
            signal_type=request.signal_type.value,
            freq=request.frequency,
            amplitude=request.amplitude,
            **kwargs,
        )

        predictor = get_predictor(request.model.value)
        result = predictor.predict(signal)

        return PredictionResponse(
            label=result["label"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            model_type=result["model_type"],
            signal_info={
                "type": request.signal_type.value,
                "frequency": request.frequency,
                "amplitude": request.amplitude,
                "samples": len(signal),
            },
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/array", response_model=PredictionResponse, tags=["prediction"])
async def predict_from_array(request: PredictFromArrayRequest):
    """Classify a signal from raw sample values."""
    if len(request.signal) != N_SAMPLES:
        raise HTTPException(
            status_code=422,
            detail=f"Signal must have exactly {N_SAMPLES} samples, got {len(request.signal)}",
        )

    try:
        signal = np.array(request.signal, dtype=np.float64)
        predictor = get_predictor(request.model.value)
        result = predictor.predict(signal)

        return PredictionResponse(
            label=result["label"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            model_type=result["model_type"],
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/upload", response_model=PredictionResponse, tags=["prediction"])
async def predict_from_upload(
    file: UploadFile = File(..., description="CSV file with signal data (first column)"),
    model: ModelType = ModelType.rf,
):
    """Upload a CSV file and classify the signal in the first column."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only CSV files are supported")

    try:
        contents = await file.read()
        data = pd.read_csv(io.BytesIO(contents))
        signal = data.iloc[:, 0].values

        if len(signal) != N_SAMPLES:
            raise HTTPException(
                status_code=422,
                detail=f"Signal must have {N_SAMPLES} samples, CSV has {len(signal)} rows",
            )

        predictor = get_predictor(model.value)
        result = predictor.predict(signal)

        return PredictionResponse(
            label=result["label"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            model_type=result["model_type"],
            signal_info={"source": file.filename, "samples": len(signal)},
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")
