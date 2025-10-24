"""
FastAPI Backend for Respiratory Disease Detection
===============================================

This module provides REST API endpoints for audio processing, feature extraction,
and disease prediction. It serves as the backend for the Streamlit frontend and
can be used for external integrations.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np
import librosa
import tempfile
import os
import json
import uuid
from datetime import datetime
import logging

# Import custom modules
from src.preprocessing.audio_preprocessing import AudioPreprocessor
from src.preprocessing.feature_extraction import FeatureExtractor
from src.models.ensemble_model import EnsembleModel
from src.utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Respiratory Disease Detection API",
    description="REST API for analyzing lung sounds and detecting respiratory diseases",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
config = Config()
preprocessor = AudioPreprocessor()
feature_extractor = FeatureExtractor()
model = None

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for audio prediction."""
    apply_noise_reduction: bool = True
    normalize_audio: bool = True
    extract_cycles: bool = True
    model_type: str = "ensemble"

class PredictionResponse(BaseModel):
    """Response model for prediction results."""
    prediction_id: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time: float
    audio_duration: float
    quality_metrics: Dict[str, float]

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    model_loaded: bool
    version: str

class AudioInfo(BaseModel):
    """Response model for audio information."""
    duration: float
    sample_rate: int
    channels: int
    format: str
    size_bytes: int

@app.on_event("startup")
async def startup_event():
    """Initialize the model and components on startup."""
    global model
    
    try:
        model = EnsembleModel()
        model_path = "data/models/trained/ensemble_model.h5"
        
        if os.path.exists(model_path):
            model.load_model(model_path)
            logger.info("✅ Model loaded successfully")
        else:
            logger.warning("⚠️ Pre-trained model not found")
            
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Respiratory Disease Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        version="1.0.0"
    )

@app.get("/models", response_class=JSONResponse)
async def list_models():
    """List available models and their information."""
    models_info = {
        "ensemble": {
            "name": "Ensemble Model",
            "description": "Combined CNN, ResNet, and traditional ML features",
            "accuracy": "94.16%",
            "classes": ["Normal", "Wheeze", "Crackle", "Pneumonia", "COPD", "Asthma", "Bronchiectasis", "Other"],
            "loaded": model is not None
        },
        "cnn": {
            "name": "CNN Model",
            "description": "Convolutional Neural Network for spectrogram analysis",
            "accuracy": "89.2%",
            "classes": ["Normal", "Wheeze", "Crackle", "Pneumonia", "COPD", "Asthma", "Bronchiectasis", "Other"],
            "loaded": False
        },
        "resnet": {
            "name": "ResNet Model", 
            "description": "Residual Network adapted for respiratory sounds",
            "accuracy": "91.8%",
            "classes": ["Normal", "Wheeze", "Crackle", "Pneumonia", "COPD", "Asthma", "Bronchiectasis", "Other"],
            "loaded": False
        }
    }
    
    return {"models": models_info}

@app.post("/analyze_audio", response_model=PredictionResponse)
async def analyze_audio(
    file: UploadFile = File(...),
    request: PredictionRequest = Depends()
):
    """
    Analyze uploaded audio file and predict respiratory disease.
    
    Args:
        file: Audio file (WAV, MP3, FLAC, M4A)
        request: Processing parameters
        
    Returns:
        Prediction results with probabilities and metadata
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    allowed_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_extension}. Allowed: {allowed_extensions}"
        )
    
    start_time = datetime.now()
    prediction_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()
            
            # Load audio
            audio, sr = librosa.load(tmp_file.name, sr=16000)
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
        
        # Preprocess audio
        processed_result = preprocessor.preprocess_pipeline(
            audio, 
            sr,
            apply_noise_reduction=request.apply_noise_reduction,
            normalize=request.normalize_audio,
            segment_cycles=request.extract_cycles
        )
        
        processed_audio = processed_result['processed_audio']
        quality_metrics = processed_result['quality_metrics']
        
        # Extract features
        if request.extract_cycles and processed_result['respiratory_cycles']:
            features_result = feature_extractor.extract_features_from_cycles(
                processed_result['respiratory_cycles']
            )
        else:
            features_result = feature_extractor.extract_comprehensive_features(processed_audio)
        
        features = features_result['features']
        
        # Make prediction
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)
        
        # Define class names
        class_names = [
            "Normal", "Wheeze", "Crackle", "Pneumonia", 
            "COPD", "Asthma", "Bronchiectasis", "Other"
        ]
        
        predicted_class = class_names[prediction[0]]
        confidence = float(np.max(probabilities))
        
        # Create probability dictionary
        prob_dict = {
            class_names[i]: float(prob) 
            for i, prob in enumerate(probabilities[0])
        }
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PredictionResponse(
            prediction_id=prediction_id,
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=prob_dict,
            processing_time=processing_time,
            audio_duration=len(audio) / sr,
            quality_metrics=quality_metrics
        )
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/extract_features", response_class=JSONResponse)
async def extract_features(file: UploadFile = File(...)):
    """
    Extract features from audio file without prediction.
    
    Args:
        file: Audio file
        
    Returns:
        Extracted features and metadata
    """
    # Validate file type
    allowed_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_extension}"
        )
    
    try:
        # Save and load audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()
            
            audio, sr = librosa.load(tmp_file.name, sr=16000)
            os.unlink(tmp_file.name)
        
        # Preprocess
        processed_result = preprocessor.preprocess_pipeline(audio, sr)
        processed_audio = processed_result['processed_audio']
        
        # Extract features
        features_result = feature_extractor.extract_comprehensive_features(processed_audio)
        
        return {
            "features": features_result['features'].tolist(),
            "feature_names": features_result['feature_names'],
            "n_features": features_result['n_features'],
            "audio_duration": len(audio) / sr,
            "sample_rate": sr,
            "quality_metrics": processed_result['quality_metrics']
        }
        
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting features: {str(e)}")

@app.post("/preprocess_audio", response_class=JSONResponse)
async def preprocess_audio(
    file: UploadFile = File(...),
    apply_noise_reduction: bool = True,
    normalize: bool = True,
    segment_cycles: bool = False
):
    """
    Preprocess audio file and return processing information.
    
    Args:
        file: Audio file
        apply_noise_reduction: Whether to apply noise reduction
        normalize: Whether to normalize audio
        segment_cycles: Whether to segment respiratory cycles
        
    Returns:
        Preprocessing results and quality metrics
    """
    try:
        # Load audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()
            
            audio, sr = librosa.load(tmp_file.name, sr=16000)
            os.unlink(tmp_file.name)
        
        # Preprocess
        result = preprocessor.preprocess_pipeline(
            audio, 
            sr,
            apply_noise_reduction=apply_noise_reduction,
            normalize=normalize,
            segment_cycles=segment_cycles
        )
        
        return {
            "original_duration": len(result['original_audio']) / sr,
            "processed_duration": len(result['processed_audio']) / sr,
            "sample_rate": result['sample_rate'],
            "respiratory_cycles_detected": len(result['respiratory_cycles']),
            "quality_metrics": result['quality_metrics'],
            "preprocessing_applied": result['preprocessing_applied']
        }
        
    except Exception as e:
        logger.error(f"Error preprocessing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error preprocessing audio: {str(e)}")

@app.get("/audio_info/{file_id}", response_class=JSONResponse)
async def get_audio_info(file_id: str):
    """
    Get information about a previously processed audio file.
    
    Args:
        file_id: Unique file identifier
        
    Returns:
        Audio file information
    """
    # This would typically look up file info from a database
    # For now, return a placeholder response
    return {
        "message": f"Audio info for file {file_id}",
        "note": "File tracking not implemented in this demo"
    }

@app.get("/supported_formats", response_class=JSONResponse)
async def get_supported_formats():
    """Get list of supported audio formats."""
    return {
        "supported_formats": [
            {
                "extension": ".wav",
                "description": "Waveform Audio File Format",
                "recommended": True
            },
            {
                "extension": ".mp3", 
                "description": "MPEG Audio Layer III",
                "recommended": False,
                "note": "Lossy compression may affect analysis quality"
            },
            {
                "extension": ".flac",
                "description": "Free Lossless Audio Codec", 
                "recommended": True
            },
            {
                "extension": ".m4a",
                "description": "MPEG-4 Audio",
                "recommended": False
            }
        ],
        "recommended_specs": {
            "sample_rate": "16kHz or higher",
            "bit_depth": "16-bit minimum",
            "duration": "5-30 seconds",
            "channels": "Mono preferred"
        }
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI application
    uvicorn.run(
        "fastapi_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )