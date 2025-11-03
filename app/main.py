from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import shutil
from typing import List

from config import Config
from model_loader import ModelLoader
from video_processor import VideoProcessor

# Initialize components
try:
    Config.validate_setup()
    model_loader = ModelLoader(Config.MODEL_PATH, Config.DEVICE, Config.CLASS_NAMES)
    video_processor = VideoProcessor(Config.FRAME_SIZE, Config.NUM_FRAMES)
    print("🎉 All components initialized successfully!")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    exit(1)

# Create FastAPI app
app = FastAPI(
    title="Gait Analysis API",
    description="AI-powered gait disorder classification from videos",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def cleanup_temp_file(file_path: str):
    """Clean up temporary files"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except:
        pass

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Gait Analysis API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loader.model is not None,
        "device": Config.DEVICE,
        "gpu_available": torch.cuda.is_available(),
        "host": Config.HOST
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model_path": Config.MODEL_PATH,
        "num_classes": len(Config.CLASS_NAMES),
        "classes": Config.CLASS_NAMES,
        "frame_size": Config.FRAME_SIZE,
        "num_frames": Config.NUM_FRAMES,
        "device": Config.DEVICE,
        "host": Config.HOST
    }

@app.post("/predict")
async def predict_gait(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...)
):
    """
    Analyze gait from uploaded video
    
    - **video**: Video file (mp4, mov, avi, mkv) up to 100MB
    """
    try:
        # Validate file type
        file_extension = os.path.splitext(video.filename)[1].lower()
        if file_extension not in Config.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {Config.ALLOWED_EXTENSIONS}"
            )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # Copy uploaded file to temporary location
            shutil.copyfileobj(video.file, temp_file)
            temp_path = temp_file.name
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_path)
        
        # Validate video file
        if not video_processor.validate_video_file(temp_path, Config.MAX_VIDEO_SIZE):
            raise HTTPException(
                status_code=400,
                detail="Invalid video file or file too large"
            )
        
        # Process video
        print(f"🔄 Processing video: {video.filename}")
        video_tensor = video_processor.preprocess_video(temp_path)
        
        # Run prediction
        print("🤖 Running model inference...")
        prediction_result = model_loader.predict(video_tensor)
        
        # Add file info to result
        prediction_result["filename"] = video.filename
        prediction_result["file_size"] = os.path.getsize(temp_path)
        
        print(f"✅ Prediction complete: {prediction_result['prediction']} "
              f"(confidence: {prediction_result['confidence']:.3f})")
        
        return prediction_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )

@app.exception_handler(500)
async def internal_exception_handler(request, exc):
    """Handle internal server errors"""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error during video processing"}
    )