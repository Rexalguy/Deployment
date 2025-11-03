import os
import torch
from typing import Dict

class Config:
    """Configuration settings for the Gait Analysis API"""
    
    # Model settings
    MODEL_PATH = r"C:\Users\user\Desktop\GaitLab\Deployment\models\gait_student_best_balanced.pth"
    NUM_CLASSES = 9
    FRAME_SIZE = 224
    NUM_FRAMES = 16
    
    # Class names mapping (update these based on your training)
    CLASS_NAMES = {
        0: "Normal",
        1: "KOA_Early", 
        2: "KOA_Mild",
        3: "KOA_Severe",
        4: "PD_Early",
        5: "PD_Mild",
        6: "PD_Severe",
        7: "Disabled_Assistive",
        8: "Disabled_NonAssistive"
    }
    
    # API settings
    MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv'}
    HOST = "0.0.0.0"
    PORT = 8000
    
    # Device settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def validate_setup(cls):
        """Validate that all required files exist"""
        if not os.path.exists(cls.MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {cls.MODEL_PATH}")
        
        print(f"✅ Configuration validated:")
        print(f"   - Model path: {cls.MODEL_PATH}")
        print(f"   - Device: {cls.DEVICE}")
        print(f"   - Classes: {len(cls.CLASS_NAMES)}")