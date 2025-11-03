import torch
import torch.nn as nn
from typing import Dict, List
import json

class GaitStudentModel(nn.Module):
    """
    Lightweight student model for deployment
    This should match your training model architecture
    """
    
    def __init__(self, num_classes=9, hidden_dim=256):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 7, 7))
        )
        
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        pooled_features = self.temporal_pool(features)
        spatial_features = pooled_features.squeeze(2)
        batch_size = spatial_features.size(0)
        flattened = spatial_features.view(batch_size, -1)
        logits = self.classifier(flattened)
        return logits

class ModelLoader:
    """Handles model loading and inference"""
    
    def __init__(self, model_path: str, device: str, class_names: Dict):
        self.model_path = model_path
        self.device = device
        self.class_names = class_names
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            # Initialize model architecture
            self.model = GaitStudentModel(num_classes=len(self.class_names))
            
            # Load trained weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded successfully from {self.model_path}")
            print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def predict(self, video_tensor: torch.Tensor) -> Dict:
        """
        Run inference on preprocessed video tensor
        Returns: prediction results with confidence scores
        """
        try:
            with torch.no_grad():
                # Move tensor to device
                video_tensor = video_tensor.to(self.device)
                
                # Get predictions
                outputs = self.model(video_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get top prediction
                confidence, predicted_idx = torch.max(probabilities, 1)
                predicted_class = self.class_names[predicted_idx.item()]
                
                # Get all class probabilities
                all_predictions = []
                for class_idx in range(len(self.class_names)):
                    all_predictions.append({
                        "class": self.class_names[class_idx],
                        "confidence": probabilities[0][class_idx].item()
                    })
                
                # Sort by confidence (descending)
                all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
                
                return {
                    "prediction": predicted_class,
                    "confidence": confidence.item(),
                    "all_predictions": all_predictions,
                    "predicted_class_idx": predicted_idx.item()
                }
                
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")