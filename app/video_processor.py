import cv2
import torch
import numpy as np
from decord import VideoReader, cpu
import tempfile
import os
from typing import List

class VideoProcessor:
    """Process videos for model inference"""
    
    def __init__(self, frame_size=224, num_frames=16):
        self.frame_size = frame_size
        self.num_frames = num_frames
    
    def preprocess_video(self, video_path: str) -> torch.Tensor:
        """
        Preprocess video for model inference
        Returns: tensor of shape [1, 3, num_frames, frame_size, frame_size]
        """
        try:
            # Load video using decord
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            # Sample frames evenly throughout the video
            if total_frames <= self.num_frames:
                frame_indices = list(range(total_frames))
                # Repeat last frame if needed
                frame_indices += [frame_indices[-1]] * (self.num_frames - total_frames)
                frame_indices = frame_indices[:self.num_frames]
            else:
                frame_indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
            
            # Get frames
            frames = vr.get_batch(frame_indices).asnumpy()  # Shape: [T, H, W, C]
            
            # Resize frames if needed
            if frames.shape[1] != self.frame_size or frames.shape[2] != self.frame_size:
                resized_frames = []
                for frame in frames:
                    resized_frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                    resized_frames.append(resized_frame)
                frames = np.array(resized_frames)
            
            # Convert to torch tensor - [C, T, H, W] format
            frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
            
            # Normalize to [0, 1] if not already
            if frames.max() > 1.0:
                frames = frames / 255.0
            
            # Apply ImageNet normalization (standard for pre-trained models)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
            frames = (frames - mean) / std
            
            # Add batch dimension
            frames = frames.unsqueeze(0)
            
            return frames
            
        except Exception as e:
            raise Exception(f"Error processing video: {str(e)}")
    
    def validate_video_file(self, file_path: str, max_size: int) -> bool:
        """Validate video file before processing"""
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > max_size:
            return False
        
        # Check if we can read the video
        try:
            vr = VideoReader(file_path, ctx=cpu(0))
            _ = len(vr)
            return True
        except:
            return False