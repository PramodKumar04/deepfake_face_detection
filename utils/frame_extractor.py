import cv2
import numpy as np


def extract_frames(video_path, fps=5):
    """
    Extract frames from a video at a specified frame rate.
    
    Args:
        video_path (str): Path to the video file
        fps (int): Number of frames to extract per second
        
    Returns:
        list: List of frame arrays (BGR format)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if video_fps == 0:
        raise ValueError("Could not determine video FPS")
    
    # Calculate frame interval
    frame_interval = int(video_fps / fps)
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Extract frame at specified intervals
        if frame_count % frame_interval == 0:
            frames.append(frame)
        
        frame_count += 1
    
    cap.release()
    
    return frames
