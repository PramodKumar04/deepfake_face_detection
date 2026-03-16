"""
inference.py

Runs deepfake detection on a video using the DeepfakeDetector model.

How the model was trained:
  - Input: (B, T=16, C=3, H=224, W=224) — 16 face frames per clip
  - Output logits: (B, 2) — index 0 = REAL, index 1 = FAKE
  - softmax(output)[0][1] = P(FAKE)
  - Threshold: P(FAKE) > 0.5 → FAKE, else → REAL
"""

import os
import torch
import numpy as np
import cv2
from PIL import Image

from utils.frame_extractor import extract_frames
from utils.face_detector import detect_faces
from utils.preprocess import get_transform


NUM_FRAMES = 16   # The model was trained with exactly 16 frames per clip


def predict_video(video_path, model, device, fps=5, threshold=0.5):
    """
    Predict if a video is a deepfake.

    Args:
        video_path : Path to video file
        model      : DeepfakeDetector (loaded by model_loader.load_model)
        device     : 'cpu' or 'cuda'
        fps        : Frames per second to extract for face sampling
        threshold  : P(FAKE) > threshold → FAKE  (default 0.5)

    Returns:
        dict with keys: prediction, confidence, frames_used
    """
    print(f"Processing video: {video_path}")

    # 1. Extract raw frames from the video
    frames = extract_frames(video_path, fps)
    print(f"  Extracted {len(frames)} raw frames")

    transform = get_transform()

    # 2. Detect faces and collect face crops (224×224 RGB tensors)
    face_tensors = []
    for i, frame in enumerate(frames):
        faces = detect_faces(frame)
        if len(faces) == 0:
            continue

        x, y, w, h = faces[0]
        face = frame[y:y + h, x:x + w]
        if face.size == 0:
            continue

        face = cv2.resize(face, (224, 224))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = Image.fromarray(face)
        face_tensors.append(transform(face))   # C, H, W

    print(f"  Found faces in {len(face_tensors)} frames")

    if len(face_tensors) == 0:
        return {
            "prediction": "No face detected",
            "confidence": 0.0,
            "frames_used": 0
        }

    # 3. Build exactly NUM_FRAMES (16) face frames for the model
    #    - If we have fewer, repeat/tile to reach 16
    #    - If we have more, sample evenly across all faces
    n = len(face_tensors)
    if n >= NUM_FRAMES:
        # Evenly sample NUM_FRAMES from the detected faces
        indices = np.linspace(0, n - 1, NUM_FRAMES, dtype=int)
        clip = [face_tensors[i] for i in indices]
    else:
        # Tile: repeat faces until we have 16
        repeats = (NUM_FRAMES + n - 1) // n   # ceiling division
        tiled = (face_tensors * repeats)[:NUM_FRAMES]
        clip = tiled

    # Stack into (1, T=16, C=3, H=224, W=224)
    clip_tensor = torch.stack(clip, dim=0).unsqueeze(0).to(device)  # 1, 16, C, H, W

    # 4. Run the full DeepfakeDetector model
    with torch.no_grad():
        logits = model(clip_tensor)          # (1, 2)
        probs  = torch.softmax(logits, dim=1)
        p_fake = probs[0][1].item()          # index 1 = FAKE
        p_real = probs[0][0].item()          # index 0 = REAL

    print(f"  P(REAL) = {p_real:.4f}  |  P(FAKE) = {p_fake:.4f}  |  Threshold = {threshold}")

    if p_fake > threshold:
        prediction = "FAKE"
        confidence = round(p_fake, 4)
    else:
        prediction = "REAL"
        confidence = round(p_real, 4)

    print(f"  Final Prediction: {prediction}  Confidence: {confidence:.4f}")

    return {
        "prediction": prediction,
        "confidence": confidence,
        "frames_used": len(face_tensors)
    }
