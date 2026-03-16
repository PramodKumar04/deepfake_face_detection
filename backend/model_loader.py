"""
model_loader.py

Loads the DeepfakeDetector model (deepfake_v4_best.pth) which was trained in
the Colab notebook deepfake_face_detection.ipynb.

Architecture details (from notebook):
  - SpatialCNN: timm EfficientNet-B4 backbone, feature_dim=1792
  - FrequencyBranch: FFT-based GAN artifact detector, out_dim=64
  - rPPGBranch: Green-channel heartbeat signal analyzer, out_dim=64
  - TemporalBiLSTM: BiLSTM with attention, input=1792, hidden=512, out_dim=512
  - DeepfakeDetector: Fuses all branches (640-dim) → FC(256) → FC(2)
    - Label 0 = REAL, Label 1 = FAKE
  - Input: (B, T=16, C=3, H=224, W=224)
  - Checkpoint format: dict with key 'model_state_dict'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os


# ─── Sub-modules (must match training code exactly) ───────────────────────────

class SpatialCNN(nn.Module):
    """EfficientNet-B4, fully unfrozen.  Outputs (B, 1792) feature vectors."""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=False)
        self.feature_dim = self.backbone.classifier.in_features  # 1792
        self.backbone.classifier  = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):          # x: B, C, H, W
        feat = self.backbone.forward_features(x)   # B, 1792, H', W'
        return self.pool(feat).squeeze(-1).squeeze(-1)  # B, 1792


class FrequencyBranch(nn.Module):
    """FFT-based GAN artifact detector."""
    def __init__(self, out_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(4),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4)
        )
        self.fc = nn.Linear(32 * 4 * 4, out_dim)

    def forward(self, x):   # x: B*T, C, H, W
        with torch.no_grad():
            f   = torch.fft.fftshift(torch.fft.fft2(x.float(), norm='ortho'))
            mag = torch.log1p(f.abs())
        mag = mag / (mag.flatten(2).max(dim=2).values.unsqueeze(-1).unsqueeze(-1) + 1e-6)
        return self.fc(self.conv(mag.to(x.dtype)).flatten(1))


class rPPGBranch(nn.Module):
    """Green-channel heartbeat signal analyzer."""
    def __init__(self, out_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.attn = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, signal):   # B, T
        x = self.conv(signal.unsqueeze(1))    # B, 64, T
        x = x.permute(0, 2, 1)               # B, T, 64
        x, _ = self.attn(x, x, x)
        return self.pool(x.permute(0, 2, 1)).squeeze(-1)  # B, 64


class TemporalBiLSTM(nn.Module):
    """BiLSTM with attention pooling over T frames."""
    def __init__(self, input_size=1792, hidden=512, num_layers=2, out_dim=512):
        super().__init__()
        self.bilstm   = nn.LSTM(input_size, hidden, num_layers,
                                 batch_first=True, bidirectional=True, dropout=0.3)
        self.attn     = nn.Linear(hidden * 2, 1)
        self.out_proj = nn.Linear(hidden * 2, out_dim)
        self.norm     = nn.LayerNorm(out_dim)

    def forward(self, x):   # B, T, F
        out, _  = self.bilstm(x)                         # B, T, hidden*2
        weights = torch.softmax(self.attn(out), dim=1)   # B, T, 1
        context = (out * weights).sum(dim=1)              # B, hidden*2
        return self.norm(self.out_proj(context))          # B, out_dim


class DeepfakeDetector(nn.Module):
    """
    Full multi-branch DeepfakeDetector.

    Input  : (B, T, C, H, W)  — B video clips, each T=16 frames at 224×224
    Output : (B, 2)            — logits for [REAL, FAKE]
    """
    def __init__(self):
        super().__init__()
        self.spatial  = SpatialCNN()
        feat_dim      = self.spatial.feature_dim   # 1792

        self.temporal = TemporalBiLSTM(feat_dim, hidden=512, num_layers=2, out_dim=512)
        self.rppg     = rPPGBranch(out_dim=64)
        self.freq     = FrequencyBranch(out_dim=64)

        fusion_dim = 512 + 64 + 64   # 640

        # Classifier with residual
        self.fc1    = nn.Linear(fusion_dim, 256)
        self.fc2    = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 2)
        self.drop1  = nn.Dropout(0.4)
        self.drop2  = nn.Dropout(0.3)
        self.norm1  = nn.LayerNorm(fusion_dim)
        self.norm2  = nn.LayerNorm(256)

    def extract_rppg(self, frames):   # B, T, C, H, W
        sig = frames[:, :, 1, :, :].mean(dim=(2, 3))   # B, T
        return (sig - sig.mean(1, keepdim=True)) / (sig.std(1, keepdim=True) + 1e-6)

    def forward(self, frames):   # B, T, C, H, W
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)

        # Spatial + Temporal
        sp   = self.spatial(flat).view(B, T, -1)   # B, T, 1792
        temp = self.temporal(sp)                    # B, 512

        # rPPG
        rppg = self.rppg(self.extract_rppg(frames))  # B, 64

        # Frequency
        freq = self.freq(flat).view(B, T, -1).mean(1)  # B, 64

        # Fusion with residual classifier
        fused = self.norm1(torch.cat([temp, rppg, freq], dim=1))  # B, 640
        x     = F.gelu(self.fc1(fused))                           # B, 256
        x     = self.drop1(x)
        x     = self.norm2(x + F.gelu(self.fc2(x)))               # residual
        x     = self.drop2(x)
        return self.fc_out(x)                                      # B, 2


# ─── Public loader ─────────────────────────────────────────────────────────────

def load_model(model_path: str, device: str):
    print(f"Loading DeepfakeDetector from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = DeepfakeDetector().to(device)

    # The checkpoint is a dict (saved by: torch.save(ckpt, path) where ckpt has 'model_state_dict')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch      = checkpoint.get('epoch', '?')
        val_auc    = checkpoint.get('val_auc', '?')
        val_acc    = checkpoint.get('val_acc', '?')
        print(f"  Checkpoint from epoch {epoch} | AUC={val_auc} | Acc={val_acc}")
    else:
        # Fallback: maybe it was saved as a plain state_dict
        state_dict = checkpoint
        print("  Warning: checkpoint has no 'model_state_dict' key — treating as raw state_dict")

    model.load_state_dict(state_dict)
    model.eval()
    print("✓ DeepfakeDetector loaded successfully.")
    print("  Architecture: EfficientNet-B4 + BiLSTM + FrequencyBranch + rPPGBranch")
    print("  Labels: index 0 = REAL, index 1 = FAKE")
    return model
