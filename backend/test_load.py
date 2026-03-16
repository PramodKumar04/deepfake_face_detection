import os
import sys
import torch
import time

print("Starting test...", flush=True)

try:
    print(f"Loading model... PyTorch version: {torch.__version__}", flush=True)
    start = time.time()
    state_dict = torch.load('deepfake_v4_best.pth', map_location='cpu', weights_only=False)
    print(f"Model loaded in {time.time()-start:.2f}s", flush=True)
    print("Keys length:", len(state_dict.keys()), flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)

print("Done.", flush=True)
