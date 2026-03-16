import torch

try:
    state_dict = torch.load('deepfake_v4_best.pth', map_location='cpu')
    keys = list(state_dict.keys())
    print("Total keys:", len(keys))
    print("Last 10 keys:")
    for k in keys[-10:]:
        print(f"  {k}: {state_dict[k].shape}")
except Exception as e:
    print(f"Error: {e}")
