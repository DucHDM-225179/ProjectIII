import numpy as np
import cv2
import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_small
from torchvision.utils import flow_to_image
from collections import deque

# --- Configuration ---
VIDEO_PATH = 'input.mp4'
OUTPUT_PATH = 'output_raft_hsv.mp4'
M_GAP = 4        # Compare current frame with the one M_GAP frames ago
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 2.0

def preprocess(img_bgr, target_size):
    """Prepares a BGR frame for RAFT."""
    # Convert BGR to RGB and then to Tensor
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
    
    # Resize to divisible by 8 (required by RAFT)
    img_tensor = F.resize(img_tensor, target_size)
    
    # Normalize to [-1, 1]
    img_tensor = (img_tensor / 127.5) - 1.0
    return img_tensor.unsqueeze(0).to(DEVICE)

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open {VIDEO_PATH}")
        return

    # Metadata
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    # Ensure dimensions are divisible by 8 for RAFT
    new_h = (height // 8) * 8
    new_w = (width // 8) * 8
    target_size = (new_h, new_w)

    # Initialize Model (raft_small is recommended for CPU)
    model = raft_small(pretrained=True, progress=True).to(DEVICE)
    model.eval()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    # Buffer for M_GAP frames
    frame_buffer = deque(maxlen=M_GAP + 1)

    print(f"Processing with RAFT on {DEVICE}... M={M_GAP}")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # RAFT needs RGB, but we keep original BGR for visualization
            current_tensor = preprocess(frame, target_size)
            frame_buffer.append((current_tensor, frame))

            if len(frame_buffer) > M_GAP:
                # 1. Get frames from buffer
                old_tensor, _ = frame_buffer[0]
                curr_tensor, curr_raw = frame_buffer[-1]

                # 2. Inference
                # RAFT returns a list of flows from iterative updates; take the last one
                list_of_flows = model(old_tensor, curr_tensor)
                predicted_flow = list_of_flows[-1]
                mag = torch.linalg.vector_norm(predicted_flow, ord=2, dim=1, keepdim=True)
                predicted_flow = torch.where(mag < THRESHOLD, torch.zeros_like(predicted_flow), predicted_flow)
                
                # 3. Post-process Flow to Image
                # flow_to_image converts (1, 2, H, W) to (1, 3, H, W) RGB
                flow_rgb_tensor = flow_to_image(predicted_flow)
                
                # Convert back to CPU/Numpy for OpenCV
                flow_rgb = flow_rgb_tensor[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                
                # 4. Resize flow visualization back to original frame size
                flow_bgr = cv2.resize(cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR), (width, height))

                # 5. Overlay on original frame
                vis = cv2.addWeighted(curr_raw, 0.6, flow_bgr, 0.4, 0)
            else:
                vis = frame

            cv2.imshow('RAFT M-Frame Flow', vis)
            out.write(vis)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done! Saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()