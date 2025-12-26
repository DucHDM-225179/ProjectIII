import numpy as np
import cv2
import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_small
from collections import deque

# --- Configuration ---
VIDEO_PATH = 'input.mp4'
OUTPUT_PATH = 'output_raft.mp4'
M_FRAMES = 72
GRID_STEP = 20         
MIN_DIST = 12          
STALE_THRESHOLD = 0.5
PRUNE_INTERVAL = M_FRAMES // 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_sparse_points(mask, step):
    h, w = mask.shape
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    free_points = mask[y, x] == 255
    return np.column_stack((x[free_points], y[free_points])).astype(np.float32)

def preprocess(img_bgr, target_size):
    """RAFT-specific preprocessing: BGR -> RGB -> Float Tensor [-1, 1]"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
    img_tensor = F.resize(img_tensor, target_size)
    img_tensor = (img_tensor / 127.5) - 1.0
    return img_tensor.unsqueeze(0).to(DEVICE)

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): return

    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # RAFT requirement: size must be divisible by 8
    new_h, new_w = (height // 8) * 8, (width // 8) * 8
    target_size = (new_h, new_w)

    # Initialize model (raft_small is significantly faster for CPU)
    model = raft_small(pretrained=True).to(DEVICE)
    model.eval()

    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    ret, first_frame = cap.read()
    if not ret: return
    
    prev_tensor = preprocess(first_frame, target_size)
    
    # Initialize points and anchors
    points = get_sparse_points(np.full((height, width), 255, dtype=np.uint8), GRID_STEP)
    anchors = points.copy()
    frame_idx = 0

    print(f"Tracking with RAFT on {DEVICE}... (Press ESC to quit)")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            curr_tensor = preprocess(frame, target_size)
            
            # 1. Calculate Flow using RAFT
            list_of_flows = model(prev_tensor, curr_tensor)
            flow_tensor = list_of_flows[-1][0] # (2, H_small, W_small)
            
            # 2. Convert flow tensor to numpy and resize back to original resolution
            # We must scale the flow values proportionally when resizing!
            flow_np = flow_tensor.permute(1, 2, 0).cpu().numpy()
            flow_resized = cv2.resize(flow_np, (width, height))
            flow_resized[:, :, 0] *= (width / new_w)
            flow_resized[:, :, 1] *= (height / new_h)

            # 3. Update Points
            if len(points) > 0:
                ix, iy = points[:, 0].astype(int), points[:, 1].astype(int)
                ix = np.clip(ix, 0, width - 1)
                iy = np.clip(iy, 0, height - 1)
                
                # Apply the displacement from the RAFT flow map
                points += flow_resized[iy, ix]

                # Pruning: Keep points in bounds
                in_bounds = (points[:, 0] >= 0) & (points[:, 0] < width) & \
                            (points[:, 1] >= 0) & (points[:, 1] < height)
                points = points[in_bounds]
                anchors = anchors[in_bounds]

                # Pruning: Remove stale (non-moving) points
                if frame_idx % PRUNE_INTERVAL == 0:
                    diff = np.linalg.norm(points - anchors, axis=1)
                    moving = diff > STALE_THRESHOLD
                    points = points[moving]
                    anchors = points.copy()

            # 4. Refill Points
            if frame_idx % M_FRAMES == 0:
                occ_mask = np.full((height, width), 255, dtype=np.uint8)
                for px, py in points:
                    cv2.circle(occ_mask, (int(px), int(py)), MIN_DIST, 0, -1)
                
                gap_points = get_sparse_points(occ_mask, GRID_STEP)
                if len(gap_points) > 0:
                    points = np.vstack([points, gap_points]) if len(points) > 0 else gap_points
                    anchors = np.vstack([anchors, gap_points]) if len(anchors) > 0 else gap_points

            # 5. Visualization
            vis = frame.copy()
            for px, py in points:
                cv2.circle(vis, (int(px), int(py)), 2, (0, 0, 255), -1)

            cv2.imshow('RAFT Pruned Tracking', vis)
            out.write(vis)
            
            # Preparation for next frame
            prev_tensor = curr_tensor
            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()