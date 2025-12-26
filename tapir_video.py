import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
tapir_dir = os.path.join(script_dir, 'tapir')
if tapir_dir not in sys.path:
    sys.path.insert(0, tapir_dir)

import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tree
from tapir import tapir_model

# --- Configuration ---
VIDEO_PATH = 'input.mp4'
OUTPUT_PATH = 'output_tapir.mp4'
CHECKPOINT_PATH = 'causal_bootstapir_checkpoint.pt'

NUM_POINTS_MAX = 128*100000  # Maximum capacity for TAPIR tracking
GRID_STEP = 30         
MIN_DIST = 20          
STALE_THRESHOLD = 2
PRUNE_INTERVAL = 10
REFILL_INTERVAL = 30
RESIZE_SIZE = (256, 256)

def preprocess_frames(frames):
    """Expects [B, T, H, W, 3], returns [-1, 1] normalized."""
    return (frames.float() / 255.0) * 2 - 1

def postprocess_occlusions(occlusions, expected_dist):
    visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
    return visibles

def get_sparse_points(mask, step):
    h, w = mask.shape
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    free_points = mask[y, x] == 255
    return np.column_stack((x[free_points], y[free_points])).astype(np.float32)

def filter_state(state, mask, device):
    """
    Slices only the tensors that have a point dimension matching the mask length.
    """
    # Convert mask to torch tensor if it isn't already
    if not torch.is_tensor(mask):
        mask = torch.from_numpy(mask).to(device)

    def _leaf_filter(x):
        if torch.is_tensor(x):
            # Point-based tensors in TAPIR typically have the point dim at index 1
            # e.g., [batch, num_points, channels] or [batch, num_points, height, width]
            if x.ndim > 1 and x.shape[1] == len(mask):
                return x[:, mask]
            return x
        return x

    return tree.map_structure(_leaf_filter, state)

def append_state(old_state, new_state):
    """
    Concatenates two states along the point dimension (index 1).
    """
    def _leaf_append(x, y):
        if torch.is_tensor(x) and torch.is_tensor(y):
            if x.ndim > 1 and x.shape[1] != y.shape[1]: # Verify it's a point-dim tensor
                 return torch.cat([x, y], dim=1)
            return x # Keep original for non-point tensors (like feature grids)
        return x
    
    return tree.map_structure(_leaf_append, old_state, new_state)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device).eval()
    torch.set_grad_enabled(False)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): return
    
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (orig_w, orig_h))

    # Calculate scaling ratios
    # TAPIR coordinates are normalized to the model input size
    wr = RESIZE_SIZE[0] / orig_w
    hr = RESIZE_SIZE[1] / orig_h

    ret, frame = cap.read()
    if not ret: return
    
    # Initial points in ORIGINAL resolution
    points = get_sparse_points(np.full((orig_h, orig_w), 255, dtype=np.uint8), GRID_STEP)
    points = points[:NUM_POINTS_MAX]
    anchors = points.copy()
    
    # Scale points to 256x256 for model initialization
    # TAPIR query points format: (frame_idx, y, x)
    scaled_q_pts = np.zeros((len(points), 3), dtype=np.float32)
    scaled_q_pts[:, 1] = points[:, 1] * hr
    scaled_q_pts[:, 2] = points[:, 0] * wr
    
    # Resize frame for model
    resized_frame = cv2.resize(frame, RESIZE_SIZE, interpolation=cv2.INTER_LINEAR)
    torch_frame = torch.from_numpy(resized_frame).to(device)[None, None]
    torch_query_pts = torch.from_numpy(scaled_q_pts).to(device)[None]
    
    query_features = model.get_feature_grids(preprocess_frames(torch_frame), is_training=False)
    query_features = model.get_query_features(
        preprocess_frames(torch_frame), 
        is_training=False, 
        query_points=torch_query_pts, 
        feature_grids=query_features
    )
    
    causal_state = model.construct_initial_causal_state(len(points), len(query_features.resolutions) - 1)
    causal_state = tree.map_structure(lambda x: x.to(device), causal_state)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Prepare 256x256 frame
        resized_frame = cv2.resize(frame, RESIZE_SIZE, interpolation=cv2.INTER_LINEAR)
        torch_frame = torch.from_numpy(resized_frame).to(device)[None, None]
        
        if len(points) > 0:
            feature_grids = model.get_feature_grids(preprocess_frames(torch_frame), is_training=False)
            trajectories = model.estimate_trajectories(
                RESIZE_SIZE, 
                is_training=False,
                feature_grids=feature_grids,
                query_features=query_features,
                query_points_in_video=None,
                query_chunk_size=64,
                causal_context=causal_state,
                get_causal_context=True,
            )
            
            causal_state = trajectories["causal_context"]
            tracks = trajectories["tracks"][-1]  
            visibles = postprocess_occlusions(trajectories["occlusion"][-1], trajectories["expected_dist"][-1])
            
            # 1. Get current predictions
            new_points_scaled = tracks[0, :, 0, :].cpu().numpy() 
            is_visible = visibles[0, :, 0].cpu().numpy()
            
            # 2. Map back to ORIGINAL resolution
            new_points = new_points_scaled.copy()
            new_points[:, 0] = new_points_scaled[:, 0] / wr 
            new_points[:, 1] = new_points_scaled[:, 1] / hr 
            
            # 3. Create the base keep_mask
            in_bounds = (new_points[:, 0] >= 0) & (new_points[:, 0] < orig_w) & \
                        (new_points[:, 1] >= 0) & (new_points[:, 1] < orig_h)
            
            keep_mask = is_visible & in_bounds
            
            # 4. Apply Pruning (Stationary point removal) if interval reached
            if frame_idx % PRUNE_INTERVAL == 0:
                # Compare new_points (current) vs anchors (last saved positions)
                diff = np.linalg.norm(new_points - anchors, axis=1)
                moving = diff > STALE_THRESHOLD
                keep_mask = keep_mask & moving
                # We update anchors for the NEXT interval here
                anchors = new_points.copy()

            points = new_points[keep_mask]
            anchors = anchors[keep_mask] # This now matches because anchors was updated/kept in sync
            query_features = filter_state(query_features, keep_mask, device)
            causal_state = filter_state(causal_state, keep_mask, device)

        # Refill Logic
        if frame_idx % REFILL_INTERVAL == 0 and len(points) < NUM_POINTS_MAX:
            occ_mask = np.full((orig_h, orig_w), 255, dtype=np.uint8)
            for px, py in points:
                cv2.circle(occ_mask, (int(px), int(py)), MIN_DIST, 0, -1)
            
            gap_points = get_sparse_points(occ_mask, GRID_STEP)
            if len(gap_points) > 0:
                num_to_add = min(len(gap_points), NUM_POINTS_MAX - len(points))
                new_seeds = gap_points[:num_to_add]
                
                # Scale new points for model
                new_q_scaled = np.zeros((len(new_seeds), 3), dtype=np.float32)
                new_q_scaled[:, 1] = new_seeds[:, 1] * hr
                new_q_scaled[:, 2] = new_seeds[:, 0] * wr
                
                torch_new_pts = torch.from_numpy(new_q_scaled).to(device)[None]
                new_feats = model.get_query_features(
                    preprocess_frames(torch_frame), 
                    is_training=False, 
                    query_points=torch_new_pts, 
                    feature_grids=model.get_feature_grids(preprocess_frames(torch_frame), is_training=False)
                )
                
                new_causal_state = model.construct_initial_causal_state(len(new_seeds), len(new_feats.resolutions) - 1)
                new_causal_state = tree.map_structure(lambda x: x.to(device), new_causal_state)
                
                points = np.vstack([points, new_seeds]) if len(points) > 0 else new_seeds
                anchors = np.vstack([anchors, new_seeds]) if len(anchors) > 0 else new_seeds
                query_features = append_state(query_features, new_feats)
                causal_state = append_state(causal_state, new_causal_state)

        # Visualizing on ORIGINAL resolution
        vis = frame.copy()
        for px, py in points:
            # We use float points here; cv2.circle will truncate for drawing, 
            # but 'points' array maintains sub-pixel precision for the next loop.
            cv2.circle(vis, (int(round(px)), int(round(py))), 3, (0, 0, 255), -1)

        cv2.imshow('TAPIR 256x256 (Scaled Back)', vis)
        out.write(vis)
        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()