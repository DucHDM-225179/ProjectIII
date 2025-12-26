import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
tapnext_dir = os.path.join(script_dir, 'tapnext')
if tapnext_dir not in sys.path:
    sys.path.insert(0, tapnext_dir)

import torch
import cv2
import numpy as np
import tqdm
from tapnext_torch import TAPNext
from tapnext_torch_utils import restore_model_from_jax_checkpoint

# --- Configuration ---
VIDEO_PATH = 'input.mp4'
OUTPUT_PATH = 'output_tapnext.mp4'
GRID_STEP = 40          
REFILL_INTERVAL = 20    
STALE_THRESHOLD = 5.0   
MIN_DIST = 15
VIS_THRESHOLD = 0.5     

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_SIZE = (256, 256) # (H, W)

def get_grid_queries_orig(h_orig, w_orig, step, mask_orig=None):
    y, x = np.mgrid[step//2:h_orig:step, step//2:w_orig:step].reshape(2, -1).astype(float)
    if mask_orig is not None:
        valid = mask_orig[y.astype(int), x.astype(int)] == 255
        x, y = x[valid], y[valid]
    if len(x) == 0: return None
    return np.column_stack((y, x))

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): return

    orig_w, orig_h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (orig_w, orig_h))

    scale_x_to_orig = orig_w / MODEL_SIZE[1]
    scale_y_to_orig = orig_h / MODEL_SIZE[0]
    scale_x_to_model = MODEL_SIZE[1] / orig_w
    scale_y_to_model = MODEL_SIZE[0] / orig_h

    model = TAPNext(image_size=MODEL_SIZE, device=DEVICE)
    model = restore_model_from_jax_checkpoint(model, 'bootstapnext_ckpt.npz')
    model.eval()

    # Init queries in original resolution
    initial_points_orig_yx = get_grid_queries_orig(orig_h, orig_w, GRID_STEP)
    anchor_points_orig_yx = initial_points_orig_yx
    
    # Scale to model and format [t, y, x]
    points_model_yx = initial_points_orig_yx * np.array([scale_y_to_model, scale_x_to_model])
    q_data = np.zeros((1, len(points_model_yx), 3), dtype=np.float32)
    q_data[0, :, 1:] = points_model_yx
    current_queries_model = torch.from_numpy(q_data).to(DEVICE)

    tracking_state = None
    is_first_step = True
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_resized = cv2.resize(frame, (MODEL_SIZE[1], MODEL_SIZE[0]))
        # [B, T, H, W, C]
        input_tensor = (torch.from_numpy(frame_resized).float() / 127.5) - 1.0
        input_tensor = input_tensor.to(DEVICE).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            if is_first_step:
                pred_tracks, _, visible_logits, tracking_state = model(
                    video=input_tensor, 
                    query_points=current_queries_model
                )
                is_first_step = False
            else:
                pred_tracks, _, visible_logits, tracking_state = model(
                    video=input_tensor,
                    state=tracking_state
                )

        # Updated indexing: shape is (1, 1, N, 2)
        # tracks_curr_model contains [y, x] coordinates in 256x256 space
        tracks_curr_model = pred_tracks[0, 0, :].cpu().numpy() 
        vis_curr = (visible_logits[0, 0, :] > 0).cpu().numpy()

        vis_frame = frame.copy()
        for i in range(len(tracks_curr_model)):
            if vis_curr[i]:
                # Flipped indexing: tracks_curr_model[i, 0] is Y, tracks_curr_model[i, 1] is X
                draw_x = int(tracks_curr_model[i, 1] * scale_x_to_orig)
                draw_y = int(tracks_curr_model[i, 0] * scale_y_to_orig)
                cv2.circle(vis_frame, (draw_x, draw_y), 3, (0, 0, 255), -1) # Red
        
        out.write(vis_frame)
        cv2.imshow('TAPNext Tracking', vis_frame)
        frame_count += 1

        # Prune & Refill Logic
        if frame_count % REFILL_INTERVAL == 0:
            # Convert current model tracks [y, x] to original [y, x]
            curr_yx_orig = tracks_curr_model * np.array([scale_y_to_orig, scale_x_to_orig])
            
            dist = np.linalg.norm(curr_yx_orig - anchor_points_orig_yx, axis=1)
            keep_mask = vis_curr.flatten() & (dist.flatten() > STALE_THRESHOLD)
            survivors_orig_yx = curr_yx_orig[keep_mask]
            
            occ_mask_orig = np.full((orig_h, orig_w), 255, dtype=np.uint8)
            for pt_y, pt_x in survivors_orig_yx:
                px, py = int(np.clip(pt_x, 0, orig_w-1)), int(np.clip(pt_y, 0, orig_h-1))
                cv2.circle(occ_mask_orig, (px, py), MIN_DIST, 0, -1)
            
            new_points_orig_yx = get_grid_queries_orig(orig_h, orig_w, GRID_STEP, mask_orig=occ_mask_orig)
            
            if new_points_orig_yx is not None:
                all_points_orig_yx = np.vstack([survivors_orig_yx, new_points_orig_yx]) if survivors_orig_yx.size > 0 else new_points_orig_yx
            else:
                all_points_orig_yx = survivors_orig_yx if survivors_orig_yx.size > 0 else None

            if all_points_orig_yx is not None and len(all_points_orig_yx) > 0:
                anchor_points_orig_yx = all_points_orig_yx
                points_model_yx = all_points_orig_yx * np.array([scale_y_to_model, scale_x_to_model])

                q_data = np.zeros((1, len(points_model_yx), 3), dtype=np.float32)
                q_data[0, :, 1:] = points_model_yx
                current_queries_model = torch.from_numpy(q_data).to(DEVICE)
                is_first_step = True 
                print(f"Frame {frame_count}: Refilled. Points: {len(all_points_orig_yx)}")

        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()