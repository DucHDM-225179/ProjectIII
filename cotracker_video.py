import torch
import cv2
import numpy as np

# --- Configuration ---
VIDEO_PATH = 'input.mp4'
OUTPUT_PATH = 'output_cotracker.mp4'
GRID_STEP = 40          
REFILL_INTERVAL = 20    
STALE_THRESHOLD = 5.0   
VIS_THRESHOLD = 0.6
MIN_DIST = 15
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_grid_queries(h, w, step, mask=None):
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(float)
    if mask is not None:
        valid = mask[y.astype(int), x.astype(int)] == 255
        x, y = x[valid], y[valid]
    if len(x) == 0: return None
    points = np.column_stack((x, y))
    queries = np.zeros((1, len(points), 3), dtype=np.float32)
    queries[0, :, 1:] = points
    return torch.from_numpy(queries).to(DEVICE)

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): return

    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online").to(DEVICE)
    step = cotracker.step
    chunk_size = step * 2

    current_queries = get_grid_queries(h, w, GRID_STEP)
    anchor_points = current_queries[0, :, 1:].cpu().numpy()
    
    frames_buffer = []  # BGR frames
    tensor_buffer = []  # Tensors
    is_first_step = True
    global_frame_count = 0

    print(f"Tracking initialized with {current_queries.shape[1]} points.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames_buffer.append(frame)
        tensor_buffer.append(torch.from_numpy(frame).permute(2, 0, 1).float().to(DEVICE))

        # We process once we have a full chunk (step * 2)
        if len(tensor_buffer) == chunk_size:
            video_chunk = torch.stack(tensor_buffer, dim=0).unsqueeze(0)

            if is_first_step:
                cotracker(video_chunk=video_chunk, is_first_step=True, queries=current_queries)
                is_first_step = False
            
            # pred_tracks: (1, T, N, 2), pred_visibility: (1, T, N, 1)
            pred_tracks, pred_visibility = cotracker(video_chunk=video_chunk)
            pred_tracks = pred_tracks[:, -step:]
            pred_visibility = pred_visibility[:, -step:]

            # --- PRUNE & REFILL (Strictly at Interval) ---
            # Use the tracks at the end of the chunk for decision making
            last_tracks = pred_tracks[0, -1].cpu().numpy()
            last_vis = pred_visibility[0, -1].cpu().numpy().flatten()
            
            # --- OUTPUT FINALIZED FRAMES ---
            # We only write out the FIRST 'step' frames of the 'step * 2' window.
            # This is because the second half is still being used as context for the next step.
            for i in range(step):
                vis_frame = frames_buffer[i].copy()
                t_idx = i
                tracks_i = pred_tracks[0, t_idx].cpu().numpy()
                vis_i = pred_visibility[0, t_idx].cpu().numpy().flatten()

                for j in range(len(tracks_i)):
                    if vis_i[j] > VIS_THRESHOLD:
                        cv2.circle(vis_frame, (int(tracks_i[j, 0]), int(tracks_i[j, 1])), 3, (0, 0, 255), -1)
                
                out.write(vis_frame)
                cv2.imshow('Tracking', vis_frame)

            # Logic: Did we cross the 20-frame interval during this step?
            if (global_frame_count // REFILL_INTERVAL) < ((global_frame_count + step) // REFILL_INTERVAL):
                # Calculate movement from anchor
                dist = np.linalg.norm(last_tracks - anchor_points, axis=1)
                keep_mask = (last_vis > VIS_THRESHOLD) & (dist > STALE_THRESHOLD)
                active_tracks = last_tracks[keep_mask]
                
                # Refill gaps
                occ_mask = np.full((h, w), 255, dtype=np.uint8)
                for pt in active_tracks:
                    cv2.circle(occ_mask, (int(pt[0]), int(pt[1])), MIN_DIST, 0, -1)
                
                new_q = get_grid_queries(h, w, GRID_STEP, mask=occ_mask)
                if active_tracks.size > 0:
                    surv_q = np.zeros((1, len(active_tracks), 3), dtype=np.float32)
                    surv_q[0, :, 1:] = active_tracks
                    current_queries = torch.cat([torch.from_numpy(surv_q).to(DEVICE), new_q], dim=1) if new_q is not None else torch.from_numpy(surv_q).to(DEVICE)
                else:
                    current_queries = new_q

                anchor_points = current_queries[0, :, 1:].cpu().numpy() if current_queries is not None else np.empty((0,2))
                is_first_step = True # Trigger re-init on NEXT iteration
                print(f"Frame {global_frame_count}: Pruned/Refilled. New count: {current_queries.shape[1]}")

            # Advance sliding window by 'step'
            frames_buffer = frames_buffer[step:]
            tensor_buffer = tensor_buffer[step:]
            global_frame_count += step
            
            print(f"Frame: {global_frame_count:04d} | Points: {current_queries.shape[1]}")
            if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()