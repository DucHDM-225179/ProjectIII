import cv2
import numpy as np
import torch
import sys
import os
import shutil

# --- Constants ---
INPUT_VIDEO = 'input.mp4'
OUTPUT_PREFIX = 'grid_track'
GRID_STEP = 30
REFILL_INTERVAL = 30
STALE_THRESHOLD = 3.0
MIN_DIST = 15
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Add subdirectories to path
sys.path.append(os.path.join(os.getcwd(), 'tapir'))
sys.path.append(os.path.join(os.getcwd(), 'tapnext'))

# --- Point Manager Class ---
class PointManager:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.points = np.empty((0, 2), dtype=np.float32)
        self.anchors = np.empty((0, 2), dtype=np.float32)
        self.visibles = np.empty((0,), dtype=bool) # Track visibility state
        self.frame_idx = 0

    def get_grid_points(self, mask):
        h, w = mask.shape
        y, x = np.mgrid[GRID_STEP//2:h:GRID_STEP, GRID_STEP//2:w:GRID_STEP].reshape(2, -1).astype(int)
        free_points = mask[y, x] == 255
        return np.column_stack((x[free_points], y[free_points])).astype(np.float32)

    def initialize_if_empty(self):
        if len(self.points) == 0:
            mask = np.full((self.height, self.width), 255, dtype=np.uint8)
            self.points = self.get_grid_points(mask)
            self.anchors = self.points.copy()
            self.visibles = np.ones(len(self.points), dtype=bool)

    def update_positions(self, new_points, visibles=None):
        # Update positions for ALL points
        self.points = new_points
        if visibles is not None:
            self.visibles = visibles.astype(bool)
        else:
            self.visibles = np.ones(len(new_points), dtype=bool)

    def prune_and_refill(self):
        # 1. Prune out of bounds
        in_bounds = (self.points[:, 0] >= 0) & (self.points[:, 0] < self.width) & \
                    (self.points[:, 1] >= 0) & (self.points[:, 1] < self.height)
        
        # 2. Prune stale points (every interval) - AND invisible points
        keep_mask = in_bounds
        
        if self.frame_idx % REFILL_INTERVAL == 0 and self.frame_idx > 0:
            diff = np.linalg.norm(self.points - self.anchors, axis=1)
            stale_mask = diff > STALE_THRESHOLD
            
            # Keep if: In Bounds AND (Visible AND Moving)
            keep_mask = keep_mask & self.visibles & stale_mask
            
            # Apply pruning
            self.points = self.points[keep_mask]
            self.anchors = self.points.copy() # Update anchors
            self.visibles = self.visibles[keep_mask]
            
            # 3. Refill
            occ_mask = np.full((self.height, self.width), 255, dtype=np.uint8)
            for px, py in self.points:
                cv2.circle(occ_mask, (int(px), int(py)), MIN_DIST, 0, -1)
            
            new_pts = self.get_grid_points(occ_mask)
            if len(new_pts) > 0:
                self.points = np.vstack([self.points, new_pts]) if len(self.points) > 0 else new_pts
                self.anchors = np.vstack([self.anchors, new_pts]) if len(self.anchors) > 0 else new_pts
                # New points are visible
                new_vis = np.ones(len(new_pts), dtype=bool)
                self.visibles = np.concatenate([self.visibles, new_vis]) if len(self.visibles) > 0 else new_vis
        else:
            self.points = self.points[keep_mask]
            self.anchors = self.anchors[keep_mask]
            self.visibles = self.visibles[keep_mask]

        self.frame_idx += 1
        return self.points, keep_mask # Return keep_mask to sync external states if needed

    def draw(self, frame):
        vis = frame.copy()
        for i, (px, py) in enumerate(self.points):
            if self.visibles[i]:
                cv2.circle(vis, (int(px), int(py)), 3, (0, 0, 255), -1) # Red for visible
            else:
                cv2.circle(vis, (int(px), int(py)), 2, (255, 0, 0), -1) # Blue for occluded
        return vis

# --- Wrappers ---

def run_lucas(video_path, output_path):
    print(f"Running Lucas-Kanade...")
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    pm = PointManager(w, h)
    pm.initialize_if_empty()
    
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    out.write(pm.draw(prev_frame)) # Write first frame
    pm.frame_idx += 1

    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if len(pm.points) > 0:
            p0 = pm.points.reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
            good = st.flatten() == 1
            p1 = p1.reshape(-1, 2)
            
            # Update with good points, prune lost immediately
            pm.points = p1[good]
            pm.anchors = pm.anchors[good]
            pm.visibles = pm.visibles[good]
            
        frame2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        out.write(pm.draw(frame2))
        
        pm.prune_and_refill()
        prev_gray = gray

    cap.release()
    out.release()

def run_farneback(video_path, output_path):
    print(f"Running Farneback...")
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    pm = PointManager(w, h)
    pm.initialize_if_empty()

    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    out.write(pm.draw(prev_frame))
    pm.frame_idx += 1

    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        if len(pm.points) > 0:
            ix, iy = pm.points[:, 0].astype(int), pm.points[:, 1].astype(int)
            ix = np.clip(ix, 0, w - 1)
            iy = np.clip(iy, 0, h - 1)
            pm.points += flow[iy, ix]
        
        frame2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        out.write(pm.draw(frame2))
        
        pm.prune_and_refill()
        prev_gray = gray

    cap.release()
    out.release()

def run_raft(video_path, output_path):
    print(f"Running RAFT...")
    from torchvision.models.optical_flow import raft_small
    import torchvision.transforms.functional as F
    
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    model = raft_small(pretrained=True).to(DEVICE).eval()
    
    new_h, new_w = (h // 8) * 8, (w // 8) * 8
    
    def preprocess(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).permute(2, 0, 1).float()
        t = F.resize(t, (new_h, new_w))
        return ((t / 127.5) - 1.0).unsqueeze(0).to(DEVICE)

    pm = PointManager(w, h)
    pm.initialize_if_empty()

    ret, prev_frame = cap.read()
    prev_tensor = preprocess(prev_frame)
    out.write(pm.draw(prev_frame))
    pm.frame_idx += 1

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret: break
            curr_tensor = preprocess(frame)
            
            list_of_flows = model(prev_tensor, curr_tensor)
            flow = list_of_flows[-1][0].permute(1, 2, 0).cpu().numpy()
            flow = cv2.resize(flow, (w, h))
            flow[:, :, 0] *= (w / new_w)
            flow[:, :, 1] *= (h / new_h)

            if len(pm.points) > 0:
                ix, iy = pm.points[:, 0].astype(int), pm.points[:, 1].astype(int)
                ix = np.clip(ix, 0, w - 1)
                iy = np.clip(iy, 0, h - 1)
                pm.points += flow[iy, ix]

            pm.prune_and_refill()
            out.write(pm.draw(frame))
            prev_tensor = curr_tensor

    cap.release()
    out.release()
    del model
    torch.cuda.empty_cache()

def run_tapir(video_path, output_path):
    print(f"Running TAPIR...")
    from tapir import tapir_model
    import tree

    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
    model.load_state_dict(torch.load('causal_bootstapir_checkpoint.pt', map_location=DEVICE))
    model.to(DEVICE).eval()
    
    RESIZE_SIZE = (256, 256)
    wr, hr = RESIZE_SIZE[0] / w, RESIZE_SIZE[1] / h
    
    pm = PointManager(w, h)
    pm.initialize_if_empty()

    # Initial Setup
    ret, frame = cap.read()
    resized_frame = cv2.resize(frame, RESIZE_SIZE)
    # TAPIR Preprocessing: [0, 255] -> [-1, 1]
    torch_frame = torch.from_numpy(resized_frame).float().to(DEVICE)[None, None] / 255.0 * 2 - 1
    
    # Init query features
    query_features = model.get_feature_grids(torch_frame, is_training=False)
    
    # Format points for TAPIR (frame_idx=0, y, x)
    scaled_q_pts = np.zeros((len(pm.points), 3), dtype=np.float32)
    scaled_q_pts[:, 1] = pm.points[:, 1] * hr # y
    scaled_q_pts[:, 2] = pm.points[:, 0] * wr # x
    torch_q_pts = torch.from_numpy(scaled_q_pts).to(DEVICE)[None]

    query_features = model.get_query_features(torch_frame, is_training=False, query_points=torch_q_pts, feature_grids=query_features)
    causal_state = model.construct_initial_causal_state(len(pm.points), len(query_features.resolutions) - 1)
    causal_state = tree.map_structure(lambda x: x.to(DEVICE), causal_state)

    out.write(pm.draw(frame))
    pm.frame_idx += 1

    def filter_tree(t, mask):
         return tree.map_structure(lambda x: x[:, mask] if (torch.is_tensor(x) and x.ndim>1 and x.shape[1]==len(mask)) else x, t)

    def append_tree(t1, t2):
         return tree.map_structure(lambda x, y: torch.cat([x, y], dim=1) if (torch.is_tensor(x) and x.ndim>1 and x.shape[1]!=y.shape[1]) else x, t1, t2)

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            resized_frame = cv2.resize(frame, RESIZE_SIZE)
            torch_frame = torch.from_numpy(resized_frame).float().to(DEVICE)[None, None] / 255.0 * 2 - 1
            
            feature_grids = model.get_feature_grids(torch_frame, is_training=False)
            trajectories = model.estimate_trajectories(
                RESIZE_SIZE, is_training=False, feature_grids=feature_grids, query_features=query_features,
                query_points_in_video=None, query_chunk_size=64, causal_context=causal_state, get_causal_context=True
            )
            
            causal_state = trajectories["causal_context"]
            tracks = trajectories["tracks"][-1][0, :, 0, :].cpu().numpy()
            visibles = (trajectories["occlusion"][-1][0, :, 0] < 0).cpu().numpy() 

            # Update PM (Positions + Visibility)
            new_pts = tracks.copy()
            new_pts[:, 0] /= wr
            new_pts[:, 1] /= hr
            pm.update_positions(new_pts, visibles)
            
            # Prune/Refill logic (State Sync)
            # 1. Prune out of bounds
            in_bounds = (pm.points[:, 0] >= 0) & (pm.points[:, 0] < w) & \
                        (pm.points[:, 1] >= 0) & (pm.points[:, 1] < h)
            keep_mask = in_bounds
            
            # 2. Prune Stale & Invisible (at interval)
            if pm.frame_idx % REFILL_INTERVAL == 0:
                 diff = np.linalg.norm(pm.points - pm.anchors, axis=1)
                 stale_mask = diff > STALE_THRESHOLD
                 keep_mask = keep_mask & pm.visibles & stale_mask
                 
                 # Apply Pruning to PM
                 pm.points = pm.points[keep_mask]
                 pm.anchors = pm.points.copy()
                 pm.visibles = pm.visibles[keep_mask]
                 
                 # Sync TAPIR State
                 query_features = filter_tree(query_features, torch.from_numpy(keep_mask).to(DEVICE))
                 causal_state = filter_tree(causal_state, torch.from_numpy(keep_mask).to(DEVICE))
                 
                 # Refill
                 occ_mask = np.full((h, w), 255, dtype=np.uint8)
                 for px, py in pm.points: cv2.circle(occ_mask, (int(px), int(py)), MIN_DIST, 0, -1)
                 new_pts = pm.get_grid_points(occ_mask)
                 
                 if len(new_pts) > 0:
                     pm.points = np.vstack([pm.points, new_pts])
                     pm.anchors = np.vstack([pm.anchors, new_pts])
                     pm.visibles = np.concatenate([pm.visibles, np.ones(len(new_pts), dtype=bool)])
                     
                     # Init new points TAPIR
                     scaled_new = np.zeros((len(new_pts), 3), dtype=np.float32)
                     scaled_new[:, 1] = new_pts[:, 1] * hr
                     scaled_new[:, 2] = new_pts[:, 0] * wr
                     torch_new = torch.from_numpy(scaled_new).to(DEVICE)[None]
                     
                     new_feats = model.get_query_features(torch_frame, is_training=False, query_points=torch_new, feature_grids=feature_grids)
                     new_state = model.construct_initial_causal_state(len(new_pts), len(new_feats.resolutions)-1)
                     new_state = tree.map_structure(lambda x: x.to(DEVICE), new_state)
                     
                     query_features = append_tree(query_features, new_feats)
                     causal_state = append_tree(causal_state, new_state)
            else:
                 # Just prune out of bounds
                 pm.points = pm.points[keep_mask]
                 pm.anchors = pm.anchors[keep_mask]
                 pm.visibles = pm.visibles[keep_mask]
                 query_features = filter_tree(query_features, torch.from_numpy(keep_mask).to(DEVICE))
                 causal_state = filter_tree(causal_state, torch.from_numpy(keep_mask).to(DEVICE))

            pm.frame_idx += 1
            out.write(pm.draw(frame))
    
    cap.release()
    out.release()
    del model
    torch.cuda.empty_cache()

def run_cotracker(video_path, output_path, version='2'):
    print(f"Running CoTracker {version}...")
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    if version == '2':
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online").to(DEVICE)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(DEVICE)
    
    window_step = model.step
    chunk_size = window_step * 2
    
    pm = PointManager(w, h)
    pm.initialize_if_empty()

    def get_queries(points):
        q = np.zeros((1, len(points), 3), dtype=np.float32)
        q[0, :, 1:] = points 
        return torch.from_numpy(q).to(DEVICE)
    
    frames_buffer = []
    tensor_buffer = []
    current_queries = get_queries(pm.points)
    is_first_step = True
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames_buffer.append(frame)
        
        # Preprocessing:
        # CoTracker2 (v2 reference) uses BGR directly [0, 255]
        # CoTracker3 (v3 reference) uses RGB [0, 255]
        if version == '2':
            # v2 reference passes frame directly (BGR) to from_numpy
            tensor_buffer.append(torch.from_numpy(frame).permute(2,0,1).float().to(DEVICE))
        else:
            # v3 reference converts to RGB first
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_buffer.append(torch.from_numpy(frame_rgb).permute(2,0,1).float().to(DEVICE))
        
        if len(tensor_buffer) == chunk_size:
            video_chunk = torch.stack(tensor_buffer, dim=0).unsqueeze(0)
            
            if is_first_step:
                model(video_chunk=video_chunk, is_first_step=True, queries=current_queries)
                is_first_step = False
            
            pred_tracks, pred_vis = model(video_chunk=video_chunk)
            
            # COPYING REFERENCE LOGIC: Slice to the last 'step' frames
            # In online mode, the model processes a sliding window.
            # We output the results corresponding to the 'step' frames that are being finalized.
            pred_tracks = pred_tracks[:, -window_step:]
            pred_vis = pred_vis[:, -window_step:]
            
            # Process output frames
            for i in range(window_step):
                vis_frame = frames_buffer[i].copy()
                # Get tracks for this frame
                pts = pred_tracks[0, i].cpu().numpy()
                vis = pred_vis[0, i].cpu().numpy().flatten()
                
                # Update PM just for drawing this frame
                pm.update_positions(pts, vis > 0.5)
                out.write(pm.draw(vis_frame))
                pm.frame_idx += 1
            
            # Prune/Refill based on LAST processed frame
            last_tracks = pred_tracks[0, window_step-1].cpu().numpy()
            last_vis = pred_vis[0, window_step-1].cpu().numpy().flatten()
            
            # Sync PM to last state
            pm.points = last_tracks
            pm.visibles = last_vis > 0.5
            
            # Check interval
            # Note: pm.frame_idx is already incremented in loop.
            if (pm.frame_idx // REFILL_INTERVAL) != ((pm.frame_idx - window_step) // REFILL_INTERVAL):
                 # Manually trigger prune_and_refill on PM
                 pm.prune_and_refill()
                 # Get updated points
                 all_pts = pm.points
                 pm.anchors = all_pts.copy()
                 
                 current_queries = get_queries(all_pts)
                 is_first_step = True

            frames_buffer = frames_buffer[window_step:]
            tensor_buffer = tensor_buffer[window_step:]

    cap.release()
    out.release()
    del model
    torch.cuda.empty_cache()

def run_tapnext(video_path, output_path):
    print(f"Running TapNext...")
    from tapnext_torch import TAPNext
    from tapnext_torch_utils import restore_model_from_jax_checkpoint
    
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    MODEL_SIZE = (256, 256)
    model = TAPNext(image_size=MODEL_SIZE, device=DEVICE)
    model = restore_model_from_jax_checkpoint(model, 'bootstapnext_ckpt.npz')
    model.to(DEVICE) # Ensure all buffers/parameters are on device
    model.eval()
    
    sy, sx = MODEL_SIZE[0]/h, MODEL_SIZE[1]/w
    
    pm = PointManager(w, h)
    pm.initialize_if_empty()
    
    # Init queries [1, N, 3] (t, y, x) - TapNext uses y, x
    def get_queries(points):
        # points is (x, y)
        q = np.zeros((1, len(points), 3), dtype=np.float32)
        q[0, :, 1] = points[:, 1] * sy # y
        q[0, :, 2] = points[:, 0] * sx # x
        return torch.from_numpy(q).to(DEVICE)

    current_queries = get_queries(pm.points)
    tracking_state = None
    is_first_step = True
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # TapNext Preprocessing: Resize, [0, 255] -> [-1, 1]
        inp = cv2.resize(frame, (MODEL_SIZE[1], MODEL_SIZE[0]))
        inp = (torch.from_numpy(inp).float() / 127.5 - 1.0).to(DEVICE)[None, None]
        
        with torch.no_grad():
            if is_first_step:
                pred_tracks, _, vis_logits, tracking_state = model(video=inp, query_points=current_queries)
                is_first_step = False
            else:
                pred_tracks, _, vis_logits, tracking_state = model(video=inp, state=tracking_state)
        
        # Tracks: [1, 1, N, 2] (y, x)
        pts_model = pred_tracks[0, 0].cpu().numpy()
        vis = (vis_logits[0, 0, :, 0] > 0).cpu().numpy() # [N]
        
        # Convert back
        pts_orig = pts_model.copy()
        pts_orig[:, 0] /= sy # y
        pts_orig[:, 1] /= sx # x
        # swap to x,y
        pts_orig = pts_orig[:, ::-1]
        
        pm.update_positions(pts_orig, vis)
        
        # Prune/Refill
        prev_idx = pm.frame_idx
        _, keep_mask = pm.prune_and_refill()
        
        # More robust check:
        if (prev_idx % REFILL_INTERVAL == 0 and prev_idx > 0):
             # This corresponds to the interval where PM prunes/refills.
             # If points changed (they almost certainly did due to refill or pruning), reset.
             current_queries = get_queries(pm.points)
             is_first_step = True
             
        out.write(pm.draw(frame))

    cap.release()
    out.release()
    del model
    torch.cuda.empty_cache()

# --- Main Dispatcher ---
if __name__ == "__main__":
    if not os.path.exists(INPUT_VIDEO):
        print(f"Input video {INPUT_VIDEO} not found.")
        sys.exit(1)

    print(f"Starting Grid Tracking on {INPUT_VIDEO}...")
    print(f"Params: Step={GRID_STEP}, Refill={REFILL_INTERVAL}, Stale={STALE_THRESHOLD}")

    # Sequence of execution
    run_lucas(INPUT_VIDEO, f"{OUTPUT_PREFIX}_lucas.mp4")
    run_farneback(INPUT_VIDEO, f"{OUTPUT_PREFIX}_farneback.mp4")
    run_raft(INPUT_VIDEO, f"{OUTPUT_PREFIX}_raft.mp4")
    run_tapir(INPUT_VIDEO, f"{OUTPUT_PREFIX}_tapir.mp4")
    run_cotracker(INPUT_VIDEO, f"{OUTPUT_PREFIX}_cotracker.mp4", version='2')
    run_cotracker(INPUT_VIDEO, f"{OUTPUT_PREFIX}_cotracker3.mp4", version='3')
    run_tapnext(INPUT_VIDEO, f"{OUTPUT_PREFIX}_tapnext.mp4")

    print("All tracking completed.")