import cv2
import numpy as np
import torch
import sys
import os
import imageio
from matplotlib import cm
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision.models.optical_flow import raft_large
import torchvision.transforms.functional as TorchF

# --- Setup Paths ---
sys.path.append(os.path.join(os.getcwd(), 'tapir'))
sys.path.append(os.path.join(os.getcwd(), 'tapnext'))

# --- Constants ---
INPUT_VIDEO = 'video/input.mp4'
GRID_STEP = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Visualizer Class ---

def draw_circle(rgb, coord, radius, color=(255, 0, 0), visible=True, color_alpha=None):
    draw = ImageDraw.Draw(rgb)
    left_up_point = (coord[0] - radius, coord[1] - radius)
    right_down_point = (coord[0] + radius, coord[1] + radius)
    color = tuple(list(color) + [color_alpha if color_alpha is not None else 255])
    draw.ellipse(
        [left_up_point, right_down_point],
        fill=tuple(color) if visible else None,
        outline=tuple(color),
    )
    return rgb

def draw_line(rgb, coord_y, coord_x, color, linewidth):
    draw = ImageDraw.Draw(rgb)
    draw.line(
        (coord_y[0], coord_y[1], coord_x[0], coord_x[1]),
        fill=tuple(color),
        width=linewidth,
    )
    return rgb

def add_weighted(rgb, alpha, original, beta, gamma):
    return (rgb * alpha + original * beta + gamma).astype("uint8")

class Visualizer:
    def __init__(
        self,
        save_dir: str = "./",
        grayscale: bool = False,
        pad_value: int = 0,
        fps: int = 10,
        mode: str = "rainbow",
        linewidth: int = 1,
        show_first_frame: int = 0,
        tracks_leave_trace: int = -1,
    ):
        self.mode = mode
        self.save_dir = save_dir
        if mode == "rainbow":
            self.color_map = plt.get_cmap("gist_rainbow")
        elif mode == "cool":
            self.color_map = plt.get_cmap(mode)
        self.show_first_frame = show_first_frame
        self.grayscale = grayscale
        self.tracks_leave_trace = tracks_leave_trace
        self.pad_value = pad_value
        self.linewidth = linewidth
        self.fps = fps

    def visualize(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor = None,
        filename: str = "video",
        alpha: float = 1.0,
    ):
        color_alpha = int(alpha * 255)
        
        if self.grayscale:
            B, T, C, H, W = video.shape
            vid_reshaped = video.view(-1, C, H, W)
            transform = transforms.Grayscale(num_output_channels=3)
            vid_gray = transform(vid_reshaped)
            video = vid_gray.view(B, T, 3, H, W)

        res_video = self.draw_tracks_on_video(
            video=video,
            tracks=tracks,
            visibility=visibility,
            color_alpha=color_alpha,
        )
        self.save_video(res_video, filename=filename)

    def save_video(self, video, filename):
        vid_np = video[0].permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        save_path = os.path.join(self.save_dir, f"{filename}.mp4")
        imageio.mimsave(save_path, vid_np, fps=self.fps)
        print(f"Saved visualization to {save_path}")

    def draw_tracks_on_video(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor = None,
        color_alpha: int = 255,
    ):
        B, T, C, H, W = video.shape
        _, _, N, D = tracks.shape

        video = video[0].permute(0, 2, 3, 1).byte().detach().cpu().numpy()
        tracks = tracks[0].long().detach().cpu().numpy()
        
        if visibility is not None:
             visibility = visibility[0].detach().cpu().numpy()

        res_video = []
        for rgb in video:
            res_video.append(rgb.copy())
            
        vector_colors = np.zeros((T, N, 3))
        
        y_min, y_max = tracks[0, :, 1].min(), tracks[0, :, 1].max()
        norm = plt.Normalize(y_min, y_max)
        
        for n in range(N):
            color = self.color_map(norm(tracks[0, n, 1]))
            color = np.array(color[:3])[None] * 255
            vector_colors[:, n] = np.repeat(color, T, axis=0)

        if self.tracks_leave_trace != 0:
            for t in range(1, T):
                rgb_pil = Image.fromarray(res_video[t])
                
                first_ind = 0
                if self.tracks_leave_trace > 0:
                    first_ind = max(0, t - self.tracks_leave_trace)
                
                for s in range(first_ind, t):
                    for i in range(N):
                        if visibility is not None and not visibility[s, i]: continue
                        
                        p1 = (int(tracks[s, i, 0]), int(tracks[s, i, 1]))
                        p2 = (int(tracks[s+1, i, 0]), int(tracks[s+1, i, 1]))
                        
                        if (0 <= p1[0] < W and 0 <= p1[1] < H) or (0 <= p2[0] < W and 0 <= p2[1] < H):
                             rgb_pil = draw_line(rgb_pil, p1, p2, vector_colors[s, i].astype(int), self.linewidth)
                
                res_video[t] = np.array(rgb_pil)

        for t in range(T):
            rgb_pil = Image.fromarray(res_video[t])
            for i in range(N):
                vis = True
                if visibility is not None:
                    vis = bool(visibility[t, i])
                
                if vis:
                    coord = (tracks[t, i, 0], tracks[t, i, 1])
                    if 0 <= coord[0] < W and 0 <= coord[1] < H:
                         rgb_pil = draw_circle(
                            rgb_pil, 
                            coord, 
                            int(self.linewidth * 2), 
                            vector_colors[t, i].astype(int), 
                            visible=True, 
                            color_alpha=color_alpha
                        )
            res_video[t] = np.array(rgb_pil)

        return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte()

# --- Shared Logic ---

def get_grid_points(H, W, step):
    y, x = np.mgrid[step//2:H:step, step//2:W:step].reshape(2, -1).astype(int)
    return np.column_stack((x, y)).astype(np.float32)

def load_video_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    return np.stack(frames)

# --- Trackers ---

def run_lucas_tracking(frames, points):
    print("Tracking with Lucas-Kanade...")
    T, H, W, _ = frames.shape
    N = len(points)
    
    tracks = np.zeros((T, N, 2), dtype=np.float32)
    visibles = np.zeros((T, N), dtype=bool)
    
    tracks[0] = points
    visibles[0] = True
    
    curr_points = points.copy()
    curr_status = np.ones(N, dtype=bool)
    
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    for t in range(1, T):
        gray = cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY)
        
        p0 = curr_points.reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
        
        p1 = p1.reshape(-1, 2)
        st = st.flatten() == 1
        
        curr_status = curr_status & st
        curr_points = p1
        
        tracks[t] = curr_points
        visibles[t] = curr_status
        
        prev_gray = gray
        
    return tracks, None

def run_farneback_tracking(frames, points):
    print("Tracking with Farneback...")
    T, H, W, _ = frames.shape
    N = len(points)
    
    tracks = np.zeros((T, N, 2), dtype=np.float32)
    visibles = np.ones((T, N), dtype=bool)
    
    tracks[0] = points
    curr_points = points.copy()
    
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    for t in range(1, T):
        gray = cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        ix = np.clip(curr_points[:, 0].astype(int), 0, W - 1)
        iy = np.clip(curr_points[:, 1].astype(int), 0, H - 1)
        
        delta = flow[iy, ix]
        curr_points += delta
        
        tracks[t] = curr_points
        
        in_bounds = (curr_points[:, 0] >= 0) & (curr_points[:, 0] < W) & \
                    (curr_points[:, 1] >= 0) & (curr_points[:, 1] < H)
        visibles[t] = in_bounds
        
        prev_gray = gray

    return tracks, None

def run_raft_tracking(frames, points):
    print("Tracking with RAFT...")
    T, H, W, _ = frames.shape
    N = len(points)
    
    tracks = np.zeros((T, N, 2), dtype=np.float32)
    visibles = np.ones((T, N), dtype=bool)
    
    tracks[0] = points
    curr_points = points.copy()
    
    model = raft_large(pretrained=True).to(DEVICE).eval()
    new_h, new_w = (H // 8) * 8, (W // 8) * 8
    
    def preprocess(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).permute(2, 0, 1).float()
        t = TorchF.resize(t, (new_h, new_w))
        return ((t / 127.5) - 1.0).unsqueeze(0).to(DEVICE)
    
    prev_tensor = preprocess(frames[0])
    
    with torch.no_grad():
        for t in range(1, T):
            curr_tensor = preprocess(frames[t])
            
            list_of_flows = model(prev_tensor, curr_tensor)
            flow = list_of_flows[-1][0].permute(1, 2, 0).cpu().numpy()
            
            flow = cv2.resize(flow, (W, H))
            flow[:, :, 0] *= (W / new_w)
            flow[:, :, 1] *= (H / new_h)
            
            ix = np.clip(curr_points[:, 0].astype(int), 0, W - 1)
            iy = np.clip(curr_points[:, 1].astype(int), 0, H - 1)
            
            delta = flow[iy, ix]
            curr_points += delta
            
            tracks[t] = curr_points
            
            in_bounds = (curr_points[:, 0] >= 0) & (curr_points[:, 0] < W) & \
                        (curr_points[:, 1] >= 0) & (curr_points[:, 1] < H)
            visibles[t] = in_bounds
            
            prev_tensor = curr_tensor
            
    del model
    torch.cuda.empty_cache()
    return tracks, None

def run_tapir_tracking(frames, points):
    print("Tracking with TAPIR...")
    from tapir import tapir_model
    import tree
    
    T, H, W, _ = frames.shape
    model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
    model.load_state_dict(torch.load('causal_bootstapir_checkpoint.pt', map_location=DEVICE))
    model.to(DEVICE).eval()
    
    RESIZE_SIZE = (256, 256)
    wr, hr = RESIZE_SIZE[1] / W, RESIZE_SIZE[0] / H
    
    frames_resized = [cv2.resize(f, (RESIZE_SIZE[1], RESIZE_SIZE[0])) for f in frames]
    
    frame0 = frames_resized[0]
    torch_frame0 = torch.from_numpy(frame0).float().to(DEVICE)[None, None] / 255.0 * 2 - 1
    
    scaled_q = np.zeros((len(points), 3), dtype=np.float32)
    scaled_q[:, 0] = 0
    scaled_q[:, 1] = points[:, 1] * hr
    scaled_q[:, 2] = points[:, 0] * wr
    torch_q = torch.from_numpy(scaled_q).to(DEVICE)[None]
    
    feature_grids = model.get_feature_grids(torch_frame0, is_training=False)
    query_features = model.get_query_features(torch_frame0, is_training=False, query_points=torch_q, feature_grids=feature_grids)
    causal_state = model.construct_initial_causal_state(len(points), len(query_features.resolutions) - 1)
    causal_state = tree.map_structure(lambda x: x.to(DEVICE), causal_state)
    
    all_tracks = []
    all_vis = []
    
    with torch.no_grad():
        for t in range(T):
            frame = frames_resized[t]
            torch_frame = torch.from_numpy(frame).float().to(DEVICE)[None, None] / 255.0 * 2 - 1
            
            feature_grids = model.get_feature_grids(torch_frame, is_training=False)
            trajectories = model.estimate_trajectories(
                RESIZE_SIZE, is_training=False, feature_grids=feature_grids, query_features=query_features,
                query_points_in_video=None, query_chunk_size=64, causal_context=causal_state, get_causal_context=True
            )
            causal_state = trajectories["causal_context"]
            
            tracks = trajectories["tracks"][-1][0, :, 0, :].cpu().numpy()
            
            occ_logits = trajectories["occlusion"][-1][0, :, 0]
            expd_logits = trajectories["expected_dist"][-1][0, :, 0]
            vis_probs = (1 - torch.sigmoid(occ_logits)) * (1 - torch.sigmoid(expd_logits))
            is_visible = (vis_probs > 0.5).cpu().numpy()
            
            tracks[:, 0] /= wr
            tracks[:, 1] /= hr
            
            all_tracks.append(tracks)
            all_vis.append(is_visible)
            
    del model
    torch.cuda.empty_cache()
    return np.stack(all_tracks), np.stack(all_vis)

def run_cotracker_tracking(frames, points, version='2'):
    print(f"Tracking with CoTracker{version}...")
    T, H, W, _ = frames.shape
    
    if version == '2':
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online").to(DEVICE)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(DEVICE)
    
    q = np.zeros((1, len(points), 3), dtype=np.float32)
    q[0, :, 0] = 0
    q[0, :, 1] = points[:, 0]
    q[0, :, 2] = points[:, 1]
    torch_q = torch.from_numpy(q).to(DEVICE)
    
    video_frames = [torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().to(DEVICE) for f in frames]
    video = torch.stack(video_frames, dim=0).unsqueeze(0)
    
    window_step = model.step if version == '2' else model.model.window_len // 2
    
    model(video_chunk=video, is_first_step=True, queries=torch_q, add_support_grid=False)
    
    pred_tracks = None
    pred_vis = None
    
    for ind in range(0, T - window_step, window_step):
        chunk = video[:, ind : ind + window_step * 2]
        if chunk.shape[1] < window_step * 2:
            pass

        pred_tracks, pred_vis = model(
            video_chunk=chunk, 
            is_first_step=False, 
            grid_size=0, 
            add_support_grid=False
        )
    
    if pred_tracks is not None:
        out_tracks = pred_tracks[0].cpu().numpy()
        out_vis = (pred_vis[0].cpu().numpy() > 0.5)
        
        if out_tracks.shape[0] > T:
            out_tracks = out_tracks[:T]
            out_vis = out_vis[:T]
        elif out_tracks.shape[0] < T:
            diff = T - out_tracks.shape[0]
            out_tracks = np.pad(out_tracks, ((0, diff), (0,0), (0,0)), mode='edge')
            out_vis = np.pad(out_vis, ((0, diff), (0,0)), mode='constant', constant_values=False)
            
        return out_tracks, out_vis
    else:
        return np.zeros((T, len(points), 2)), np.zeros((T, len(points)), dtype=bool)

def run_tapnext_tracking(frames, points):
    print("Tracking with TAPNext...")
    from tapnext_torch import TAPNext
    from tapnext_torch_utils import restore_model_from_jax_checkpoint
    
    T, H, W, _ = frames.shape
    MODEL_SIZE = (256, 256)
    
    model = TAPNext(image_size=MODEL_SIZE, device=DEVICE)
    model = restore_model_from_jax_checkpoint(model, 'bootstapnext_ckpt.npz')
    model.to(DEVICE).eval()
    
    sy, sx = MODEL_SIZE[0]/H, MODEL_SIZE[1]/W
    
    q_tn = np.zeros((1, len(points), 3), dtype=np.float32)
    q_tn[0, :, 0] = 0
    q_tn[0, :, 1] = points[:, 1] * sy
    q_tn[0, :, 2] = points[:, 0] * sx
    torch_q = torch.from_numpy(q_tn).to(DEVICE)
    
    tracking_state = None
    is_first_step = True
    
    all_tracks = []
    all_vis = []
    
    with torch.no_grad():
        for t in range(T):
            frame = frames[t]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            inp = cv2.resize(frame_rgb, (MODEL_SIZE[1], MODEL_SIZE[0]))
            inp = (torch.from_numpy(inp).float() / 127.5 - 1.0).to(DEVICE)[None, None]
            
            if is_first_step:
                pred_tracks, _, vis_logits, tracking_state = model(video=inp, query_points=torch_q)
                is_first_step = False
            else:
                pred_tracks, _, vis_logits, tracking_state = model(video=inp, state=tracking_state)
            
            pts_model = pred_tracks[0, 0].cpu().numpy()
            vis_probs = torch.sigmoid(vis_logits[0, 0, :, 0]).cpu().numpy()
            vis = vis_probs > 0.5
            
            pts_orig = pts_model.copy()
            pts_orig[:, 0] /= sy
            pts_orig[:, 1] /= sx
            pts_orig = pts_orig[:, ::-1]
            
            all_tracks.append(pts_orig)
            all_vis.append(vis)
            
    del model
    torch.cuda.empty_cache()
    return np.stack(all_tracks), np.stack(all_vis)


# --- Main ---

def main():
    if not os.path.exists(INPUT_VIDEO):
        print(f"Error: {INPUT_VIDEO} not found.")
        return

    print(f"Loading {INPUT_VIDEO}...")
    frames = load_video_frames(INPUT_VIDEO)
    T, H, W, _ = frames.shape
    print(f"Video loaded: {T} frames, {W}x{H}")

    points = get_grid_points(H, W, GRID_STEP)
    print(f"Initialized {len(points)} grid points.")
    
    vid_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
    vid_tensor = vid_tensor[:, [2, 1, 0], :, :]
    vid_tensor = vid_tensor.unsqueeze(0).float()
    
    algos = {
        'lucas': run_lucas_tracking,
        'farneback': run_farneback_tracking,
        'raft': run_raft_tracking,
        'tapir': run_tapir_tracking,
        'cotracker': lambda f, p: run_cotracker_tracking(f, p, version='2'),
        'cotracker3': lambda f, p: run_cotracker_tracking(f, p, version='3'),
        'tapnext': run_tapnext_tracking
    }

    for name, func in algos.items():
        print(f"\n--- Running {name} ---")
        try:
            tracks, vis = func(frames, points)
            
            tracks_t = torch.from_numpy(tracks).unsqueeze(0).float()
            
            vis_t = torch.from_numpy(vis).unsqueeze(0) if vis is not None else None
            is_grayscale = (name in ['lucas', 'farneback'])

            viz = Visualizer(
                save_dir='./video', 
                grayscale=is_grayscale, 
                mode='rainbow', 
                fps=24, 
                tracks_leave_trace=4
            )
            
            viz.visualize(
                video=vid_tensor, 
                tracks=tracks_t, 
                visibility=vis_t, 
                filename=f"output_visualize_{name}"
            )
            
        except Exception as e:
            print(f"Failed to run {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\nAll Done.")

if __name__ == "__main__":
    main()