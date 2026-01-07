import cv2
import numpy as np
import torch
import sys
import os
import io
import json
import tarfile
from huggingface_hub import hf_hub_download

# Setup paths
sys.path.append(os.path.join(os.getcwd(), 'tapir'))
sys.path.append(os.path.join(os.getcwd(), 'tapnext'))

# --- Constants ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
EVAL_SIZE = (256, 256) # (H, W) for metrics
NUM_VIDEOS = 20
POINTS_PER_VIDEO = 250

# --- Metrics ---
def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
    get_trackwise_metrics: bool = False,
):
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)"""
    summing_axis = (2,) if get_trackwise_metrics else (1, 2)
    metrics = {}

    # query_points: [B, N, 3] (t, y, x)
    # gt_tracks: [B, N, T, 2] (x, y)
    
    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)
    if query_mode == 'first':
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == 'strided':
        query_frame_to_eval_frames = 1 - eye
    else:
        raise ValueError('Unknown query mode ' + query_mode)

    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    
    # [B, N, T]
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0

    # Occlusion accuracy
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=summing_axis,
    ) / np.sum(evaluation_points, axis=summing_axis)
    metrics['occlusion_accuracy'] = occ_acc

    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    
    for thresh in [1, 2, 4, 8, 16]:
        # True positives: within threshold AND both visible
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        # Frac within threshold (ignoring prediction visibility)
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=summing_axis,
        )
        count_visible_points = np.sum(
            visible & evaluation_points, axis=summing_axis
        )
        # Avoid div by zero
        frac_correct = np.divide(count_correct, count_visible_points, out=np.zeros_like(count_correct, dtype=float), where=count_visible_points!=0)
        metrics['pts_within_' + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        # Jaccard
        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=summing_axis
        )
        
        gt_positives = np.sum(visible & evaluation_points, axis=summing_axis)
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(
            false_positives & evaluation_points, axis=summing_axis
        )
        
        denom = gt_positives + false_positives
        jaccard = np.divide(true_positives, denom, out=np.zeros_like(true_positives, dtype=float), where=denom!=0)
        
        metrics['jaccard_' + str(thresh)] = jaccard
        all_jaccard.append(jaccard)

    metrics['average_jaccard'] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
    )
    metrics['average_pts_within_thresh'] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    )
    return metrics

# --- Data Loading ---
def get_data(num_videos=20, points_per_video=250):
    print(f"Downloading/Processing first {num_videos} samples from Hugging Face Hub...")
    data_list = []
    
    for i in range(num_videos):
        filename = f"{i:04d}.tar.gz"
        print(f"  [{i+1}/{num_videos}] Fetching {filename}...")
        
        try:
            path = hf_hub_download(repo_id="facebook/CoTracker3_Kubric", filename=filename, repo_type="dataset")
        except Exception as e:
            print(f"    Error downloading {filename}: {e}")
            continue
            
        try:
            with tarfile.open(path, "r:gz") as tar:
                members = tar.getmembers()
                
                # Find image files
                image_members = [m for m in members if m.name.lower().endswith(('.png', '.jpg', '.jpeg')) and '._' not in m.name]
                try:
                    image_members.sort(key=lambda x: int(os.path.splitext(os.path.basename(x.name))[0]))
                except:
                    image_members.sort(key=lambda x: x.name)
                
                # Find npy file
                npy_member = next((m for m in members if m.name.lower().endswith('.npy') and '._' not in m.name), None)
                
                if not image_members:
                    print(f"    No images found in {filename}.")
                    continue
                    
                # Load images
                frames = []
                for m in image_members:
                    f = tar.extractfile(m)
                    if f:
                        bytes_data = f.read()
                        nparr = np.frombuffer(bytes_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            frames.append(img)
                if not frames:
                     print(f"    Failed to decode images in {filename}.")
                     continue
                frames = np.stack(frames) # [T, H, W, 3] (BGR)
                H, W = frames.shape[1], frames.shape[2]

                # Load Tracks/Occ
                gt_tracks = None
                gt_occ = None
                
                if npy_member:
                    f = tar.extractfile(npy_member)
                    if f:
                        f_bytes = io.BytesIO(f.read())
                        data = np.load(f_bytes, allow_pickle=True)
                        if isinstance(data, np.ndarray) and data.dtype == object:
                            data = data.item()
                        
                        if isinstance(data, dict):
                            # data['coords'] shape: [N, T, 2] (x, y) or (y, x)? 
                            # Inspection: (x, y) based on width/height logic in Kubric dataset
                            # data['visibility'] shape: [N, T]. False=Visible, True=Occluded (based on logic)
                            
                            gt_tracks = data.get('coords')
                            gt_occ = data.get('visibility') # True if Occluded
                        else:
                            print(f"    .npy content is not a dict in {filename}")
                
                if gt_tracks is None or gt_occ is None:
                    print(f"    Could not find tracks/visibility in {filename}.")
                    continue

                # Filter valid tracks (visible at frame 0)
                # gt_occ is True if occluded. So visible if gt_occ == False
                visible_frame0 = np.where(gt_occ[:, 0] == False)[0]
                
                if len(visible_frame0) == 0:
                    print(f"    No visible tracks at frame 0 in {filename}.")
                    continue

                rng = np.random.RandomState(SEED + i) 
                if len(visible_frame0) > points_per_video:
                    indices = rng.choice(visible_frame0, points_per_video, replace=False)
                else:
                    indices = visible_frame0

                sampled_tracks = gt_tracks[indices]
                sampled_occ = gt_occ[indices]
                
                # Queries: (t, y, x)
                queries = np.zeros((len(sampled_tracks), 3), dtype=np.float32)
                queries[:, 0] = 0
                queries[:, 1] = sampled_tracks[:, 0, 1] # y
                queries[:, 2] = sampled_tracks[:, 0, 0] # x
                
                data_list.append({
                    'video': frames,
                    'gt_tracks': sampled_tracks, # [N, T, 2] (x, y)
                    'gt_occ': sampled_occ,       # [N, T] (True=Occluded)
                    'queries': queries           # [N, 3] (t, y, x)
                })
                
        except Exception as e:
            print(f"    Error processing tar content of {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return data_list

# --- Trackers ---

def eval_tapir(frames, queries):
    # frames: [T, H, W, 3] BGR
    # queries: [N, 3] (t, y, x)
    from tapir import tapir_model
    import tree

    T, H, W, _ = frames.shape
    model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
    model.load_state_dict(torch.load('causal_bootstapir_checkpoint.pt', map_location=DEVICE))
    model.to(DEVICE).eval()

    RESIZE_SIZE = (256, 256) # TAPIR typical resolution
    wr, hr = RESIZE_SIZE[1] / W, RESIZE_SIZE[0] / H

    # Resize all frames
    frames_resized = [cv2.resize(f, (RESIZE_SIZE[1], RESIZE_SIZE[0])) for f in frames]
    frames_resized = np.stack(frames_resized)
    
    # Initial setup
    frame0 = frames_resized[0]
    torch_frame0 = torch.from_numpy(frame0).float().to(DEVICE)[None, None] / 255.0 * 2 - 1
    
    # Scale queries to resized dimensions
    scaled_q = np.zeros_like(queries)
    scaled_q[:, 0] = queries[:, 0]
    scaled_q[:, 1] = queries[:, 1] * hr
    scaled_q[:, 2] = queries[:, 2] * wr
    torch_q = torch.from_numpy(scaled_q).to(DEVICE)[None] # [1, N, 3]

    feature_grids = model.get_feature_grids(torch_frame0, is_training=False)
    query_features = model.get_query_features(torch_frame0, is_training=False, query_points=torch_q, feature_grids=feature_grids)
    causal_state = model.construct_initial_causal_state(len(queries), len(query_features.resolutions) - 1)
    causal_state = tree.map_structure(lambda x: x.to(DEVICE), causal_state)

    all_tracks = []
    all_occ = []

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
            
            # [1, N, 2]
            tracks = trajectories["tracks"][-1][0, :, 0, :].cpu().numpy()
            
            # Set visibility
            occ_logits = trajectories["occlusion"][-1][0, :, 0]
            expd_logits = trajectories["expected_dist"][-1][0, :, 0]
            vis_probs = (1 - torch.sigmoid(occ_logits)) * (1 - torch.sigmoid(expd_logits))
            is_visible = (vis_probs > 0.5).cpu().numpy()
            is_occluded = ~is_visible
            
            # Scale back to original
            tracks[:, 0] /= wr # x
            tracks[:, 1] /= hr # y
            
            all_tracks.append(tracks)
            all_occ.append(is_occluded)

    del model
    torch.cuda.empty_cache()
    return np.stack(all_tracks, axis=1), np.stack(all_occ, axis=1) # [N, T, 2], [N, T]

def eval_cotracker(frames, queries, version='2'):
    T, H, W, _ = frames.shape
    
    if version == '2':
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online").to(DEVICE)
        window_step = model.step
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(DEVICE)
        window_step = model.model.window_len // 2

    # Prepare queries: [1, N, 3] (t, x, y)
    q_ct = np.zeros((1, len(queries), 3), dtype=np.float32)
    q_ct[0, :, 0] = queries[:, 0]
    q_ct[0, :, 1] = queries[:, 2] # x
    q_ct[0, :, 2] = queries[:, 1] # y
    torch_q = torch.from_numpy(q_ct).to(DEVICE)
    
    # Prepare video: [1, T, 3, H, W]
    video_frames = [torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().to(DEVICE) for f in frames]
    video = torch.stack(video_frames, dim=0).unsqueeze(0)

    model(video_chunk=video, is_first_step=True, queries=torch_q, add_support_grid=False)
    
    pred_tracks = None
    pred_vis = None
    
    for ind in range(0, T - window_step, window_step):
        # evaluator.py: video_chunk=sample.video[:, ind : ind + online_model.step * 2]
        chunk = video[:, ind : ind + window_step * 2]
        pred_tracks, pred_vis = model(
            video_chunk=chunk, 
            is_first_step=False, 
            grid_size=0, 
            add_support_grid=False
        )
        
    if pred_tracks is not None:
        # CoTrackerOnlinePredictor returns [B, T, N, D]
        out_tracks = pred_tracks[0].cpu().numpy()
        out_occ = (pred_vis[0].cpu().numpy() < 0.5)
    else:
        out_tracks = np.zeros((T, len(queries), 2))
        out_occ = np.zeros((T, len(queries)), dtype=bool)

    if out_tracks.shape[0] > T:
        out_tracks = out_tracks[:T]
        out_occ = out_occ[:T]
             
    out_tracks = out_tracks.transpose(1, 0, 2)
    out_occ = out_occ.transpose(1, 0)
    
    del model
    torch.cuda.empty_cache()
    return out_tracks, out_occ

def eval_tapnext(frames, queries):
    from tapnext_torch import TAPNext
    from tapnext_torch_utils import restore_model_from_jax_checkpoint

    T, H, W, _ = frames.shape
    MODEL_SIZE = (256, 256)
    model = TAPNext(image_size=MODEL_SIZE, device=DEVICE)
    model = restore_model_from_jax_checkpoint(model, 'bootstapnext_ckpt.npz')
    model.to(DEVICE).eval()

    sy, sx = MODEL_SIZE[0]/H, MODEL_SIZE[1]/W
    
    # Queries: [1, N, 3] (t, y, x)
    q_tn = np.zeros((1, len(queries), 3), dtype=np.float32)
    q_tn[0, :, 0] = queries[:, 0]
    q_tn[0, :, 1] = queries[:, 1] * sy
    q_tn[0, :, 2] = queries[:, 2] * sx
    torch_q = torch.from_numpy(q_tn).to(DEVICE)
    
    tracking_state = None
    is_first_step = True
    
    all_tracks = []
    all_occ = []
    
    with torch.no_grad():
        for t in range(T):
            frame = frames[t]
            inp = cv2.resize(frame, (MODEL_SIZE[1], MODEL_SIZE[0]))
            inp = (torch.from_numpy(inp).float() / 127.5 - 1.0).to(DEVICE)[None, None]
            
            if is_first_step:
                pred_tracks, _, vis_logits, tracking_state = model(video=inp, query_points=torch_q)
                is_first_step = False
            else:
                pred_tracks, _, vis_logits, tracking_state = model(video=inp, state=tracking_state)
            
            # Tracks: [1, 1, N, 2] (y, x)
            pts_model = pred_tracks[0, 0].cpu().numpy()
            vis_probs = torch.sigmoid(vis_logits[0, 0, :, 0]).cpu().numpy()
            vis = vis_probs > 0.5
            
            # Convert back
            pts_orig = pts_model.copy()
            pts_orig[:, 0] /= sy # y
            pts_orig[:, 1] /= sx # x
            # swap to x, y
            pts_orig = pts_orig[:, ::-1]
            
            all_tracks.append(pts_orig)
            all_occ.append(~vis)
            
    del model
    torch.cuda.empty_cache()
    return np.stack(all_tracks, axis=1), np.stack(all_occ, axis=1)

# --- Main ---
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Load Data
    data = get_data(num_videos=NUM_VIDEOS, points_per_video=POINTS_PER_VIDEO)
    print(f"Loaded {len(data)} videos.")
    if len(data) == 0:
        print("No videos loaded. Exiting.")
        return
    
    # Define methods
    methods = ['TAPIR', 'CoTracker2', 'CoTracker3', 'TAPNext']
    
    result_files = []

    for method_name in methods:
        print(f"\n=== Evaluating {method_name} ===")
        method_results = []
        
        for i, item in enumerate(data):
            frames = item['video'] # BGR
            queries = item['queries']
            gt_tracks = item['gt_tracks'] 
            gt_occ = item['gt_occ'] 
            
            H, W = frames.shape[1], frames.shape[2]
            print(f"  Processing video {i+1}/{len(data)} ({W}x{H})...")
            
            if method_name == 'TAPIR':
                frames_in = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames])
                p_tracks, p_occ = eval_tapir(frames_in, queries)
            elif method_name == 'CoTracker2':
                p_tracks, p_occ = eval_cotracker(frames, queries, version='2')
            elif method_name == 'CoTracker3':
                p_tracks, p_occ = eval_cotracker(frames, queries, version='3')
            elif method_name == 'TAPNext':
                frames_in = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames])
                p_tracks, p_occ = eval_tapnext(frames_in, queries)
            
            # Scale everything to EVAL_SIZE (256x256) for metrics
            sx, sy = EVAL_SIZE[1] / W, EVAL_SIZE[0] / H
            
            gt_tracks_eval = gt_tracks.copy()
            gt_tracks_eval[..., 0] *= sx
            gt_tracks_eval[..., 1] *= sy
            
            p_tracks_eval = p_tracks.copy()
            p_tracks_eval[..., 0] *= sx
            p_tracks_eval[..., 1] *= sy
            
            # Prepare for metrics
            gt_tracks_b = gt_tracks_eval[None]
            gt_occ_b = gt_occ[None]
            p_tracks_b = p_tracks_eval[None]
            p_occ_b = p_occ[None]
            q_b = queries[None].copy()
            # queries are (t, y, x). Not scaled for determining 'query_frame', but technically y,x not used.
            # Passing them anyway.

            m = compute_tapvid_metrics(q_b, gt_occ_b, gt_tracks_b, p_occ_b, p_tracks_b, query_mode='first')
            
            m_flat = {k: float(np.mean(v)) for k, v in m.items()}
            m_flat['video_idx'] = i
            method_results.append(m_flat)

        print(f"\n--- {method_name} Summary ---")
        keys = ['average_jaccard', 'average_pts_within_thresh', 'occlusion_accuracy']
        summary = {}
        for k in keys:
            if method_results:
                avg = np.mean([r[k] for r in method_results])
            else:
                avg = 0.0
            summary[k] = avg
            print(f"  {k}: {avg*100:.2f}") 
        
        fname = f"results_{method_name}.json"
        with open(fname, 'w') as f:
            json.dump({'summary': summary, 'videos': method_results}, f, indent=2)
        result_files.append(fname)

    tar_name = "evaluation_results.tar.gz"
    with tarfile.open(tar_name, "w:gz") as tar:
        for f in result_files:
            tar.add(f)
    print(f"\nAll detailed results compressed to {tar_name}")

if __name__ == "__main__":
    main()
