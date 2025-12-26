import numpy as np
import cv2

# --- Configuration ---
VIDEO_PATH = 'input.mp4'
OUTPUT_PATH = 'output_farneback.mp4'
M_FRAMES = 72
GRID_STEP = 20         
MIN_DIST = 12          
STALE_THRESHOLD = 0.5
PRUNE_INTERVAL = M_FRAMES // 2

def get_sparse_points(mask, step):
    h, w = mask.shape
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    free_points = mask[y, x] == 255
    return np.column_stack((x[free_points], y[free_points])).astype(np.float32)

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): return

    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    ret, first_frame = cap.read()
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    # Points array and an anchor array to track displacement
    points = get_sparse_points(np.full((height, width), 255, dtype=np.uint8), GRID_STEP)
    anchors = points.copy() # Store position from PRUNE_INTERVAL ago
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        if len(points) > 0:
            ix, iy = points[:, 0].astype(int), points[:, 1].astype(int)
            ix = np.clip(ix, 0, width - 1)
            iy = np.clip(iy, 0, height - 1)
            
            points += flow[iy, ix]

            in_bounds = (points[:, 0] >= 0) & (points[:, 0] < width) & \
                        (points[:, 1] >= 0) & (points[:, 1] < height)
            
            points = points[in_bounds]
            anchors = anchors[in_bounds]

            if frame_idx % PRUNE_INTERVAL == 0:
                diff = np.linalg.norm(points - anchors, axis=1)
                moving = diff > STALE_THRESHOLD
                points = points[moving]
                anchors = points.copy()

        if frame_idx % M_FRAMES == 0:
            occ_mask = np.full((height, width), 255, dtype=np.uint8)
            for px, py in points:
                cv2.circle(occ_mask, (int(px), int(py)), MIN_DIST, 0, -1)
            
            gap_points = get_sparse_points(occ_mask, GRID_STEP)
            if len(gap_points) > 0:
                points = np.vstack([points, gap_points])
                anchors = np.vstack([anchors, gap_points])

        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for px, py in points:
            cv2.circle(vis, (int(px), int(py)), 2, (0, 0, 255), -1)

        cv2.imshow('Farneback Pruned Tracking', vis)
        out.write(vis)
        
        prev_gray = gray
        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()