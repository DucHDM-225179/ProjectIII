import numpy as np
import cv2

# --- Configuration ---
VIDEO_PATH = 'input.mp4'
OUTPUT_PATH = 'output_lucas.mp4'

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=0,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    # Video Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    tracks = []
    frame_idx = 0
    detect_interval = 5
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
                    
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

        if len(tracks) > 0:
            img0, img1 = prev_gray, frame_gray
            # Get latest position of each track
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            
            # Forward Optical Flow (t -> t+1)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            # Backward Optical Flow (t+1 -> t) for verification
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            
            # Back-tracking check: Distance between original p0 and back-tracked p0r
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
            good = d < 1
            
            new_tracks = []
            for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > 10: # Keep track history length
                    del tr[0]
                new_tracks.append(tr)
                # Draw red point for tracking
                cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
            
            tracks = new_tracks

        # Detect new features periodically
        if frame_idx % detect_interval == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            for x, y in [np.int32(tr[-1]) for tr in tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)
            
            p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    tracks.append([(x, y)])

        out.write(vis)
        cv2.imshow('Tracking', vis)
        
        prev_gray = frame_gray
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == 27: # ESC to exit
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()