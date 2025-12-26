import numpy as np
import cv2
from collections import deque

# --- Configuration ---
VIDEO_PATH = 'input.mp4'
OUTPUT_PATH = 'output_farneback_hsv.mp4'
M_GAP = 4  # The frame gap: compare current frame with the one M_GAP frames ago
THRESHOLD = 2.0 # Minimum magnitude to show flow

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open {VIDEO_PATH}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    # We use a deque to store the previous grayscale frames
    # maxlen=M_GAP + 1 ensures we always have the frame from M steps ago at index 0
    frame_buffer = deque(maxlen=M_GAP + 1)

    # Initialize HSV canvas
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    print(f"Processing video with M={M_GAP}... Press 'ESC' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_buffer.append(current_gray)

        # We can only calculate flow if we have enough frames in the buffer
        if len(frame_buffer) > M_GAP:
            # Compare current frame with the oldest frame in buffer (M frames ago)
            prvs_gray = frame_buffer[0]
            
            # Calculate Flow
            flow = cv2.calcOpticalFlowFarneback(
                prvs_gray, current_gray, None, 
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Post-processing
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag[mag < THRESHOLD] = 0
            
            # Map to HSV
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Visualization
            vis = cv2.addWeighted(cv2.cvtColor(current_gray, cv2.COLOR_GRAY2BGR), 0.5, bgr_flow, 0.5, 0)
        else:
            # For the first M frames, we don't have flow yet, so just write the original frame
            vis = cv2.cvtColor(current_gray, cv2.COLOR_GRAY2BGR)

        cv2.imshow('Farneback M-Frame Flow', vis)
        out.write(vis)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done! Saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()