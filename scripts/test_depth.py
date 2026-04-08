import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.depth_estimator import DepthEstimator
from core.segmentor import Segmentor

def main():
    depth_estimator = DepthEstimator()
    segmentor = Segmentor()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    DEPTH_SKIP = 3
    SEG_SKIP   = 5

    frame_idx = 0

    DISPLAY_W = 400
    DISPLAY_H = 225

    print("Running pipeline — press Q to quit.")

    # initialize with black placeholders
    depth_colored = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    seg_vis       = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # run depth
        if frame_idx % DEPTH_SKIP == 0:
            depth = depth_estimator.predict(frame)
            depth_vis = (depth * 255).astype(np.uint8)
            depth_colored = cv2.resize(
                cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO),
                (DISPLAY_W, DISPLAY_H)
            )

        # run segmentor
        if frame_idx % SEG_SKIP == 0:
            mask = segmentor.predict(frame)
            print(f"[frame {frame_idx}] mask unique values: {np.unique(mask)}")
            seg_vis = cv2.resize(
                cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                (DISPLAY_W, DISPLAY_H)
            )

        # resize raw frame
        frame_s = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

        # add labels
        for img, label in zip(
            [frame_s, depth_colored, seg_vis],
            ["raw", "depth", "person mask"]
        ):
            cv2.putText(img, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 2x2 grid — top row: raw | depth, bottom row: mask centered
        top_row   = np.hstack([frame_s, depth_colored])

        # pad mask to same width as top row so vstack works
        pad_w     = DISPLAY_W // 2
        pad_left  = np.zeros((DISPLAY_H, pad_w, 3), dtype=np.uint8)
        pad_right = np.zeros((DISPLAY_H, pad_w, 3), dtype=np.uint8)
        bottom_row = np.hstack([pad_left, seg_vis, pad_right])

        combined = np.vstack([top_row, bottom_row])
        cv2.imshow("DepthAR — raw | depth | mask", combined)

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
