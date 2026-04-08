import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.depth_estimator import DepthEstimator
from core.segmentor import Segmentor
from core.occlusion_reasoner import OcclusionReasoner

def main():
    depth_estimator   = DepthEstimator()
    segmentor         = Segmentor()
    occlusion_reasoner = OcclusionReasoner(
        depth_threshold=0.4,
        boundary_kernel=15,
        alpha=0.7
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    DEPTH_SKIP = 3
    SEG_SKIP   = 5
    frame_idx  = 0
    DISPLAY_W  = 400
    DISPLAY_H  = 225

    print("Running pipeline — press Q to quit.")

    depth         = None
    mask          = None
    depth_colored = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    seg_vis       = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    occ_vis       = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

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
            seg_vis = cv2.resize(
                cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                (DISPLAY_W, DISPLAY_H)
            )
            
        # run occlusion reasoner when both depth and mask are available
        if depth is not None and mask is not None:
            # ensure depth and mask are the same size before reasoning
            depth_resized = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_resized  = cv2.resize(mask, (w, h),  interpolation=cv2.INTER_NEAREST)

            occlusion = occlusion_reasoner.predict(depth_resized, mask_resized)
            occ_colored = cv2.applyColorMap(occlusion, cv2.COLORMAP_VIRIDIS)
            occ_vis = cv2.resize(occ_colored, (DISPLAY_W, DISPLAY_H))

        # resize raw frame
        frame_s = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

        # add labels
        for img, label in zip(
            [frame_s, depth_colored, seg_vis, occ_vis],
            ["raw", "depth", "person mask", "occlusion mask"]
        ):
            cv2.putText(img, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 2x2 grid
        top_row    = np.hstack([frame_s, depth_colored])
        bottom_row = np.hstack([seg_vis, occ_vis])
        combined   = np.vstack([top_row, bottom_row])

        cv2.imshow("DepthAR — raw | depth | mask | occlusion", combined)

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()