import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.depth_estimator import DepthEstimator
from core.segmentor import Segmentor
from core.occlusion_reasoner import OcclusionReasoner
from ar.renderer import ARRenderer

def draw_hud(frame, position, mode):
    """Draw minimal HUD overlay showing controls and cube position."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    controls = "W/S: forward/back  |  A/D: left/right  |  I/K: up/down  |  TAB: debug  |  Q: quit"
    pos_text = f"cube position: x={position[0]:.1f}  y={position[1]:.1f}  z={position[2]:.1f}"

    cv2.putText(frame, controls, (10, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, pos_text, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 255, 100), 1, cv2.LINE_AA)

    mode_label = "MODE: DEBUG" if mode == "debug" else "MODE: AR"
    cv2.putText(frame, mode_label, (w - 140, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 100), 1, cv2.LINE_AA)
    return frame

def main():
    depth_estimator    = DepthEstimator()
    segmentor          = Segmentor()
    occlusion_reasoner = OcclusionReasoner(
        depth_threshold=0.35,
        boundary_kernel=15,
        alpha=0.8
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    ret, test_frame = cap.read()
    if not ret:
        print("Error: could not read from webcam.")
        return

    frame_h, frame_w = test_frame.shape[:2]
    renderer = ARRenderer(frame_w=frame_w, frame_h=frame_h)

    DEPTH_SKIP = 3
    SEG_SKIP   = 5
    frame_idx  = 0
    DISPLAY_W  = 400
    DISPLAY_H  = 225

    # cube state
    # cube state
    position  = np.array([3.5,  0.5, 8.0], dtype=np.float32)
    rotation  = np.array([0.3,  0.5, 0.1], dtype=np.float32)
    rot_speed = np.array([0.005, 0.01, 0.002], dtype=np.float32)
    scale     = 80.0
    MOVE_STEP = 0.3

    # display mode: "ar" = fullscreen AR, "debug" = 2x2 grid
    display_mode = "ar"

    depth         = None
    mask          = None
    occlusion     = None
    depth_colored = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    seg_vis       = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    occ_vis       = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)

    cv2.namedWindow("DepthAR", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DepthAR", 800, 500)
    print("DepthAR running — press Q to quit, TAB to toggle debug view.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # animate cube rotation every frame (cheap — no model inference)
        rotation = rotation + rot_speed

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

        # run occlusion reasoner
        if depth is not None and mask is not None:
            depth_resized = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_resized  = cv2.resize(mask,  (w, h), interpolation=cv2.INTER_NEAREST)
            occlusion     = occlusion_reasoner.predict(depth_resized, mask_resized)
            occ_colored   = cv2.applyColorMap(occlusion, cv2.COLORMAP_VIRIDIS)
            occ_vis       = cv2.resize(occ_colored, (DISPLAY_W, DISPLAY_H))

        # AR compositing
        if occlusion is not None:
            ar_frame = renderer.composite(
                frame, occlusion,
                position=np.array([1.2, 0.2, 5.0], dtype=np.float32),
                scale=60.0
            )
        else:
            ar_frame = frame.copy()

        # display
        if display_mode == "ar":
            ar_frame = draw_hud(ar_frame, position, display_mode)
            cv2.imshow("DepthAR", ar_frame)

        else:  # debug grid
            frame_s = cv2.resize(frame,    (DISPLAY_W, DISPLAY_H))
            ar_s    = cv2.resize(ar_frame, (DISPLAY_W, DISPLAY_H))

            for img, label in zip(
                [frame_s, ar_s, seg_vis, occ_vis],
                ["raw", "DepthAR", "person mask", "occlusion mask"]
            ):
                cv2.putText(img, label, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            top_row    = np.hstack([frame_s, ar_s])
            bottom_row = np.hstack([seg_vis, occ_vis])
            combined   = np.vstack([top_row, bottom_row])
            combined   = draw_hud(combined, position, display_mode)
            cv2.imshow("DepthAR", combined)

        frame_idx += 1

        # keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 9:  # TAB
            display_mode = "debug" if display_mode == "ar" else "ar"
        elif key == ord("a"):
            position[0] -= MOVE_STEP   # move left
        elif key == ord("d"):
            position[0] += MOVE_STEP   # move right
        elif key == ord("w"):
            position[2] -= MOVE_STEP   # move closer
        elif key == ord("s"):
            position[2] += MOVE_STEP   # move farther
        elif key == ord("i"):
            position[1] -= MOVE_STEP   # move up
        elif key == ord("k"):
            position[1] += MOVE_STEP   # move down

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()