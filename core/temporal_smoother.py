import cv2
import numpy as np
from collections import deque


class TemporalSmoother:
    """
    Addresses the core weakness of self-supervised monocular depth models:
    temporal inconsistency. Because models like Depth Anything V2 process
    each frame independently, depth estimates flicker frame-to-frame even
    for static scenes — a direct consequence of the per-frame photometric
    loss used during self-supervised training.

    This module maintains a rolling buffer of depth maps and applies
    exponential moving average (EMA) smoothing to stabilize estimates
    across time. It also computes a per-pixel consistency score that
    shows WHERE the depth model is most uncertain — which is exactly
    the signal we use to decide where the symbolic correction layer
    should intervene most aggressively.

    Reference connection:
        Self-supervised depth models minimize photometric reprojection
        error between adjacent frames during training, but at inference
        time no such temporal constraint exists. This module re-introduces
        temporal consistency as a post-processing step.
    """

    def __init__(
        self,
        buffer_size: int = 8,
        ema_alpha: float = 0.35,
        consistency_thresh: float = 0.08
    ):
        """
        Args:
            buffer_size         : number of frames to keep in rolling buffer
            ema_alpha           : EMA weight for current frame (higher = less smoothing)
                                  0.35 balances responsiveness with stability
            consistency_thresh  : per-pixel variance above this = high uncertainty
        """
        self.buffer_size          = buffer_size
        self.ema_alpha            = ema_alpha
        self.consistency_thresh   = consistency_thresh

        self.buffer               = deque(maxlen=buffer_size)
        self.smoothed_depth       = None
        self.consistency_map      = None

    def update(self, depth: np.ndarray) -> tuple:
        """
        Update buffer with new depth frame, return smoothed depth
        and consistency map.

        Args:
            depth : float32 H x W, values in [0, 1]

        Returns:
            smoothed_depth   : float32 H x W — temporally stable depth
            consistency_map  : float32 H x W — per-pixel uncertainty [0, 1]
                               0 = very consistent, 1 = highly uncertain
        """
        self.buffer.append(depth.copy())

        # EMA smoothing — weight current frame by alpha, history by (1-alpha)
        if self.smoothed_depth is None:
            self.smoothed_depth = depth.copy()
        else:
            self.smoothed_depth = (
                self.ema_alpha * depth +
                (1.0 - self.ema_alpha) * self.smoothed_depth
            )

        # consistency map = per-pixel variance across the buffer
        # high variance means the model is changing its mind about this pixel
        if len(self.buffer) >= 2:
            stack = np.stack(list(self.buffer), axis=0)  # (N, H, W)
            variance = np.var(stack, axis=0)             # (H, W)

            # normalize variance to [0, 1] for visualization
            # clip at consistency_thresh to avoid extreme values dominating
            consistency = np.clip(variance / self.consistency_thresh, 0.0, 1.0)
            self.consistency_map = consistency.astype(np.float32)
        else:
            self.consistency_map = np.zeros_like(depth)

        return self.smoothed_depth, self.consistency_map

    def get_consistency_visualization(
        self,
        w: int,
        h: int
    ) -> np.ndarray:
        """
        Returns a colored heatmap of the consistency map for display.
        Warm colors = high uncertainty (model flickering here)
        Cool colors = stable depth estimates

        Args:
            w, h : target display width and height

        Returns:
            BGR heatmap (h x w x 3)
        """
        if self.consistency_map is None:
            return np.zeros((h, w, 3), dtype=np.uint8)

        vis = (self.consistency_map * 255).astype(np.uint8)
        colored = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        return cv2.resize(colored, (w, h))

    def is_ready(self) -> bool:
        return len(self.buffer) >= 2

    def reset(self):
        self.buffer.clear()
        self.smoothed_depth = None
        self.consistency_map = None