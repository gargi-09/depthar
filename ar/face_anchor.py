import cv2
import numpy as np
import math


class FaceAnchoredReticle:
    """
    A sci-fi targeting reticle anchored to the detected face.
    Uses monocular depth to scale the reticle in real time —
    closer face = larger reticle, demonstrating depth-driven AR.

    The reticle orbits, rotates, and pulses based on the
    temporal consistency score from the depth model.
    """

    def __init__(self):
        # OpenCV face detector — lightweight, CPU friendly
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # orbit state
        self.orbit_angle   = 0.0
        self.ring_angle    = 0.0
        self.pulse_phase   = 0.0
        self.last_face     = None   # (cx, cy, w, h) — persist when face lost
        self.smoothed_scale = 1.0

    def detect_face(self, frame: np.ndarray):
        """Returns (cx, cy, w, h) of largest face or None."""
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return None
        # pick largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        return (x + w // 2, y + h // 2, w, h)

    def _draw_rotating_ring(
        self,
        canvas, cx, cy, radius, angle,
        color, thickness=1, segments=8, gap_ratio=0.3
    ):
        """Draws a dashed rotating ring — the core reticle element."""
        seg_angle = 2 * math.pi / segments
        gap       = seg_angle * gap_ratio

        for i in range(segments):
            start_a = angle + i * seg_angle + gap / 2
            end_a   = angle + (i + 1) * seg_angle - gap / 2
            pts = []
            steps = max(6, int((end_a - start_a) / 0.05))
            for j in range(steps + 1):
                a   = start_a + (end_a - start_a) * j / steps
                px  = int(cx + radius * math.cos(a))
                py  = int(cy + radius * math.sin(a))
                pts.append((px, py))
            for j in range(len(pts) - 1):
                cv2.line(canvas, pts[j], pts[j+1], color, thickness, cv2.LINE_AA)

    def _draw_corner_brackets(self, canvas, cx, cy, size, color):
        """Draws the four corner brackets of a targeting reticle."""
        half  = size // 2
        arm   = size // 4
        thick = 2

        corners = [
            (cx - half, cy - half, 1,  1),
            (cx + half, cy - half, -1, 1),
            (cx - half, cy + half, 1,  -1),
            (cx + half, cy + half, -1, -1),
        ]
        for bx, by, dx, dy in corners:
            cv2.line(canvas,
                     (bx, by), (bx + dx * arm, by),
                     color, thick, cv2.LINE_AA)
            cv2.line(canvas,
                     (bx, by), (bx, by + dy * arm),
                     color, thick, cv2.LINE_AA)

    def _draw_scanlines(self, canvas, cx, cy, size, alpha, color):
        """Draws horizontal scan lines inside the reticle box."""
        half = size // 2
        step = max(4, size // 10)
        overlay = canvas.copy()
        for y in range(cy - half, cy + half, step):
            cv2.line(overlay,
                     (cx - half, y), (cx + half, y),
                     color, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha * 0.15, canvas, 1 - alpha * 0.15, 0, canvas)

    def _draw_crosshair(self, canvas, cx, cy, size, color):
        """Draws center crosshair with a gap in the middle."""
        gap  = size // 6
        arm  = size // 4
        thick = 1
        # horizontal
        cv2.line(canvas, (cx - arm, cy), (cx - gap, cy), color, thick, cv2.LINE_AA)
        cv2.line(canvas, (cx + gap, cy), (cx + arm, cy), color, thick, cv2.LINE_AA)
        # vertical
        cv2.line(canvas, (cx, cy - arm), (cx, cy - gap), color, thick, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy + gap), (cx, cy + arm), color, thick, cv2.LINE_AA)
        # center dot
        cv2.circle(canvas, (cx, cy), 2, color, -1, cv2.LINE_AA)

    def _draw_hud_readout(
        self, canvas, cx, cy, size,
        depth_val, consistency, color
    ):
        """Draws sci-fi style data readouts around the reticle."""
        half    = size // 2
        font    = cv2.FONT_HERSHEY_SIMPLEX
        fscale  = max(0.3, size / 400.0)
        thick   = 1

        dist_str  = f"DIST: {depth_val:.2f}m"
        conf_str  = f"CONF: {consistency:.0%}"
        lock_str  = "[ LOCKED ]"

        # bottom left
        cv2.putText(canvas, dist_str,
                    (cx - half, cy + half + 18),
                    font, fscale, color, thick, cv2.LINE_AA)
        # bottom right
        cv2.putText(canvas, conf_str,
                    (cx + half // 2, cy + half + 18),
                    font, fscale, color, thick, cv2.LINE_AA)
        # top center
        tw = cv2.getTextSize(lock_str, font, fscale, thick)[0][0]
        cv2.putText(canvas, lock_str,
                    (cx - tw // 2, cy - half - 8),
                    font, fscale, color, thick, cv2.LINE_AA)

    def render(
        self,
        frame: np.ndarray,
        depth: np.ndarray,
        consistency_score: float,
        dt: float = 0.05
    ) -> np.ndarray:
        """
        Main render call. Detects face, anchors reticle, scales by depth.

        Args:
            frame             : BGR webcam frame
            depth             : float32 H x W depth map [0,1]
            consistency_score : from TemporalSmoother [0,1] — drives pulse
            dt                : time delta per frame for animation

        Returns:
            frame with reticle composited on top
        """
        canvas = frame.copy()
        h, w   = frame.shape[:2]

        # --- face detection ---
        face = self.detect_face(frame)
        if face is not None:
            self.last_face = face
        elif self.last_face is not None:
            face = self.last_face   # hold last known position
        else:
            return canvas           # no face ever detected yet

        cx, cy, fw, fh = face

        # --- depth-driven scale ---
        # sample depth at face center — closer face (lower depth) = bigger reticle
        fx = np.clip(cx, 0, w - 1)
        fy = np.clip(cy, 0, h - 1)
        face_depth = float(depth[fy, fx]) if depth is not None else 0.5

        # invert: low depth (close) = large scale
        target_scale  = 1.0 + (1.0 - face_depth) * 1.5
        # smooth the scale so it doesn't jump
        self.smoothed_scale += 0.15 * (target_scale - self.smoothed_scale)

        base_size = int(max(fw, fh) * 1.6 * self.smoothed_scale)
        base_size = max(80, min(base_size, 500))

        # --- animation ---
        pulse_speed     = 1.0 + (1.0 - consistency_score) * 4.0
        self.pulse_phase  = (self.pulse_phase + dt * pulse_speed) % (2 * math.pi)
        self.orbit_angle  = (self.orbit_angle + dt * 0.8) % (2 * math.pi)
        self.ring_angle   = (self.ring_angle  + dt * 1.5) % (2 * math.pi)

        pulse = 0.6 + 0.4 * math.sin(self.pulse_phase)

        # --- colors ---
        # primary: cyan-green sci-fi color, intensity driven by pulse
        pri_int  = int(180 + 75 * pulse)
        primary  = (0, pri_int, pri_int)
        secondary = (0, int(pri_int * 0.5), pri_int)
        dim      = (0, int(pri_int * 0.3), int(pri_int * 0.3))

        # --- orbit point (small orb orbiting the reticle) ---
        orbit_r  = base_size // 2 + 20
        orb_x    = int(cx + orbit_r * math.cos(self.orbit_angle))
        orb_y    = int(cy + orbit_r * math.sin(self.orbit_angle))
        orb_size = max(4, int(8 * self.smoothed_scale * pulse))
        cv2.circle(canvas, (orb_x, orb_y), orb_size, primary, -1, cv2.LINE_AA)
        cv2.circle(canvas, (orb_x, orb_y), orb_size + 3, dim,  1,  cv2.LINE_AA)

        # second orbit orb — opposite side, different phase
        orb2_x = int(cx + orbit_r * math.cos(self.orbit_angle + math.pi))
        orb2_y = int(cy + orbit_r * math.sin(self.orbit_angle + math.pi))
        cv2.circle(canvas, (orb2_x, orb2_y), max(3, orb_size - 2), secondary, -1, cv2.LINE_AA)

        # --- outer rotating dashed ring ---
        self._draw_rotating_ring(
            canvas, cx, cy,
            radius=base_size // 2 + 5,
            angle=self.ring_angle,
            color=primary, thickness=1, segments=8, gap_ratio=0.25
        )

        # --- inner counter-rotating ring ---
        self._draw_rotating_ring(
            canvas, cx, cy,
            radius=base_size // 4,
            angle=-self.ring_angle * 1.5,
            color=secondary, thickness=1, segments=6, gap_ratio=0.3
        )

        # --- corner brackets ---
        self._draw_corner_brackets(canvas, cx, cy, base_size, primary)

        # --- scan lines ---
        self._draw_scanlines(canvas, cx, cy, base_size, pulse, primary)

        # --- crosshair ---
        self._draw_crosshair(canvas, cx, cy, base_size, primary)

        # --- HUD readout ---
        self._draw_hud_readout(
            canvas, cx, cy, base_size,
            depth_val=face_depth * 5.0,   # fake metric units for display
            consistency=1.0 - consistency_score,
            color=primary
        )

        # blend canvas back onto frame for subtle transparency
        cv2.addWeighted(canvas, 0.85, frame, 0.15, 0, canvas)

        return canvas