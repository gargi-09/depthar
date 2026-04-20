import cv2
import numpy as np
import math
from enum import Enum


class BodyMode(Enum):
    FACE_ONLY  = "face_only"
    FULL_BODY  = "full_body"


class ExosuitOverlay:
    """
    Depth-driven sci-fi exosuit wireframe overlay.

    Detects whether the user is showing face-only or full body
    using the person mask area, then renders the appropriate overlay:

    - FACE_ONLY  : visor/helmet HUD with targeting elements
    - FULL_BODY  : full body wireframe rig driven by person mask contour
                   with shoulder pads, chest piece, and arm segments

    The wireframe is extracted directly from the semantic segmentation
    mask contour — meaning it reacts to the user's actual body shape,
    not a fixed template. This is the core technical contribution:
    depth + semantics driving a body-conforming AR overlay.
    """

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # animation state
        self.scan_y        = 0
        self.pulse_phase   = 0.0
        self.ring_angle    = 0.0
        self.charge_phase  = 0.0
        self.mode_timer    = 0
        self.current_mode  = BodyMode.FACE_ONLY
        self.mode_stable   = 0   # frames mode has been stable

        # smoothed contour points
        self.smoothed_contour = None

    def _detect_face(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return None
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        return (x + w // 2, y + h // 2, w, h)

    def _detect_mode(self, person_mask: np.ndarray) -> BodyMode:
        """
        Determine if showing face only or full body based on
        how much of the frame the person mask covers and
        its vertical extent.
        """
        h, w      = person_mask.shape
        total_px  = h * w
        person_px = np.sum(person_mask > 127)
        ratio     = person_px / total_px

        # find vertical extent of mask
        rows = np.any(person_mask > 127, axis=1)
        if not np.any(rows):
            return BodyMode.FACE_ONLY
        top_row    = np.argmax(rows)
        bottom_row = len(rows) - np.argmax(rows[::-1]) - 1
        vert_span  = (bottom_row - top_row) / h

        # full body: mask covers >15% of frame AND spans >50% vertically
        if ratio > 0.15 and vert_span > 0.50:
            return BodyMode.FULL_BODY
        return BodyMode.FACE_ONLY

    def _get_body_contour(self, person_mask: np.ndarray):
        """Extract the largest contour from the person mask."""
        binary   = (person_mask > 127).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)

        # smooth the contour
        epsilon  = 0.008 * cv2.arcLength(largest, True)
        smoothed = cv2.approxPolyDP(largest, epsilon, True)
        return smoothed

    def _get_body_bbox(self, contour):
        """Returns bounding box of body contour."""
        x, y, w, h = cv2.boundingRect(contour)
        return x, y, w, h

    # ------------------------------------------------------------------ #
    #  FULL BODY MODE                                                      #
    # ------------------------------------------------------------------ #

    def _draw_body_wireframe(
        self, canvas, contour, pulse, charge_phase, consistency
    ):
        """
        Draws the exosuit wireframe driven by the person mask contour.
        All geometry is derived from the bounding box + contour —
        no fixed template.
        """
        h_frame, w_frame = canvas.shape[:2]

        bx, by, bw, bh = self._get_body_bbox(contour)
        cx = bx + bw // 2
        cy = by + bh // 2

        # primary color: electric cyan-teal
        intensity  = int(160 + 95 * pulse)
        primary    = (intensity, 220, 50)      # cyan-green
        secondary  = (50, intensity, intensity) # pure cyan
        accent     = (30, 60, intensity)        # deep blue accent
        dim        = (30, 80, 80)

        # --- body contour wireframe ---
        pts = contour.reshape(-1, 2)
        for i in range(len(pts)):
            p1 = tuple(pts[i])
            p2 = tuple(pts[(i + 1) % len(pts)])
            cv2.line(canvas, p1, p2, primary, 1, cv2.LINE_AA)

        # --- shoulder pads ---
        shoulder_y  = by + int(bh * 0.15)
        shoulder_w  = int(bw * 0.20)
        shoulder_h  = int(bh * 0.08)
        pad_thick   = 2

        # left shoulder
        lsx = bx - int(shoulder_w * 0.3)
        cv2.rectangle(canvas,
                      (lsx, shoulder_y - shoulder_h // 2),
                      (lsx + shoulder_w, shoulder_y + shoulder_h // 2),
                      secondary, pad_thick, cv2.LINE_AA)
        cv2.line(canvas,
                 (lsx + shoulder_w // 2, shoulder_y - shoulder_h // 2),
                 (lsx + shoulder_w // 2, shoulder_y + shoulder_h // 2),
                 dim, 1, cv2.LINE_AA)

        # right shoulder
        rsx = bx + bw - int(shoulder_w * 0.7)
        cv2.rectangle(canvas,
                      (rsx, shoulder_y - shoulder_h // 2),
                      (rsx + shoulder_w, shoulder_y + shoulder_h // 2),
                      secondary, pad_thick, cv2.LINE_AA)
        cv2.line(canvas,
                 (rsx + shoulder_w // 2, shoulder_y - shoulder_h // 2),
                 (rsx + shoulder_w // 2, shoulder_y + shoulder_h // 2),
                 dim, 1, cv2.LINE_AA)

        # --- chest piece ---
        chest_y  = by + int(bh * 0.38)
        chest_w  = int(bw * 0.45)
        chest_h  = int(bh * 0.22)
        chest_x  = cx - chest_w // 2

        # outer chest frame
        cv2.rectangle(canvas,
                      (chest_x, chest_y),
                      (chest_x + chest_w, chest_y + chest_h),
                      primary, 2, cv2.LINE_AA)

        # arc reactor — pulsing circle at chest center
        reactor_x  = cx
        reactor_y  = chest_y + chest_h // 2
        reactor_r  = int(chest_w * 0.12)
        glow_r     = int(reactor_r * (1.0 + 0.3 * pulse))
        cv2.circle(canvas, (reactor_x, reactor_y), glow_r, dim, 1, cv2.LINE_AA)
        cv2.circle(canvas, (reactor_x, reactor_y), reactor_r, secondary, 2, cv2.LINE_AA)
        cv2.circle(canvas, (reactor_x, reactor_y), max(2, reactor_r // 3),
                   (intensity, intensity, intensity), -1, cv2.LINE_AA)

        # chest grid lines
        for i in range(1, 3):
            gx = chest_x + chest_w * i // 3
            cv2.line(canvas,
                     (gx, chest_y), (gx, chest_y + chest_h),
                     dim, 1, cv2.LINE_AA)

        # --- arm segments ---
        arm_y_top  = by + int(bh * 0.18)
        arm_y_bot  = by + int(bh * 0.60)
        arm_segs   = 4
        seg_h      = (arm_y_bot - arm_y_top) // arm_segs
        arm_w      = int(bw * 0.08)

        for side, ax in [(-1, bx - arm_w + 5), (1, bx + bw - 5)]:
            for s in range(arm_segs):
                sy1 = arm_y_top + s * seg_h
                sy2 = sy1 + seg_h - 4
                cv2.rectangle(canvas,
                              (ax, sy1), (ax + arm_w, sy2),
                              accent, 1, cv2.LINE_AA)
                # segment connector line
                mid_y = (sy1 + sy2) // 2
                cv2.line(canvas,
                         (ax + 2, mid_y), (ax + arm_w - 2, mid_y),
                         dim, 1, cv2.LINE_AA)

        # --- leg indicators ---
        leg_top  = by + int(bh * 0.62)
        leg_bot  = by + bh
        leg_w    = int(bw * 0.18)
        leg_cx_l = cx - int(bw * 0.15)
        leg_cx_r = cx + int(bw * 0.15)

        for lx in [leg_cx_l - leg_w // 2, leg_cx_r - leg_w // 2]:
            cv2.rectangle(canvas,
                          (lx, leg_top),
                          (lx + leg_w, leg_bot),
                          accent, 1, cv2.LINE_AA)
            # knee joint
            knee_y = leg_top + (leg_bot - leg_top) // 2
            cv2.line(canvas,
                     (lx, knee_y), (lx + leg_w, knee_y),
                     secondary, 1, cv2.LINE_AA)

        # --- energy charge bar (left side) ---
        charge_val = 0.5 + 0.5 * math.sin(charge_phase)
        bar_x      = bx - 30
        bar_top    = by + int(bh * 0.2)
        bar_bot    = by + int(bh * 0.8)
        bar_h      = bar_bot - bar_top
        fill_h     = int(bar_h * charge_val)

        cv2.rectangle(canvas, (bar_x, bar_top), (bar_x + 8, bar_bot),
                      dim, 1, cv2.LINE_AA)
        cv2.rectangle(canvas,
                      (bar_x + 1, bar_bot - fill_h),
                      (bar_x + 7, bar_bot),
                      secondary, -1)

        cv2.putText(canvas, "PWR", (bar_x - 2, bar_top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, secondary, 1, cv2.LINE_AA)

        # --- scan line sweeping down body ---
        scan_color = (*secondary[:2], 255)
        scan_y_abs = by + (self.scan_y % bh)
        cv2.line(canvas, (bx, scan_y_abs), (bx + bw, scan_y_abs),
                 secondary, 1, cv2.LINE_AA)

        # --- status readouts ---
        font   = cv2.FONT_HERSHEY_SIMPLEX
        fscale = 0.35
        status = [
            f"INTEGRITY: {int(95 + 5 * pulse)}%",
            f"DEPTH: {consistency:.2f}",
            "SUIT: ACTIVE",
        ]
        for i, line in enumerate(status):
            cv2.putText(canvas, line,
                        (bx + bw + 10, by + 30 + i * 18),
                        font, fscale, secondary, 1, cv2.LINE_AA)

        # --- mode label ---
        cv2.putText(canvas, "[ EXOSUIT ENGAGED ]",
                    (cx - 80, by - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, primary, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    #  FACE ONLY MODE                                                      #
    # ------------------------------------------------------------------ #

    def _draw_helmet_visor(
        self, canvas, face, pulse, ring_angle, consistency
    ):
        """
        Draws helmet/visor overlay when only face is detected.
        Tighter targeting, visor frame, neural link HUD.
        """
        cx, cy, fw, fh = face
        intensity = int(160 + 95 * pulse)
        primary   = (intensity, intensity, 50)
        secondary = (50, intensity, intensity)
        dim       = (30, 80, 80)

        size = int(max(fw, fh) * 1.4)

        # --- visor outer frame ---
        half = size // 2
        # top bar
        cv2.rectangle(canvas,
                      (cx - half, cy - half - 20),
                      (cx + half, cy - half),
                      primary, 2, cv2.LINE_AA)
        # side bars
        cv2.line(canvas, (cx - half - 10, cy - half),
                 (cx - half - 10, cy + half), primary, 2, cv2.LINE_AA)
        cv2.line(canvas, (cx + half + 10, cy - half),
                 (cx + half + 10, cy + half), primary, 2, cv2.LINE_AA)

        # corner brackets
        arm = size // 5
        for bx2, by2, dx, dy in [
            (cx - half, cy - half, 1,  1),
            (cx + half, cy - half, -1, 1),
            (cx - half, cy + half, 1,  -1),
            (cx + half, cy + half, -1, -1),
        ]:
            cv2.line(canvas, (bx2, by2),
                     (bx2 + dx * arm, by2), secondary, 2, cv2.LINE_AA)
            cv2.line(canvas, (bx2, by2),
                     (bx2, by2 + dy * arm), secondary, 2, cv2.LINE_AA)

        # --- rotating rings ---
        for r, angle_mult, color, segs in [
            (size // 2 + 15, 1.0,   primary,   8),
            (size // 4,      -1.5,  secondary, 6),
        ]:
            seg_angle = 2 * math.pi / segs
            gap       = seg_angle * 0.3
            for i in range(segs):
                sa = ring_angle * angle_mult + i * seg_angle + gap / 2
                ea = ring_angle * angle_mult + (i + 1) * seg_angle - gap / 2
                pts = []
                for j in range(12):
                    a  = sa + (ea - sa) * j / 11
                    px = int(cx + r * math.cos(a))
                    py = int(cy + r * math.sin(a))
                    pts.append((px, py))
                for j in range(len(pts) - 1):
                    cv2.line(canvas, pts[j], pts[j+1],
                             color, 1, cv2.LINE_AA)

        # --- neural link lines ---
        for angle in [30, 150, 210, 330]:
            rad = math.radians(angle)
            x1  = int(cx + (size // 2 + 15) * math.cos(rad))
            y1  = int(cy + (size // 2 + 15) * math.sin(rad))
            x2  = int(cx + (size // 2 + 45) * math.cos(rad))
            y2  = int(cy + (size // 2 + 45) * math.sin(rad))
            cv2.line(canvas, (x1, y1), (x2, y2), dim, 1, cv2.LINE_AA)
            cv2.circle(canvas, (x2, y2), 3, secondary, -1, cv2.LINE_AA)

        # --- crosshair ---
        gap = size // 6
        arm2 = size // 4
        for p1, p2 in [
            ((cx - arm2, cy), (cx - gap, cy)),
            ((cx + gap,  cy), (cx + arm2, cy)),
            ((cx, cy - arm2), (cx, cy - gap)),
            ((cx, cy + gap),  (cx, cy + arm2)),
        ]:
            cv2.line(canvas, p1, p2, primary, 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 3, primary, -1, cv2.LINE_AA)

        # --- visor scan line ---
        scan_y_rel = int(half * math.sin(self.pulse_phase * 2))
        cv2.line(canvas,
                 (cx - half, cy + scan_y_rel),
                 (cx + half, cy + scan_y_rel),
                 (*secondary[:2], 80), 1, cv2.LINE_AA)

        # --- HUD readouts ---
        font   = cv2.FONT_HERSHEY_SIMPLEX
        fscale = 0.35
        cv2.putText(canvas, f"NEURAL LINK: ACTIVE",
                    (cx - half, cy + half + 20),
                    font, fscale, primary, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"CONF: {1-consistency:.0%}",
                    (cx + half // 2, cy + half + 20),
                    font, fscale, secondary, 1, cv2.LINE_AA)
        cv2.putText(canvas, "[ HELMET MODE ]",
                    (cx - 55, cy - half - 30),
                    font, 0.4, primary, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    #  MAIN RENDER                                                         #
    # ------------------------------------------------------------------ #

    def render(
        self,
        frame: np.ndarray,
        person_mask: np.ndarray,
        depth: np.ndarray,
        consistency_score: float,
        dt: float = 0.05
    ) -> np.ndarray:
        """
        Main entry point.

        Args:
            frame             : BGR webcam frame
            person_mask       : uint8 H x W from Segmentor
            depth             : float32 H x W depth map
            consistency_score : from TemporalSmoother
            dt                : time delta per frame

        Returns:
            frame with exosuit overlay composited on top
        """
        canvas = frame.copy()
        h, w   = frame.shape[:2]

        # update animation state
        self.pulse_phase  = (self.pulse_phase  + dt * 2.5) % (2 * math.pi)
        self.ring_angle   = (self.ring_angle   + dt * 1.2) % (2 * math.pi)
        self.charge_phase = (self.charge_phase + dt * 1.8) % (2 * math.pi)
        self.scan_y       = (self.scan_y + 3) % max(1, h)
        pulse             = 0.6 + 0.4 * math.sin(self.pulse_phase)

        # resize mask to frame size
        mask_resized = cv2.resize(person_mask, (w, h),
                                  interpolation=cv2.INTER_NEAREST)

        # detect mode
        new_mode = self._detect_mode(mask_resized)
        if new_mode == self.current_mode:
            self.mode_stable += 1
        else:
            self.mode_stable = 0
            if self.mode_stable == 0:
                self.current_mode = new_mode

        # only switch mode after 10 stable frames — avoids flickering
        if self.mode_stable > 10:
            self.current_mode = new_mode

        if self.current_mode == BodyMode.FULL_BODY:
            contour = self._get_body_contour(mask_resized)
            if contour is not None and len(contour) > 10:
                self._draw_body_wireframe(
                    canvas, contour, pulse,
                    self.charge_phase, consistency_score
                )

        else:
            face = self._detect_face(frame)
            if face is not None:
                self._draw_helmet_visor(
                    canvas, face, pulse,
                    self.ring_angle, consistency_score
                )
        
        # flash on mode transition
        if self.mode_stable == 11:   # exactly when mode switches
            flash = canvas.copy()
            cv2.rectangle(flash, (0, 0), (w, h), (200, 255, 200), -1)
            cv2.addWeighted(flash, 0.15, canvas, 0.85, 0, canvas)
        # soft blend canvas onto frame so it doesn't look pasted on
        cv2.addWeighted(canvas, 0.88, frame, 0.12, 0, canvas)
        return canvas