from PIL.ImageChops import overlay
import cv2
import numpy as np


class ARRenderer:
    """
    Composites a virtual 3D object into a real webcam frame
    using the occlusion mask to correctly hide the virtual object
    behind real-world foreground pixels.

    The virtual object is a wireframe + filled cube rendered using
    OpenCV's projectPoints — no OpenGL dependency needed, runs fine on CPU.
    """

    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h

        # camera intrinsics (approximate for a standard webcam)
        self.focal_length = frame_w * 0.6   # was frame_w
        self.center       = (frame_w // 2, frame_h // 2)
        self.camera_matrix = np.array([
            [self.focal_length, 0,                 self.center[0]],
            [0,                 self.focal_length, self.center[1]],
            [0,                 0,                 1             ]
        ], dtype=np.float32)

        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        # 3D cube vertices centered at origin, side length = 1 unit
        s = 0.5
        self.cube_vertices = np.array([
            [-s, -s, -s], [ s, -s, -s],
            [ s,  s, -s], [-s,  s, -s],
            [-s, -s,  s], [ s, -s,  s],
            [ s,  s,  s], [-s,  s,  s],
        ], dtype=np.float32)

        # cube edges: pairs of vertex indices
        self.cube_edges = [
            (0,1),(1,2),(2,3),(3,0),  # back face
            (4,5),(5,6),(6,7),(7,4),  # front face
            (0,4),(1,5),(2,6),(3,7),  # connecting edges
        ]

        # cube faces: groups of 4 vertex indices (for filled rendering)
        self.cube_faces = [
            [0,1,2,3],  # back
            [4,5,6,7],  # front
            [0,1,5,4],  # bottom
            [2,3,7,6],  # top
            [0,3,7,4],  # left
            [1,2,6,5],  # right
        ]

        # face colors (BGR) — semi-transparent fills
        self.face_colors = [
                (40, 20, 20),
                (40, 20, 20),
                (40, 20, 20),
                (40, 20, 20),
                (40, 20, 20),
                (40, 20, 20),
            ]

    def _project_cube(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        scale: float
    ) -> np.ndarray:
        """
        Projects 3D cube vertices onto 2D image plane.

        Args:
            position : (3,) translation vector [x, y, z] in camera space
            rotation : (3,) rotation vector (Rodrigues)
            scale    : uniform scale factor

        Returns:
            points_2d : (8, 2) projected pixel coordinates
        """
        scaled_vertices = self.cube_vertices * scale

        points_2d, _ = cv2.projectPoints(
            scaled_vertices,
            rotation,
            position,
            self.camera_matrix,
            self.dist_coeffs
        )
        return points_2d.reshape(-1, 2).astype(np.int32)

    def _draw_filled_cube(
        self,
        canvas: np.ndarray,
        points_2d: np.ndarray,
        alpha: float = 0.6
    ) -> np.ndarray:
        """
        Draws a filled semi-transparent cube onto canvas.
        """
        overlay = canvas.copy()

        for face_idx, face in enumerate(self.cube_faces):
            pts = points_2d[face].reshape((-1, 1, 2))
            color = self.face_colors[face_idx]
            cv2.fillPoly(overlay, [pts], color)

        # blend overlay with original for transparency
        result = cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0)        
        return result

    def _draw_wireframe(
        self,
        canvas: np.ndarray,
        points_2d: np.ndarray
    ) -> np.ndarray:
        """
        Draws wireframe edges on top of filled cube.
        """
        for start_idx, end_idx in self.cube_edges:
            pt1 = tuple(points_2d[start_idx])
            pt2 = tuple(points_2d[end_idx])
            cv2.line(canvas, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)

        return canvas

    def composite(
        self,
        frame: np.ndarray,
        occlusion_mask: np.ndarray,
        position: np.ndarray = None,
        rotation: np.ndarray = None,
        scale: float = 150.0
    ) -> np.ndarray:
        """
        Main entry point. Renders virtual cube into frame with
        correct occlusion against real-world foreground.

        Args:
            frame          : BGR webcam frame (H x W x 3)
            occlusion_mask : uint8 H x W, 255 = real foreground
            position       : (3,) 3D position of cube center
                             defaults to center of scene
            rotation       : (3,) Rodrigues rotation vector
                             defaults to slight tilt for visibility
            scale          : size of cube in pixels (projected)

        Returns:
            composited frame (H x W x 3)
        """
        h, w = frame.shape[:2]

        if position is None:
            # place cube slightly to the right and in front
            position = np.array([1.5, 0.5, 6.0], dtype=np.float32)

        if rotation is None:
            # slight rotation so we see multiple faces
            rotation = np.array([0.3, 0.5, 0.1], dtype=np.float32)

        # project cube vertices to 2D
        points_2d = self._project_cube(position, rotation, scale)

        # render virtual object onto a blank canvas
        virtual_canvas = frame.copy()
        virtual_canvas = self._draw_filled_cube(virtual_canvas, points_2d)
        virtual_canvas = self._draw_wireframe(virtual_canvas, points_2d)

        # occlusion compositing:
        # where occlusion_mask = 255 (real foreground) → use original frame
        # where occlusion_mask = 0   (background)      → use virtual canvas
        # feather the occlusion mask edges for smooth blending
        # instead of a hard binary cutoff, we blur the boundary zone
        # so the cube fades out naturally as it goes behind your body
        # stronger interior masking — don't let cube bleed onto person
        soft_mask = occlusion_mask.astype(np.float32)
        soft_mask = cv2.bilateralFilter(soft_mask, d=9, sigmaColor=75, sigmaSpace=75)
        blurred   = cv2.GaussianBlur(soft_mask, (21, 21), 0)

        # hard interior + soft boundary
        interior  = (soft_mask > 180).astype(np.float32)
        boundary  = (soft_mask > 20).astype(np.float32) * (1 - interior)
        soft_mask = interior * 255 + boundary * blurred
        soft_mask = soft_mask / 255.0
        soft_mask = np.clip(soft_mask, 0, 1)

        fg_mask = np.stack([soft_mask] * 3, axis=-1)

        if fg_mask.shape[:2] != (h, w):
            fg_mask = cv2.resize(fg_mask, (w, h))

        composited = (
            frame.astype(np.float32) * fg_mask +
            virtual_canvas.astype(np.float32) * (1.0 - fg_mask)
        ).astype(np.uint8)

        return composited