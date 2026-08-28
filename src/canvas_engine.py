import cv2
import numpy as np
import io
from PIL import Image

class CanvasEngine:
    def __init__(self, width=1280, height=720, history_limit=15):
        self.width = width
        self.height = height
        self.history_limit = history_limit
        
        # Canvas layers: Black canvas by default
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.history = []
        
        # Default drawing state
        self.active_color = (255, 229, 0) # Electric Cyan (BGR)
        self.brush_size = 6
        self.eraser_size = 35
        self.is_erasing = False
        self.prev_point = None
        self.is_drawing_stroke = False

    def resize_if_needed(self, width, height):
        """
        Resizes canvas if frame size changes dynamically.
        """
        if self.width != width or self.height != height:
            self.width = width
            self.height = height
            self.canvas = cv2.resize(self.canvas, (width, height))
            self.history.clear()

    def set_color(self, bgr_color):
        self.active_color = bgr_color
        self.is_erasing = False

    def set_brush_size(self, size):
        self.brush_size = max(1, min(50, size))

    def set_eraser_size(self, size):
        self.eraser_size = max(10, min(100, size))

    def enable_eraser(self, enable=True):
        self.is_erasing = enable

    def start_stroke(self):
        """
        Call when starting a new continuous line stroke to save undo state.
        """
        if not self.is_drawing_stroke:
            self.save_state()
            self.is_drawing_stroke = True

    def end_stroke(self):
        """
        Call when finger is lifted / gesture pauses.
        """
        self.prev_point = None
        self.is_drawing_stroke = False

    def save_state(self):
        """
        Saves a snapshot of current canvas state into history stack for undo.
        """
        if len(self.history) >= self.history_limit:
            self.history.pop(0)
        self.history.append(self.canvas.copy())

    def undo(self):
        """
        Restores previous canvas state from history.
        """
        if self.history:
            self.canvas = self.history.pop()
            self.prev_point = None
            return True
        return False

    def clear(self):
        """
        Clears canvas to black.
        """
        self.save_state()
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.prev_point = None

    def draw_line(self, curr_point):
        """
        Draws a smooth line from prev_point to curr_point on canvas.
        """
        if self.prev_point is None:
            self.prev_point = curr_point
            return

        if self.is_erasing:
            cv2.line(self.canvas, self.prev_point, curr_point, (0, 0, 0), self.eraser_size, cv2.LINE_AA)
            cv2.circle(self.canvas, curr_point, self.eraser_size // 2, (0, 0, 0), -1)
        else:
            cv2.line(self.canvas, self.prev_point, curr_point, self.active_color, self.brush_size, cv2.LINE_AA)
            cv2.circle(self.canvas, curr_point, self.brush_size // 2, self.active_color, -1)

        self.prev_point = curr_point

    def blend_with_camera(self, camera_frame, alpha=0.7):
        """
        Blends active drawing canvas with camera feed.
        """
        # Resize camera if needed
        if camera_frame.shape[:2] != (self.height, self.width):
            camera_frame = cv2.resize(camera_frame, (self.width, self.height))

        # Create gray mask of canvas strokes
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask_inv = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY_INV)

        # Blackout area of drawing in camera frame
        frame_bg = cv2.bitwise_and(camera_frame, camera_frame, mask=mask_inv)
        
        # Combine camera background with canvas drawing
        combined = cv2.add(frame_bg, self.canvas)
        return combined

    def get_png_bytes(self):
        """
        Returns PNG binary buffer for web download.
        """
        # Convert BGR canvas to RGB for PIL / Streamlit
        canvas_rgb = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(canvas_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()
