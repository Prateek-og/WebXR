import cv2
import numpy as np

# Color Definitions (BGR for OpenCV, Hex/RGB for Streamlit)
COLORS_BGR = {
    "Electric Cyan": (255, 229, 0),      # #00E5FF in RGB -> (255, 229, 0) BGR
    "Neon Purple": (249, 0, 213),        # #D500F9 -> (249, 0, 213) BGR
    "Neon Green": (118, 230, 0),         # #00E676 -> (118, 230, 0) BGR
    "Bright Yellow": (0, 234, 255),      # #FFEA00 -> (0, 234, 255) BGR
    "Bright Red": (68, 23, 255),         # #FF1744 -> (68, 23, 255) BGR
    "Pure White": (255, 255, 255),
    "Eraser": (0, 0, 0)
}

COLORS_HEX = {
    "Electric Cyan": "#00E5FF",
    "Neon Purple": "#D500F9",
    "Neon Green": "#00E676",
    "Bright Yellow": "#FFEA00",
    "Bright Red": "#FF1744",
    "Pure White": "#FFFFFF",
    "Eraser": "#121212"
}

GESTURE_EMOJIS = {
    "DRAW": ("✏️", "Drawing", (0, 230, 118)),       # Green badge
    "PAUSE": ("✋", "Paused", (128, 128, 128)),     # Gray badge
    "ERASER": ("🧹", "Eraser Mode", (68, 23, 255)),  # Red/Orange badge
    "CLEAR": ("✊", "Clear Canvas", (0, 140, 255)),   # Orange badge
    "SELECT": ("✌️", "Hover / Select", (255, 229, 0)),# Cyan badge
    "SAVE": ("👍", "Save Artwork", (249, 0, 213))   # Purple badge
}

def smooth_point(prev_point, curr_point, alpha=0.6):
    """
    Applies Exponential Moving Average (EMA) to smooth out finger tracking coordinates.
    """
    if prev_point is None:
        return curr_point
    
    smoothed_x = int(alpha * curr_point[0] + (1 - alpha) * prev_point[0])
    smoothed_y = int(alpha * curr_point[1] + (1 - alpha) * prev_point[1])
    return (smoothed_x, smoothed_y)

def draw_status_badge(frame, gesture_key, active_color_bgr=(255, 229, 0), brush_size=5):
    """
    Draws a sleek modern status badge at top-left of the camera frame.
    """
    emoji, text, badge_color = GESTURE_EMOJIS.get(gesture_key, ("❓", "Unknown", (100, 100, 100)))
    
    # Background card parameters
    cv2.rectangle(frame, (15, 15), (280, 75), (20, 20, 20), -1)
    cv2.rectangle(frame, (15, 15), (280, 75), (50, 50, 50), 1)
    
    # Status indicator circle
    cv2.circle(frame, (35, 45), 8, badge_color, -1)
    
    # Status text
    status_str = f"Status: {text}"
    cv2.putText(frame, status_str, (55, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)
    
    # Active color preview dot & brush size
    cv2.circle(frame, (250, 45), brush_size // 2 + 3, active_color_bgr, -1)
    cv2.circle(frame, (250, 45), brush_size // 2 + 4, (200, 200, 200), 1)

    return frame
