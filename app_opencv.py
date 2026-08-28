import cv2
import numpy as np
import time
from src.hand_tracker import HandTracker
from src.gesture_recognizer import GestureRecognizer
from src.canvas_engine import CanvasEngine
from src.utils import COLORS_BGR, smooth_point, draw_status_badge

def main():
    cap = cv2.VideoCapture(0)
    
    # Try setting 1280x720 resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Read first frame to get actual dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not access camera.")
        return

    h, w, _ = frame.shape
    
    # Initialize Core Engines
    tracker = HandTracker(max_hands=1, detection_con=0.75, track_con=0.75)
    recognizer = GestureRecognizer()
    canvas = CanvasEngine(width=w, height=h)
    
    smoothed_index_pos = None
    color_keys = list(COLORS_BGR.keys())[:-1] # Exclude Eraser key
    current_color_idx = 0
    canvas.set_color(COLORS_BGR[color_keys[current_color_idx]])
    
    print("=" * 60)
    print(" ✏️  AIRDRAW — Desktop OpenCV Mode Started")
    print("=" * 60)
    print(" Controls:")
    print("  - ☝️ Index Up      : Draw")
    print("  - ✋ Open Palm     : Pause Drawing")
    print("  - 🤏 Pinch         : Eraser Mode")
    print("  - ✌️ Two Fingers   : Hover Mode")
    print("  - ✊ Fist          : Clear Canvas")
    print(" Shortcuts:")
    print("  - 'c' : Clear Canvas")
    print("  - 'u' : Undo")
    print("  - 's' : Save Artwork to airdraw_output.png")
    print("  - 'e' : Toggle Eraser")
    print("  - '1-6': Select Color")
    print("  - '+' / '-': Change Brush Size")
    print("  - 'q' : Quit")
    print("=" * 60)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for intuitive mirror action
        frame = cv2.flip(frame, 1)
        
        # 1. Process Hand Tracking
        processed_frame = tracker.find_hands(frame.copy(), draw=True)
        lm_list = tracker.find_positions(processed_frame)
        
        # 2. Gesture Recognition
        gesture = recognizer.detect_gesture(lm_list)
        
        # 3. Canvas Action Logic
        if lm_list and len(lm_list) >= 21:
            raw_index_pos = (lm_list[8][1], lm_list[8][2])
            smoothed_index_pos = smooth_point(smoothed_index_pos, raw_index_pos, alpha=0.65)
            
            if gesture == "DRAW":
                canvas.enable_eraser(False)
                canvas.start_stroke()
                canvas.draw_line(smoothed_index_pos)
                cv2.circle(processed_frame, smoothed_index_pos, canvas.brush_size, canvas.active_color, -1)
                
            elif gesture == "ERASER":
                canvas.enable_eraser(True)
                canvas.start_stroke()
                canvas.draw_line(smoothed_index_pos)
                cv2.circle(processed_frame, smoothed_index_pos, canvas.eraser_size, (50, 50, 255), 2)
                
            elif gesture == "CLEAR":
                canvas.clear()
                canvas.end_stroke()
                smoothed_index_pos = None
                
            else:
                canvas.end_stroke()
                # Draw pointer dot when in hover/pause
                cv2.circle(processed_frame, smoothed_index_pos, 5, (200, 200, 200), -1)
        else:
            canvas.end_stroke()
            smoothed_index_pos = None

        # 4. Generate Views
        # Main Display: Camera Feed blended with Drawing Canvas
        output_display = canvas.blend_with_camera(processed_frame, alpha=0.85)
        
        # Draw floating webcam PIP preview in upper right corner (width 240px)
        pip_w, pip_h = 240, 135
        pip_frame = cv2.resize(processed_frame, (pip_w, pip_h))
        output_display[15:15+pip_h, w-15-pip_w:w-15] = pip_frame
        cv2.rectangle(output_display, (w-15-pip_w, 15), (w-15, 15+pip_h), (0, 229, 255), 1)

        # Draw Status Badge top left
        output_display = draw_status_badge(
            output_display, 
            gesture, 
            active_color_bgr=canvas.active_color, 
            brush_size=canvas.brush_size
        )

        # Calculate FPS
        curr_time = time.time()
        fps = int(1.0 / (curr_time - prev_time + 1e-6))
        prev_time = curr_time
        cv2.putText(output_display, f"FPS: {fps}", (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        # Bottom Shortcut Legend overlay
        legend_str = "[c]Clear  [u]Undo  [s]Save  [e]Eraser  [1-6]Color  [+/-]Size  [q]Quit"
        cv2.putText(output_display, legend_str, (120, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        # Show Output Window
        cv2.imshow("AirDraw — AR Air Drawing Platform", output_display)

        # 5. Keyboard Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # q or ESC
            break
        elif key == ord('c'):
            canvas.clear()
        elif key == ord('u'):
            canvas.undo()
        elif key == ord('e'):
            canvas.enable_eraser(not canvas.is_erasing)
        elif key == ord('s'):
            cv2.imwrite("airdraw_output.png", canvas.canvas)
            print("💾 Saved drawing to airdraw_output.png")
        elif key == ord('+') or key == ord('='):
            canvas.set_brush_size(canvas.brush_size + 2)
        elif key == ord('-') or key == ord('_'):
            canvas.set_brush_size(canvas.brush_size - 2)
        elif ord('1') <= key <= ord('6'):
            idx = key - ord('1')
            if idx < len(color_keys):
                color_name = color_keys[idx]
                canvas.set_color(COLORS_BGR[color_name])

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
