import os
import sys

# --- Streamlit Cloud / MediaPipe Hack ---
# MediaPipe forces the installation of 'opencv-contrib-python' (the GUI version), 
# which crashes on Streamlit Cloud due to missing system UI libraries.
# We intercept the ImportError and forcibly swap it for the headless version.
try:
    import cv2
except ImportError:
    print("OpenCV GUI version failed to import. Falling back to headless...")
    os.system(f"{sys.executable} -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless")
    os.system(f"{sys.executable} -m pip install opencv-contrib-python-headless==4.10.0.84")
    import cv2
# -----------------------------------------

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import cv2
import numpy as np
import time
from PIL import Image

from src.hand_tracker import HandTracker
from src.gesture_recognizer import GestureRecognizer
from src.canvas_engine import CanvasEngine
from src.utils import COLORS_BGR, COLORS_HEX, GESTURE_EMOJIS, smooth_point, draw_status_badge

# Page Configuration
st.set_page_config(
    page_title="AirDraw — AR Air Drawing Platform",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Charcoal Creative App UI
st.markdown("""
<style>
    /* Dark Theme Styles */
    .stApp {
        background-color: #121212;
        color: #EEEEEE;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #1E1E1E 0%, #252525 100%);
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .header-title {
        font-size: 28px;
        font-weight: 700;
        color: #00E5FF;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .header-subtitle {
        font-size: 14px;
        color: #AAAAAA;
        margin-top: 4px;
    }

    /* Status Badge */
    .badge-card {
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 10px 16px;
        border-left: 4px solid #00E5FF;
        font-size: 15px;
        font-weight: 600;
    }
    
    /* Control Section Cards */
    .control-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #2C2C2C;
        margin-bottom: 15px;
    }
    
    /* Color Swatch Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 229, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Shared Global Canvas Engine for WebRTC Frame Processing
if "canvas_engine" not in st.session_state:
    st.session_state.canvas_engine = CanvasEngine(width=1280, height=720)

if "active_color_name" not in st.session_state:
    st.session_state.active_color_name = "Electric Cyan"

if "brush_size" not in st.session_state:
    st.session_state.brush_size = 6

if "eraser_mode" not in st.session_state:
    st.session_state.eraser_mode = False


class AirDrawVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.tracker = HandTracker(max_hands=1, detection_con=0.75, track_con=0.75)
        self.recognizer = GestureRecognizer()
        self.smoothed_index_pos = None
        self.prev_time = time.time()
        self.last_gesture = "PAUSE"
        
        self.canvas_engine = None
        self.active_color_name = "Electric Cyan"
        self.brush_size = 6
        self.eraser_mode = False
        
        # Debounce timers for destructive gestures
        self.last_clear_time = 0
        self.last_save_time = 0
        self.save_triggered = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        
        # Mirror image for natural hand movement
        img_bgr = cv2.flip(img_bgr, 1)
        h, w, _ = img_bgr.shape
        
        if self.canvas_engine is None:
            return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")
            
        canvas_engine = self.canvas_engine
        canvas_engine.resize_if_needed(w, h)
        
        # Apply Session State Color & Brush settings
        bgr_color = COLORS_BGR.get(self.active_color_name, (255, 229, 0))
        canvas_engine.set_color(bgr_color)
        canvas_engine.set_brush_size(self.brush_size)
        canvas_engine.enable_eraser(self.eraser_mode)

        # Process Hand Detection
        processed_frame = self.tracker.find_hands(img_bgr.copy(), draw=True)
        lm_list = self.tracker.find_positions(processed_frame)
        
        # Detect Gesture
        gesture = self.recognizer.detect_gesture(lm_list)
        self.last_gesture = gesture

        # Drawing Logic
        if lm_list and len(lm_list) >= 21:
            raw_index_pos = (lm_list[8][1], lm_list[8][2])
            self.smoothed_index_pos = smooth_point(self.smoothed_index_pos, raw_index_pos, alpha=0.65)

            if gesture == "DRAW":
                canvas_engine.enable_eraser(False)
                canvas_engine.start_stroke()
                canvas_engine.draw_line(self.smoothed_index_pos)
                cv2.circle(processed_frame, self.smoothed_index_pos, canvas_engine.brush_size, canvas_engine.active_color, -1)

            elif gesture == "ERASER":
                canvas_engine.enable_eraser(True)
                canvas_engine.start_stroke()
                canvas_engine.draw_line(self.smoothed_index_pos)
                cv2.circle(processed_frame, self.smoothed_index_pos, canvas_engine.eraser_size, (50, 50, 255), 2)

            elif gesture == "CLEAR":
                now = time.time()
                if now - self.last_clear_time > 1.5:
                    canvas_engine.clear()
                    self.last_clear_time = now
                canvas_engine.end_stroke()
                self.smoothed_index_pos = None

            elif gesture == "SAVE":
                now = time.time()
                if now - self.last_save_time > 2.0:
                    self.save_triggered = True
                    self.last_save_time = now
                canvas_engine.end_stroke()
                # Draw visual save indicator on frame
                cv2.putText(processed_frame, "SAVED!", (w // 2 - 80, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 230, 118), 3, cv2.LINE_AA)

            else:
                canvas_engine.end_stroke()
                # Pointer circle
                cv2.circle(processed_frame, self.smoothed_index_pos, 5, (200, 200, 200), -1)
        else:
            canvas_engine.end_stroke()
            self.smoothed_index_pos = None

        # Blend drawing canvas over live camera feed
        output_display = canvas_engine.blend_with_camera(processed_frame, alpha=0.85)

        # Draw PIP Floating Preview of raw camera in top right
        pip_w, pip_h = 200, 115
        if w > pip_w + 30 and h > pip_h + 30:
            pip_frame = cv2.resize(processed_frame, (pip_w, pip_h))
            output_display[15:15+pip_h, w-15-pip_w:w-15] = pip_frame
            cv2.rectangle(output_display, (w-15-pip_w, 15), (w-15, 15+pip_h), (0, 229, 255), 1)

        # Draw status badge
        output_display = draw_status_badge(
            output_display, 
            gesture, 
            active_color_bgr=canvas_engine.active_color,
            brush_size=canvas_engine.brush_size
        )

        return av.VideoFrame.from_ndarray(output_display, format="bgr24")


def main():
    # Onboarding overlay on first load
    if "onboarding_shown" not in st.session_state:
        st.session_state.onboarding_shown = False
    
    if not st.session_state.onboarding_shown:
        with st.container():
            st.info(
                "**Welcome to AirDraw!** 👋\n\n"
                "Show **☝️ Index finger** to draw in the air · "
                "Show **✋ Open palm** to pause · "
                "**🤏 Pinch** to erase · "
                "**✊ Fist** to clear · "
                "**👍 Thumbs up** to save\n\n"
                "Click **START** on the video feed below to begin!"
            )
            if st.button("✅ Got it!", key="dismiss_onboarding"):
                st.session_state.onboarding_shown = True
                st.rerun()

    # Header Banner
    st.markdown("""
    <div class="header-card">
        <div class="header-title">✏️ AIRDRAW <span style="font-size:14px; background:#00E5FF22; color:#00E5FF; padding:4px 10px; border-radius:20px; border:1px solid #00E5FF44;">🟢 AR Camera Active</span></div>
        <div class="header-subtitle">Draw in the air with index finger tracking. No touch. No mouse. No physical pen.</div>
    </div>
    """, unsafe_allow_html=True)

    # Layout Columns: Sidebar controls + Main Workspace
    col_ctrl, col_canvas = st.columns([1, 2.5])

    with col_ctrl:
        st.markdown("### 🎨 Toolbar Controls")
        
        # Color Selection Swatches
        st.markdown("**Palette Selection**")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🩵 Cyan", use_container_width=True):
                st.session_state.active_color_name = "Electric Cyan"
                st.session_state.eraser_mode = False
            if st.button("🟡 Yellow", use_container_width=True):
                st.session_state.active_color_name = "Bright Yellow"
                st.session_state.eraser_mode = False
        with c2:
            if st.button("💜 Purple", use_container_width=True):
                st.session_state.active_color_name = "Neon Purple"
                st.session_state.eraser_mode = False
            if st.button("🔴 Red", use_container_width=True):
                st.session_state.active_color_name = "Bright Red"
                st.session_state.eraser_mode = False
        with c3:
            if st.button("💚 Green", use_container_width=True):
                st.session_state.active_color_name = "Neon Green"
                st.session_state.eraser_mode = False
            if st.button("🤍 White", use_container_width=True):
                st.session_state.active_color_name = "Pure White"
                st.session_state.eraser_mode = False

        st.divider()

        # Brush & Eraser Sliders
        st.session_state.brush_size = st.slider("🖌️ Brush Size", min_value=1, max_value=30, value=st.session_state.brush_size)
        
        # Eraser Toggle
        eraser_clicked = st.button(
            "🧹 Eraser Mode ON" if st.session_state.eraser_mode else "✏️ Brush Mode ON", 
            use_container_width=True,
            type="primary" if st.session_state.eraser_mode else "secondary"
        )
        if eraser_clicked:
            st.session_state.eraser_mode = not st.session_state.eraser_mode

        st.divider()

        # Canvas Action Buttons
        act_col1, act_col2 = st.columns(2)
        with act_col1:
            if st.button("↩️ Undo Stroke", use_container_width=True):
                st.session_state.canvas_engine.undo()
                st.toast("Undid last stroke!", icon="↩️")
        with act_col2:
            if st.button("🗑️ Clear Canvas", use_container_width=True):
                st.session_state.canvas_engine.clear()
                st.toast("Canvas cleared!", icon="🗑️")

        st.divider()

        # Download PNG Artwork Button
        png_bytes = st.session_state.canvas_engine.get_png_bytes()
        st.download_button(
            label="💾 Download Artwork (PNG)",
            data=png_bytes,
            file_name="airdraw_art.png",
            mime="image/png",
            use_container_width=True
        )

        # Onboarding / Gesture Cheat Sheet Expander
        with st.expander("❓ Gesture Cheat Sheet & Controls"):
            st.markdown("""
            | Gesture | Action |
            |---|---|
            | ☝️ **Index Up** | Draw stroke in air |
            | ✋ **Open Palm** | Pause / Stop drawing |
            | 🤏 **Pinch** | Eraser Mode |
            | ✌️ **Two Fingers** | Hover / Selection Mode |
            | ✊ **Fist** | Clear Canvas |
            | 👍 **Thumbs Up** | Save |
            """)

    with col_canvas:
        st.markdown("### 🖥️ Virtual Canvas & AR Video Feed")
        
        # WebRTC Live Streaming Component
        ctx = webrtc_streamer(
            key="airdraw",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_processor_factory=AirDrawVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

        if ctx.video_processor:
            ctx.video_processor.canvas_engine = st.session_state.canvas_engine
            ctx.video_processor.active_color_name = st.session_state.active_color_name
            ctx.video_processor.brush_size = st.session_state.brush_size
            ctx.video_processor.eraser_mode = st.session_state.eraser_mode
            
            # Check if SAVE gesture was triggered in the video thread
            if ctx.video_processor.save_triggered:
                ctx.video_processor.save_triggered = False
                st.toast("Artwork saved via 👍 gesture!", icon="💾")

if __name__ == "__main__":
    main()
