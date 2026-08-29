import cv2
import os
import urllib.request
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks.python import vision

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12),
    (9,13), (13,14), (14,15), (15,16),
    (13,17), (0,17), (17,18), (18,19), (19,20)
]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hand_landmarker.task")

def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading hand_landmarker.task model file...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully!")

class HandTracker:
    def __init__(self, max_hands=1, detection_con=0.7, track_con=0.7):
        ensure_model_exists()
        
        base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_con,
            min_hand_presence_confidence=track_con
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.latest_result = None
        self._frame_timestamp_ms = 0
        
        # Landmark indices reference
        self.INDEX_TIP = 8
        self.INDEX_PIP = 6
        self.MIDDLE_TIP = 12
        self.MIDDLE_PIP = 10
        self.RING_TIP = 16
        self.RING_PIP = 14
        self.PINKY_TIP = 20
        self.PINKY_PIP = 18
        self.THUMB_TIP = 4
        self.THUMB_IP = 3

    def find_hands(self, frame, draw=True):
        """
        Processes BGR image, detects hands, and draws connections if requested.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        self._frame_timestamp_ms += 33
        self.latest_result = self.detector.detect_for_video(mp_image, self._frame_timestamp_ms)
        
        if draw and self.latest_result and self.latest_result.hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in self.latest_result.hand_landmarks:
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                
                # Draw Connections
                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(frame, points[start_idx], points[end_idx], (0, 229, 255), 2)
                    
                # Draw Landmarks
                for p in points:
                    cv2.circle(frame, p, 4, (255, 0, 213), -1)
                    
        return frame

    def find_positions(self, frame):
        """
        Returns list of (id, x, y) pixel coordinates for detected hand landmarks.
        """
        lm_list = []
        if self.latest_result and self.latest_result.hand_landmarks and len(self.latest_result.hand_landmarks) > 0:
            hand_lms = self.latest_result.hand_landmarks[0]
            h, w, _ = frame.shape
            for lm_id, lm in enumerate(hand_lms):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((lm_id, cx, cy))
        return lm_list
