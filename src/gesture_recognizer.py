import math

class GestureRecognizer:
    def __init__(self, pinch_threshold=40):
        self.pinch_threshold = pinch_threshold

    def get_finger_states(self, lm_list):
        """
        Determines which fingers are extended (True) or folded (False).
        lm_list format: [(id, x, y), ...]
        Returns dict: {'thumb': bool, 'index': bool, 'middle': bool, 'ring': bool, 'pinky': bool}
        """
        if len(lm_list) < 21:
            return None

        # Y-coordinates comparison (In image coordinates, y=0 is TOP, so y_tip < y_pip means UP)
        thumb_up = lm_list[4][1] < lm_list[3][1] if lm_list[4][0] < lm_list[17][0] else lm_list[4][1] > lm_list[3][1] # X check for thumb orientation
        index_up = lm_list[8][2] < lm_list[6][2]
        middle_up = lm_list[12][2] < lm_list[10][2]
        ring_up = lm_list[16][2] < lm_list[14][2]
        pinky_up = lm_list[20][2] < lm_list[18][2]

        return {
            "thumb": thumb_up,
            "index": index_up,
            "middle": middle_up,
            "ring": ring_up,
            "pinky": pinky_up
        }

    def calculate_distance(self, p1, p2):
        """
        Euclidean distance between two 2D points p1(x, y) and p2(x, y).
        """
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def detect_gesture(self, lm_list):
        """
        Recognizes active gesture state based on landmark coordinates.
        Returns gesture key: 'DRAW', 'PAUSE', 'ERASER', 'SELECT', 'CLEAR', 'SAVE', or 'NONE'
        """
        if not lm_list or len(lm_list) < 21:
            return "NONE"

        states = self.get_finger_states(lm_list)
        if not states:
            return "NONE"

        index_pos = (lm_list[8][1], lm_list[8][2])
        thumb_pos = (lm_list[4][1], lm_list[4][2])
        pinch_dist = self.calculate_distance(index_pos, thumb_pos)

        # Pinch detection (Index tip close to Thumb tip) -> Eraser
        if pinch_dist < self.pinch_threshold:
            return "ERASER"

        # 1. Open Palm (All 4 fingers up) -> Pause
        if states["index"] and states["middle"] and states["ring"] and states["pinky"]:
            return "PAUSE"

        # 2. Index + Middle up -> Hover / Select mode
        if states["index"] and states["middle"] and not states["ring"] and not states["pinky"]:
            return "SELECT"

        # 3. Only Index up -> Draw mode
        if states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]:
            return "DRAW"

        # 4. Fist (All fingers down) -> Clear Trigger
        if not states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]:
            return "CLEAR"

        # 5. Thumbs up -> Save trigger
        if states["thumb"] and not states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]:
            return "SAVE"

        # Default fallback if hand detected but gesture doesn't match single rule
        return "PAUSE"
