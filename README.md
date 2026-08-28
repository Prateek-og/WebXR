# AirDraw — AR / Air Drawing Platform ✏️✨

> "AirDraw is a computer-vision based interactive drawing platform that uses real-time hand and finger tracking to let users draw in the air using their webcam, with gesture-based controls, a distraction-free canvas-first UI, and downloadable digital artwork."

---

## 🌟 Key Features

- ☝️ **Touch-Free Air Drawing**: Track index finger movement in 3D space with MediaPipe and draw live strokes.
- 🎨 **Procreate/Figma-Inspired UI**: Dark charcoal (`#121212`) canvas-first theme with electric neon brush colors.
- 🛡️ **Smooth Motion Filtering**: Exponential Moving Average (EMA) coordinate smoothing to eliminate jittery lines.
- 🤖 **Real-Time Gesture Recognition**: Instant switching between Draw, Pause, Eraser, Clear Canvas, and Hover modes.
- ↩️ **Undo History Stack**: Multi-level undo stroke history.
- 💾 **Export Artwork**: Download high-resolution PNG artwork files directly from the app.
- 🚀 **Dual Application Modes**:
  - **Level 1 Desktop MVP**: High FPS local OpenCV native window (`app_opencv.py`).
  - **Level 2 Web App**: Streamlit web platform with WebRTC webcam streaming (`app.py`).

---

## ⚙️ Tech Stack

| Layer | Technology | Description |
|---|---|---|
| Core Logic | Python 3.10+ | Application logic |
| Computer Vision | OpenCV (`cv2`) | Frame processing, canvas blending, image export |
| Hand Tracking | MediaPipe Hands | 21-point hand landmark detection |
| Numerical Ops | NumPy | Matrix operations, canvas arrays |
| Web UI | Streamlit + `streamlit-webrtc` | Browser UI & real-time video streaming |

---

## 🖐️ Gesture Controls Cheat Sheet

| Gesture | Action | Description |
|---|---|---|
| ☝️ **Index Up** | **Draw** | Raise index finger to draw neon strokes |
| ✋ **Open Palm** | **Pause** | Show open palm to hover without drawing |
| 🤏 **Pinch** | **Eraser** | Pinch index & thumb to erase strokes |
| ✌️ **Two Fingers** | **Select / Hover** | Index & middle fingers raised |
| ✊ **Fist** | **Clear Canvas** | Fold all fingers into a fist to clear |
| 👍 **Thumbs Up** | **Save** | Trigger artwork export |

---

## 🚀 Getting Started

### 1. Installation

Clone repository and install requirements:

```bash
pip install -r requirements.txt
```

### 2. Launching Applications

#### Option A: Streamlit Web Application (Recommended for Web UI)

```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser and allow camera access.

#### Option B: Desktop Native OpenCV Application (High FPS Mode)

```bash
python app_opencv.py
```

##### Keyboard Shortcuts for Desktop OpenCV Mode:
- `c` : Clear Canvas
- `u` : Undo last stroke
- `s` : Save artwork to `airdraw_output.png`
- `e` : Toggle Eraser
- `1 - 6` : Select neon color palette
- `+` / `-` : Increase / Decrease brush size
- `q` / `ESC` : Quit

---

## 📂 Project Structure

```
Walll2/
├── app.py                     # Streamlit Web App (Level 2 Web UI)
├── app_opencv.py              # OpenCV Desktop Native App (Level 1 MVP)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .streamlit/
│   └── config.toml            # Dark charcoal theme config
└── src/
    ├── __init__.py
    ├── hand_tracker.py        # MediaPipe 21-landmark tracker
    ├── gesture_recognizer.py  # Rule-based gesture recognition engine
    ├── canvas_engine.py       # Virtual canvas & line smoothing engine
    └── utils.py               # Color palettes & status badge UI helpers
```
