# config.py
# Central configuration for TrafficSurveyAI.
# Edit ONLY this file to calibrate the system for your camera and road.

# =============================================================================
# MODEL
# =============================================================================
MODEL_PATH = "yolo11n.pt"   # swap to yolo11m.pt for better accuracy

# =============================================================================
# INPUT / OUTPUT
# =============================================================================
VIDEO_PATH  = "videos/Traffic.mp4"
OUTPUT_PATH = "output/annotated_output.mp4"

# =============================================================================
# VEHICLE CLASSES  (COCO IDs)
# =============================================================================
# 2=car  3=motorcycle  5=bus  7=truck
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

# =============================================================================
# COUNTING LINE
# =============================================================================
# LINE_ORIENTATION : "horizontal"  →  a line across the frame (constant Y)
#                    "vertical"    →  a line down the frame   (constant X)
LINE_ORIENTATION = "horizontal"

# Position of the line in pixels.
#   horizontal → COUNT_LINE_Y is used
#   vertical   → COUNT_LINE_X is used
COUNT_LINE_Y = 720    # Y pixel (for horizontal line) — 720×1280 video
COUNT_LINE_X = 360    # X pixel (for vertical   line)

# Directional counting:
#   "both"  → count vehicles crossing in either direction
#   "down"  → count only vehicles moving top→bottom  (delta Y > 0)
#   "up"    → count only vehicles moving bottom→top  (delta Y < 0)
#   "right" → count only vehicles moving left→right  (delta X > 0)
#   "left"  → count only vehicles moving right→left  (delta X < 0)
COUNT_DIRECTION = "both"

# =============================================================================
# PERSPECTIVE / HOMOGRAPHY CALIBRATION
# =============================================================================
# Four points in the IMAGE that form a known rectangle on the road.
# Mark them in order:  top-left, top-right, bottom-right, bottom-left.
# Use a calibration tool or pause the video and note pixel coordinates.
#
# Default values are approximate for the 720×1280 Traffic.mp4 video.
# Adjust these to match YOUR video frame.
PERSPECTIVE_SRC = [
    [220, 500],   # top-left     of the road region
    [500, 500],   # top-right    of the road region
    [620, 900],   # bottom-right of the road region
    [100, 900],   # bottom-left  of the road region
]

# The real-world width and height (metres) of the rectangle above.
PERSPECTIVE_REAL_WIDTH_M  = 14.0   # road width  in metres
PERSPECTIVE_REAL_HEIGHT_M = 20.0   # road length in metres

# =============================================================================
# SPEED ESTIMATION
# =============================================================================
# Smoothing: number of recent per-frame speed samples to average.
SPEED_SMOOTH_WINDOW = 8

# Minimum pixel displacement per frame to be considered moving.
# Below this → vehicle is stationary, speed reported as 0.
MIN_MOVE_PX = 3

# Hard cap: any instantaneous speed above this is treated as a spike and
# discarded.  Prevents tracker ID-jumps from inflating speed readings.
MAX_REALISTIC_KMH = 130

# Speed limit for this road (km/h).  Vehicles above this show OVERSPEED.
SPEED_LIMIT_KMH = 60

# Minimum number of trajectory frames before speed is displayed.
# Avoids showing garbage values on the first couple of frames.
MIN_TRAJECTORY_LEN = 4

# =============================================================================
# CONFIDENCE THRESHOLDS  (per class)
# =============================================================================
# Lower value = more detections (higher recall, more false positives).
# Higher value = fewer detections (higher precision, may miss vehicles).
#
# Motorcycles are small and often partially occluded, so they get a lower
# threshold to ensure they are never missed.
# Adjust these values if you see too many false positives or missed vehicles.
CLASS_CONF = {
    2: 0.35,   # Car
    3: 0.15,   # Motorcycle  ← intentionally low so none are missed
    5: 0.35,   # Bus
    7: 0.35,   # Truck
}

# Default threshold for any class not listed above
DEFAULT_CONF = 0.35
# If the video FPS metadata is unreliable, force a fixed value here.
# Set to None to read FPS automatically from the video file.
FORCE_FPS = None   # e.g.  30.0

# =============================================================================
# DISPLAY
# =============================================================================
FONT        = 0      # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.52
THICKNESS   = 2

# Draw vehicle trajectory trail (last N centre points)
DRAW_TRAIL      = True
TRAIL_LENGTH    = 25   # number of past positions to draw

# Colours (BGR)
COLOR_BOX       = (0, 255, 0)      # green  – normal box
COLOR_OVERSPEED = (0, 60, 255)     # red    – overspeed box
COLOR_LINE      = (255, 0, 255)    # magenta – counting line
COLOR_TRAIL     = (0, 215, 255)    # amber  – trajectory trail
COLOR_DOT       = (0, 255, 255)    # yellow – centre dot
COLOR_TEXT      = (255, 255, 255)  # white  – labels
COLOR_HUD_BG    = (30, 30, 30)     # dark   – HUD background
