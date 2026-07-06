# config.py
# Central configuration for TrafficSurveyAI — Government Traffic Survey System.
# This is the ONLY file you need to edit to calibrate the system.
# Every tunable parameter lives here. Nothing is hardcoded anywhere else.

# =============================================================================
# MODEL
# =============================================================================
# yolo11x.pt = extra-large model, highest accuracy.
# This is an offline survey — accuracy over speed.
MODEL_PATH = "yolo11x.pt"

# =============================================================================
# INPUT / OUTPUT
# =============================================================================
VIDEO_PATH  = "videos/Traffic.mp4"
OUTPUT_PATH = "output/annotated_output.mp4"

# =============================================================================
# VEHICLE CLASSES  (COCO class IDs)
# =============================================================================
# Only these four classes are processed. Everything else is ignored.
# 2=car  3=motorcycle  5=bus  7=truck
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

# =============================================================================
# DETECTION SETTINGS
# =============================================================================
# imgsz: inference resolution. 1280 is mandatory for small motorcycles and
#        far vehicles in Indian dense traffic. Do not reduce this.
IMGSZ = 1280

# Per-class confidence thresholds.
# Motorcycles get a very low threshold because they are small, often occluded
# by other vehicles, and critical to count in Indian traffic surveys.
# Raise Car/Bus/Truck if you see too many false positives on those classes.
CLASS_CONF = {
    2: 0.30,   # Car
    3: 0.08,   # Motorcycle — intentionally low; catch every bike
    5: 0.30,   # Bus
    7: 0.30,   # Truck
}
DEFAULT_CONF = 0.30   # fallback for any unlisted class

# IoU threshold for Non-Maximum Suppression.
# 0.40 is looser than default (0.45–0.7) to prevent motorcycle boxes from
# being suppressed when they overlap with a car or truck box beside them.
IOU_THRESH = 0.40

# agnostic_nms = True: NMS runs across ALL classes together.
# This prevents a high-confidence Car box from eating a Motorcycle box
# that partially overlaps it in the same region.
AGNOSTIC_NMS = True

# ByteTrack tuning is in bytetrack_custom.yaml (project root).
# Edit that file to adjust track_buffer, match_thresh, etc.

# =============================================================================
# COUNTING LINE
# =============================================================================
# LINE_ORIENTATION: "horizontal" = constant Y line across the frame
#                   "vertical"   = constant X line down the frame
LINE_ORIENTATION = "horizontal"

COUNT_LINE_Y = 720   # Y pixel — adjust to sit clearly across all lanes
COUNT_LINE_X = 360   # X pixel — used only for vertical orientation

# Directional counting filter:
#   "both"  = count vehicles crossing in either direction
#   "down"  = count only top→bottom movement (delta Y > 0)
#   "up"    = count only bottom→top movement (delta Y < 0)
#   "right" = count only left→right movement (delta X > 0)
#   "left"  = count only right→left movement (delta X < 0)
COUNT_DIRECTION = "both"

# =============================================================================
# PERSPECTIVE / HOMOGRAPHY CALIBRATION
# =============================================================================
# Define four points on the road surface in IMAGE pixel coordinates.
# These four points must form a rectangle in the REAL WORLD (bird's eye view).
# Good reference: lane markings, road edges, or painted boxes.
#
# Order: top-left, top-right, bottom-right, bottom-left
#
# HOW TO CALIBRATE:
#   1. Pause the video on a clear frame.
#   2. Find a known rectangle on the road (e.g. two lane lines × two distances).
#   3. Note the pixel coordinates of its four corners.
#   4. Measure the real-world width and height of that rectangle in metres.
#   5. Enter them below.
#
# Default values are approximate for the 720×1280 Traffic.mp4 video.
PERSPECTIVE_SRC = [
    [220, 500],   # top-left
    [500, 500],   # top-right
    [620, 900],   # bottom-right
    [100, 900],   # bottom-left
]

# Real-world dimensions of the rectangle defined above (metres).
PERSPECTIVE_REAL_WIDTH_M  = 14.0   # road width  (left edge to right edge)
PERSPECTIVE_REAL_HEIGHT_M = 20.0   # road length (top line to bottom line)

# =============================================================================
# SPEED ESTIMATION
# =============================================================================
# FPS is always read automatically from the video file.
# Set FORCE_FPS only if the video's metadata FPS is known to be wrong.
FORCE_FPS = None

# Smoothing window: average speed over this many consecutive frames.
# 15 frames at 30 FPS = 0.5 seconds of smoothing. Good balance.
# Increase to 20–25 for very noisy or shaky footage.
SPEED_SMOOTH_WINDOW = 15

# Minimum pixel displacement per frame to be treated as movement.
# Below this threshold, the vehicle is considered stationary (speed = 0).
# Prevents tiny tracker jitter from generating fake speed readings.
MIN_MOVE_PX = 4

# Hard spike rejection cap.
# Any single-frame speed above this value is discarded as physically impossible.
# 150 km/h is a safe upper bound for urban Indian roads.
MAX_REALISTIC_KMH = 150

# Speed limit for this road (km/h). Vehicles above this are flagged OVERSPEED.
SPEED_LIMIT_KMH = 60

# Minimum trajectory length before speed is shown on screen.
# Avoids displaying garbage values from the first 1–2 detections.
MIN_TRAJECTORY_LEN = 6

# =============================================================================
# CAMERA / PROCESSING
# =============================================================================
# FPS is always read automatically from the video file.
# Set FORCE_FPS only if the video's metadata FPS is known to be wrong.
FORCE_FPS = None

# Show live preview window while processing.
# False = headless mode — no window, fastest processing, recommended for surveys.
# True  = show annotated frames in real time (slower due to imshow overhead).
SHOW_PREVIEW = True

# =============================================================================
# DISPLAY
# =============================================================================
FONT       = 0      # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.50
THICKNESS  = 2

DRAW_TRAIL   = True
TRAIL_LENGTH = 30   # number of past centre points to draw per vehicle

# Colours (BGR)
COLOR_BOX       = (0, 220, 0)      # green  — normal bounding box
COLOR_OVERSPEED = (0, 0, 255)      # red    — overspeed vehicle
COLOR_LINE      = (255, 0, 255)    # magenta — counting line
COLOR_TRAIL     = (0, 200, 255)    # amber  — trajectory trail
COLOR_DOT       = (0, 255, 255)    # yellow — centre dot
COLOR_TEXT      = (255, 255, 255)  # white  — all text labels
COLOR_HUD_BG    = (20, 20, 20)     # near-black — HUD background panel
