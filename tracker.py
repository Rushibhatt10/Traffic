# tracker.py
# Vehicle Detection + ByteTrack Tracking
# Production quality — optimised for Indian dense traffic, offline survey.
#
# Design decisions:
#   • imgsz=1280      : full native resolution — mandatory for small motorcycles
#                       and far vehicles. Never reduce this for accuracy.
#   • yolo11x         : largest YOLO model, best small-object detection.
#   • per-class conf  : motorcycles get conf=0.08 so no bike is ever missed.
#   • agnostic_nms    : NMS across all classes — stops car boxes suppressing
#                       motorcycle boxes that partially overlap them.
#   • iou=0.40        : looser NMS so two nearby bikes aren't merged into one.
#   • track_buffer=60 : ByteTrack keeps a lost ID alive for 60 frames (~2 sec).
#                       This recovers the SAME track ID after occlusion.
#   • Every frame is processed. No frame skipping.

import cv2
import torch
from ultralytics import YOLO
import config

# ── CUDA / CPU auto-selection ─────────────────────────────────────────────────
# Use GPU if available — significantly faster inference.
# Fall back to CPU otherwise.
# FP16 (half precision) is only supported on CUDA; CPU must use FP32.
_DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
_USE_HALF = torch.cuda.is_available()   # True on GPU, False on CPU

# On CPU: maximise thread usage for parallel inference
if _DEVICE == "cpu":
    torch.set_num_threads(torch.get_num_threads())

# ── Load model ONCE at startup ────────────────────────────────────────────────
# Never reload the model inside the frame loop — loading takes 2–3 seconds.
print(f"[Tracker] Loading model : {config.MODEL_PATH}")
print(f"[Tracker] Device        : {_DEVICE.upper()}"
      + (" (FP16 enabled)" if _USE_HALF else " (FP32)"))
model = YOLO(config.MODEL_PATH)
print(f"[Tracker] Model ready.")

# Custom ByteTrack config (project root) — tuned for dense Indian traffic
_TRACKER_CFG = "bytetrack_custom.yaml"


# =============================================================================
# DETECTION + TRACKING
# =============================================================================

def get_tracks(frame) -> list[dict]:
    """
    Run YOLO11x inference + ByteTrack on a single frame.
    Every frame must be passed — no skipping.

    Returns a list of vehicle dicts:
        {
            "track_id" : int,           unique ByteTrack ID
            "class_id" : int,           COCO class ID
            "label"    : str,           "Car" / "Motorcycle" / "Bus" / "Truck"
            "box"      : (x1,y1,x2,y2) bounding box in pixels
            "cx"       : int,           box centre X
            "cy"       : int,           box centre Y
            "conf"     : float,         detection confidence
        }
    """
    # ── YOLO inference ────────────────────────────────────────────────────────
    # conf = minimum threshold passed to YOLO. We set it to the LOWEST
    # per-class threshold (motorcycle=0.08) so YOLO doesn't discard low-conf
    # motorcycles before we even see them. We then apply per-class filtering
    # ourselves below.
    #
    # persist=True   : tells Ultralytics to keep ByteTrack state between calls.
    #                  Without this, every frame would start a fresh tracker.
    # agnostic_nms   : NMS runs across all classes — crucial so a Car detection
    #                  doesn't suppress a Motorcycle box that overlaps it.
    results = model.track(
        frame,
        persist      = True,          # keep ByteTrack state between frames
        tracker      = _TRACKER_CFG,
        imgsz        = config.IMGSZ,
        conf         = min(config.CLASS_CONF.values()),  # lowest = 0.08 (motorcycle)
        iou          = config.IOU_THRESH,
        agnostic_nms = config.AGNOSTIC_NMS,
        device       = _DEVICE,       # GPU if available, else CPU
        half         = _USE_HALF,     # FP16 on GPU for faster inference
        verbose      = False,
    )

    vehicles = []
    boxes = results[0].boxes

    # No detections or ByteTrack hasn't assigned IDs yet
    if boxes is None or boxes.id is None:
        return vehicles

    for box in boxes:
        class_id = int(box.cls[0])

        # ── Filter: only process the four vehicle classes ──────────────────
        if class_id not in config.VEHICLE_CLASSES:
            continue

        conf = float(box.conf[0])

        # ── Per-class confidence gate ──────────────────────────────────────
        # This is the second filter. YOLO passed everything above 0.08.
        # Now we apply each class's own threshold.
        # Cars/Buses/Trucks need conf ≥ 0.30 to avoid false positives.
        # Motorcycles only need conf ≥ 0.08 to catch every bike.
        threshold = config.CLASS_CONF.get(class_id, config.DEFAULT_CONF)
        if conf < threshold:
            continue

        track_id        = int(box.id[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Centre point of the bounding box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        vehicles.append({
            "track_id" : track_id,
            "class_id" : class_id,
            "label"    : config.VEHICLE_CLASSES[class_id],
            "box"      : (x1, y1, x2, y2),
            "cx"       : cx,
            "cy"       : cy,
            "conf"     : conf,
        })

    return vehicles


# =============================================================================
# DRAWING
# =============================================================================

def draw_boxes(frame, vehicles: list, extra_info: dict = None) -> None:
    """
    Draw bounding boxes and all annotation text on the frame.

    extra_info: dict keyed by track_id, built by speed_estimator.
        {
            track_id: {
                "overspeed": bool,
                "lines"    : ["45.2 km/h", "Avg:43.1 Max:52.0", "!! OVERSPEED"]
            }
        }
    """
    if extra_info is None:
        extra_info = {}

    for v in vehicles:
        tid             = v["track_id"]
        x1, y1, x2, y2 = v["box"]
        info            = extra_info.get(tid, {})
        overspeed       = info.get("overspeed", False)

        # Red box for overspeed vehicles, green for normal
        box_color = config.COLOR_OVERSPEED if overspeed else config.COLOR_BOX

        # ── Bounding box ──────────────────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, config.THICKNESS)

        # ── Centre dot ────────────────────────────────────────────────────
        cv2.circle(frame, (v["cx"], v["cy"]), 3, config.COLOR_DOT, -1)

        # ── Main label: "Motorcycle #7  0.62" ─────────────────────────────
        label = f"{v['label']} #{tid}  {v['conf']:.2f}"
        cv2.putText(frame, label,
                    (x1, y1 - 6),
                    config.FONT, config.FONT_SCALE,
                    box_color, config.THICKNESS)

        # ── Extra lines (speed, avg, max, OVERSPEED) ──────────────────────
        lines = info.get("lines", [])
        for i, line_text in enumerate(lines):
            color = config.COLOR_OVERSPEED if "OVERSPEED" in line_text \
                    else config.COLOR_TEXT
            cv2.putText(frame, line_text,
                        (x1, y2 + 15 + i * 17),
                        config.FONT, config.FONT_SCALE,
                        color, config.THICKNESS)


def draw_frame_info(frame, frame_no: int, fps: float,
                    vehicle_count: int) -> None:
    """
    Draw a small top-right HUD showing:
        Frame number
        Processing FPS (how fast frames are being processed)
        Number of vehicles currently visible
    """
    h, w = frame.shape[:2]
    lines = [
        f"Frame : {frame_no}",
        f"FPS   : {fps:.1f}",
        f"Visible: {vehicle_count}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text,
                    (w - 160, 20 + i * 20),
                    config.FONT, 0.48,
                    config.COLOR_TEXT, 1)
