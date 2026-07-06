# tracker.py
# Runs YOLO detection and ByteTrack on every frame.
# Returns a clean list of tracked vehicles for the rest of the pipeline.

from ultralytics import YOLO
import config

# Load the YOLO model once at import time
model = YOLO(config.MODEL_PATH)


def get_tracks(frame):
    """
    Run YOLO + ByteTrack on a single frame.

    Returns a list of dicts, one per tracked vehicle:
        {
            "track_id" : int,
            "class_id" : int,
            "label"    : str,   e.g. "Car"
            "box"      : (x1, y1, x2, y2)   – pixel coords
            "cx"       : int,   centre-x
            "cy"       : int,   centre-y
        }
    Only vehicles whose class_id is in config.VEHICLE_CLASSES are returned.
    """
    results = model.track(
        frame,
        persist=True,          # keeps ByteTrack state across frames
        tracker="bytetrack.yaml",
        conf=min(config.CLASS_CONF.values()),  # let YOLO pass all candidates;
                                               # we apply per-class filter below
        verbose=False,
    )

    vehicles = []

    # results[0].boxes contains all detections for this frame
    boxes = results[0].boxes
    if boxes is None or boxes.id is None:
        return vehicles   # no detections / no tracks yet

    for box in boxes:
        class_id = int(box.cls[0])

        # Skip classes we don't care about
        if class_id not in config.VEHICLE_CLASSES:
            continue

        # Apply per-class confidence threshold
        conf      = float(box.conf[0])
        threshold = config.CLASS_CONF.get(class_id, config.DEFAULT_CONF)
        if conf < threshold:
            continue

        track_id        = int(box.id[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
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


def draw_boxes(frame, vehicles, extra_info: dict = None):
    """
    Draw bounding box + label on the frame for every tracked vehicle.

    extra_info: optional dict keyed by track_id that can supply
                additional text (speed, WRONG SIDE, etc.) to show
                beneath the main label.  Built by other modules and
                passed in here so drawing stays in one place.
    """
    import cv2

    if extra_info is None:
        extra_info = {}

    for v in vehicles:
        track_id = v["track_id"]
        x1, y1, x2, y2 = v["box"]

        # Pick box colour: red if overspeed, else green
        info      = extra_info.get(track_id, {})
        overspeed = info.get("overspeed", False)
        color     = config.COLOR_OVERSPEED if overspeed else config.COLOR_BOX

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.THICKNESS)

        # Main label:  "Car #4"
        main_label = f"{v['label']} #{track_id}"
        cv2.putText(frame, main_label,
                    (x1, y1 - 8),
                    config.FONT, config.FONT_SCALE,
                    color, config.THICKNESS)

        # Extra info lines (speed, overspeed)
        lines = info.get("lines", [])
        for i, line_text in enumerate(lines):
            line_color = config.COLOR_OVERSPEED if "OVERSPEED" in line_text \
                    else config.COLOR_TEXT
            cv2.putText(frame, line_text,
                        (x1, y2 + 16 + i * 16),
                        config.FONT, config.FONT_SCALE,
                        line_color, config.THICKNESS)

    return frame
