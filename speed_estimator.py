# speed_estimator.py
# Professional Speed Estimation for TrafficSurveyAI.
#
# Algorithm:
#   1. Build a perspective homography matrix ONCE from four road calibration
#      points (config.PERSPECTIVE_SRC).  This maps pixels → real metres,
#      correcting for camera angle and perspective distortion.
#   2. Every frame, transform each vehicle centre to real-world metres.
#   3. Compute Euclidean distance moved since the last frame.
#   4. distance × FPS → m/s → km/h.
#   5. Smooth with a rolling average; reject spikes; ignore stationary noise.
#   6. Track current / average / maximum speed per vehicle.
#   7. When a vehicle disappears, log its final stats to the terminal.

import math
import cv2
import numpy as np
import config


# =============================================================================
# HOMOGRAPHY  — built once at import time
# =============================================================================

def _build_homography() -> np.ndarray:
    """
    Compute the perspective transform that maps image pixels to real-world
    metres.  Source = four road corners in pixels (config.PERSPECTIVE_SRC).
    Destination = an upright rectangle of the road's real dimensions.
    """
    src = np.float32(config.PERSPECTIVE_SRC)
    W   = config.PERSPECTIVE_REAL_WIDTH_M
    H   = config.PERSPECTIVE_REAL_HEIGHT_M

    dst = np.float32([
        [0, 0],   # top-left
        [W, 0],   # top-right
        [W, H],   # bottom-right
        [0, H],   # bottom-left
    ])

    matrix, _ = cv2.findHomography(src, dst)
    return matrix


_H = _build_homography()   # computed once, reused every frame


def _pixel_to_metres(cx: int, cy: int) -> tuple[float, float]:
    """Convert a pixel point to real-world (x, y) in metres via homography."""
    pt     = np.float32([[[cx, cy]]])
    result = cv2.perspectiveTransform(pt, _H)
    return float(result[0][0][0]), float(result[0][0][1])


# =============================================================================
# STATE  (module-level — persists across every frame)
# =============================================================================

# Real-world trajectory per vehicle: { tid: [(frame_no, rx, ry), ...] }
_trajectory: dict[int, list[tuple[int, float, float]]] = {}

# Rolling speed samples for smoothing: { tid: [kmh, kmh, ...] }
_speed_samples: dict[int, list[float]] = {}

# Speed stats per vehicle: { tid: {"current": f, "avg": f, "max": f} }
_speed_stats: dict[int, dict[str, float]] = {}

# Last known pixel position per vehicle (for pixel-movement check)
_prev_pixel_pos: dict[int, tuple[int, int]] = {}

# Pixel-space trail buffer for drawing: { tid: [(cx, cy), ...] }
_pixel_trail: dict[int, list[tuple[int, int]]] = {}

# Vehicle type label per id: { tid: "Car" }
_vehicle_labels: dict[int, str] = {}

# Track IDs visible in the previous frame (used for departure detection)
_prev_visible: set[int] = set()

# Log of every vehicle that has left the scene
departed_log: list[dict] = []

# Global frame counter
_frame_no: int = 0


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _instantaneous_speed(tid: int, fps: float) -> float | None:
    """
    Compute raw (unsmoothed) speed in km/h from the last two trajectory
    points.  Returns None if fewer than 2 points exist.
    """
    traj = _trajectory.get(tid, [])
    if len(traj) < 2:
        return None

    f1, rx1, ry1 = traj[-2]
    f2, rx2, ry2 = traj[-1]

    dist_m      = math.hypot(rx2 - rx1, ry2 - ry1)
    frame_delta = max(f2 - f1, 1)          # handles any dropped frames
    speed_mps   = dist_m * (fps / frame_delta)
    return speed_mps * 3.6


def _smooth_speed(tid: int, raw_kmh: float) -> float:
    """
    Append a speed sample, reject spikes, return the rolling-window average.
    Samples above MAX_REALISTIC_KMH are silently dropped.
    """
    samples = _speed_samples.setdefault(tid, [])

    if raw_kmh <= config.MAX_REALISTIC_KMH:
        samples.append(raw_kmh)

    if len(samples) > config.SPEED_SMOOTH_WINDOW:
        samples.pop(0)

    return sum(samples) / len(samples) if samples else 0.0


def _log_departure(tid: int) -> None:
    """
    Called when a track ID vanishes.  Prints a one-line summary to the
    terminal and appends the entry to departed_log.
    """
    stats = _speed_stats.get(tid)
    label = _vehicle_labels.get(tid, "Vehicle")

    if not stats or (stats["avg"] == 0 and stats["max"] == 0):
        return   # no meaningful data — skip

    entry = {
        "id"   : tid,
        "label": label,
        "avg"  : round(stats["avg"], 1),
        "max"  : round(stats["max"], 1),
    }
    departed_log.append(entry)

    flag = "  ⚠ OVERSPEED" if entry["max"] > config.SPEED_LIMIT_KMH else ""
    print(f"[Vehicle Log]  {label:<12} #{tid:<4} | "
          f"Avg: {entry['avg']:>5.1f} km/h | "
          f"Max: {entry['max']:>5.1f} km/h"
          f"{flag}")


# =============================================================================
# PUBLIC API
# =============================================================================

def update(vehicles: list, fps: float) -> dict[int, dict[str, float]]:
    """
    Process one frame of tracked vehicles.

    Steps:
      1. Detect departures (IDs gone since last frame) and log them.
      2. For each vehicle: transform pixels → metres, build trajectory,
         compute + smooth speed, update stats.

    Returns:
        _speed_stats  { tid: {"current": f, "avg": f, "max": f} }
    """
    global _frame_no
    _frame_no += 1

    # ── Departure detection ───────────────────────────────────────────────────
    current_ids = {v["track_id"] for v in vehicles}
    for tid in _prev_visible - current_ids:
        _log_departure(tid)
    _prev_visible.clear()
    _prev_visible.update(current_ids)

    # ── Per-vehicle speed update ──────────────────────────────────────────────
    for v in vehicles:
        tid  = v["track_id"]
        cx   = v["cx"]
        cy   = v["cy"]

        # Remember label for later logging
        _vehicle_labels[tid] = v["label"]

        # Transform pixel centre to real-world metres
        rx, ry = _pixel_to_metres(cx, cy)

        # Append to trajectory
        traj = _trajectory.setdefault(tid, [])
        traj.append((_frame_no, rx, ry))
        if len(traj) > 60:
            traj.pop(0)

        # Wait for enough points before estimating speed
        if len(traj) < config.MIN_TRAJECTORY_LEN:
            _prev_pixel_pos[tid] = (cx, cy)
            continue

        # Ignore stationary / jitter movement
        prev_cx, prev_cy = _prev_pixel_pos.get(tid, (cx, cy))
        pixel_move = math.hypot(cx - prev_cx, cy - prev_cy)
        _prev_pixel_pos[tid] = (cx, cy)

        stats = _speed_stats.setdefault(
            tid, {"current": 0.0, "avg": 0.0, "max": 0.0}
        )

        if pixel_move < config.MIN_MOVE_PX:
            stats["current"] = 0.0
            continue

        # Compute raw → smooth → update stats
        raw = _instantaneous_speed(tid, fps)
        if raw is None:
            continue

        smoothed         = _smooth_speed(tid, raw)
        stats["current"] = smoothed
        stats["max"]     = max(stats["max"], smoothed)

        samples      = _speed_samples.get(tid, [])
        stats["avg"] = sum(samples) / len(samples) if samples else smoothed

    return _speed_stats


def update_pixel_trail(vehicles: list) -> None:
    """
    Append each vehicle's current pixel centre to its trail buffer.
    Call this once per frame BEFORE draw_pixel_trails().
    """
    for v in vehicles:
        tid   = v["track_id"]
        trail = _pixel_trail.setdefault(tid, [])
        trail.append((v["cx"], v["cy"]))
        if len(trail) > config.TRAIL_LENGTH:
            trail.pop(0)


def build_extra_info(vehicles: list,
                     speed_stats: dict,
                     existing: dict = None) -> dict:
    """
    Build the extra_info dict consumed by tracker.draw_boxes().

    For each vehicle injects:
        Line 0 : "45.3 km/h"
        Line 1 : "Avg:43.1 Max:52.7"
        Line 2 : "!! OVERSPEED"   ← only when current > SPEED_LIMIT_KMH

    Merges with existing dict so other modules' data is preserved.
    """
    info = existing if existing is not None else {}

    for v in vehicles:
        tid = v["track_id"]
        if tid not in info:
            info[tid] = {"wrong_side": False, "lines": []}

        stats = speed_stats.get(tid)
        if stats is None:
            continue

        cur = stats["current"]
        avg = stats["avg"]
        mx  = stats["max"]

        # Remove stale speed lines from previous frame
        info[tid]["lines"] = [
            ln for ln in info[tid]["lines"]
            if "km/h" not in ln and "Avg" not in ln and "OVERSPEED" not in ln
        ]

        if cur > 0 or avg > 0:
            info[tid]["lines"].insert(0, f"{cur:.1f} km/h")
            info[tid]["lines"].insert(1, f"Avg:{avg:.1f} Max:{mx:.1f}")

        overspeed = cur > config.SPEED_LIMIT_KMH
        info[tid]["overspeed"] = overspeed
        if overspeed:
            info[tid]["lines"].insert(2, "!! OVERSPEED")

    return info


def draw_pixel_trails(frame, vehicles: list) -> None:
    """
    Draw a fading trajectory trail for each vehicle.
    Older segments are thinner and darker; newest is brightest.
    Only drawn when config.DRAW_TRAIL is True.
    """
    if not config.DRAW_TRAIL:
        return

    for v in vehicles:
        tid   = v["track_id"]
        trail = _pixel_trail.get(tid, [])
        if len(trail) < 2:
            continue

        for i in range(1, len(trail)):
            alpha     = i / len(trail)          # 0.0 (oldest) → 1.0 (newest)
            thickness = max(1, int(alpha * 3))
            b = int(config.COLOR_TRAIL[0] * alpha)
            g = int(config.COLOR_TRAIL[1] * alpha)
            r = int(config.COLOR_TRAIL[2] * alpha)
            cv2.line(frame, trail[i - 1], trail[i], (b, g, r), thickness)


def draw_speed_stats(frame, vehicles: list, speed_stats: dict) -> None:
    """
    Draw a speed summary HUD at the bottom-right of the frame:
        • Average speed of all currently visible moving vehicles
        • Session maximum speed (highest ever recorded)
        • Configured speed limit
    """
    h, w = frame.shape[:2]

    session_max    = 0.0
    visible_speeds = []

    for v in vehicles:
        tid   = v["track_id"]
        stats = speed_stats.get(tid)
        if stats and stats["current"] > 0:
            visible_speeds.append(stats["current"])
            session_max = max(session_max, stats["max"])

    if not visible_speeds:
        return

    global_avg = sum(visible_speeds) / len(visible_speeds)

    hud_lines = [
        "  SPEED STATS  ",
        f"  Visible Avg : {global_avg:.1f} km/h",
        f"  Session Max : {session_max:.1f} km/h",
        f"  Speed Limit : {config.SPEED_LIMIT_KMH} km/h",
    ]

    line_h = 22
    hud_h  = len(hud_lines) * line_h + 10
    hud_w  = 210
    hud_x  = w - hud_w - 8
    hud_y  = h - hud_h - 8

    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h),
                  config.COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, text in enumerate(hud_lines):
        y     = hud_y + line_h * (i + 1)
        color = config.COLOR_LINE if i == 0 else config.COLOR_TEXT
        cv2.putText(frame, text, (hud_x + 4, y),
                    config.FONT, 0.48, color, 1)
