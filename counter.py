# counter.py
# Professional Vehicle Counter for TrafficSurveyAI.
#
# Counting logic:
#   • A virtual line (horizontal OR vertical) is drawn across the frame.
#   • A vehicle is counted the moment its centre point crosses the line.
#   • Crossing is detected by a sign-change in the signed distance from
#     the line between the previous frame and the current frame.
#   • Once counted, the track_id is stored in `counted_ids` forever,
#     so a vehicle that disappears and reappears is NEVER counted twice.
#   • Directional filtering (up/down/left/right/both) is supported.

import cv2
import config

# =============================================================================
# STATE  (module-level so it persists across every call)
# =============================================================================

# Last known centre position per track_id  { track_id: (cx, cy) }
_prev_pos: dict[int, tuple[int, int]] = {}

# track_ids that have already been counted — never counted again
_counted_ids: set[int] = set()

# Per-class counts  { "Car": 0, "Motorcycle": 0, ... }
counts: dict[str, int] = {label: 0 for label in config.VEHICLE_CLASSES.values()}


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _signed_distance(cx: int, cy: int) -> float:
    """
    Return a signed value that tells which side of the counting line the
    vehicle's centre is on.

    Horizontal line (y = COUNT_LINE_Y):
        positive → vehicle is BELOW the line  (cy > LINE_Y)
        negative → vehicle is ABOVE the line  (cy < LINE_Y)

    Vertical line (x = COUNT_LINE_X):
        positive → vehicle is to the RIGHT of the line  (cx > LINE_X)
        negative → vehicle is to the LEFT  of the line  (cx < LINE_X)
    """
    if config.LINE_ORIENTATION == "horizontal":
        return cy - config.COUNT_LINE_Y
    else:
        return cx - config.COUNT_LINE_X


def _crossed(prev_d: float, curr_d: float) -> bool:
    """
    A crossing happens when the sign of the signed distance flips between
    two consecutive frames — i.e. prev_d and curr_d have opposite signs.
    Exactly on the line (distance == 0) is treated as a crossing.
    """
    return (prev_d < 0 <= curr_d) or (prev_d > 0 >= curr_d)


def _direction_allowed(prev_cx: int, prev_cy: int,
                        curr_cx: int, curr_cy: int) -> bool:
    """
    Check whether the vehicle's movement direction matches the configured
    COUNT_DIRECTION.  Returns True if the vehicle should be counted.
    """
    direction = config.COUNT_DIRECTION

    if direction == "both":
        return True

    if config.LINE_ORIENTATION == "horizontal":
        delta = curr_cy - prev_cy
        if direction == "down"  and delta > 0: return True
        if direction == "up"    and delta < 0: return True
    else:
        delta = curr_cx - prev_cx
        if direction == "right" and delta > 0: return True
        if direction == "left"  and delta < 0: return True

    return False


# =============================================================================
# PUBLIC API
# =============================================================================

def update(vehicles: list) -> dict[str, int]:
    """
    Process one frame's worth of tracked vehicles.

    For each vehicle:
      1. Compute signed distance from the counting line.
      2. Compare to the previous frame's signed distance.
      3. If a sign-change occurred AND direction is allowed AND the
         track_id has not been counted before → increment the counter.

    Args:
        vehicles: list of vehicle dicts from tracker.get_tracks()

    Returns:
        counts dict  { "Car": N, "Motorcycle": N, ... }
    """
    for v in vehicles:
        tid  = v["track_id"]
        cx   = v["cx"]
        cy   = v["cy"]
        curr_d = _signed_distance(cx, cy)

        if tid in _prev_pos:
            prev_cx, prev_cy = _prev_pos[tid]
            prev_d = _signed_distance(prev_cx, prev_cy)

            # Line was crossed AND vehicle hasn't been counted yet
            if _crossed(prev_d, curr_d) and tid not in _counted_ids:
                if _direction_allowed(prev_cx, prev_cy, cx, cy):
                    _counted_ids.add(tid)
                    counts[v["label"]] += 1

        _prev_pos[tid] = (cx, cy)

    return counts


def draw(frame, vehicles: list) -> None:
    """
    Draw all counter visuals onto the frame in-place:

      • The counting line (magenta)
      • A small centre dot on each vehicle
      • A highlight flash on the line when a vehicle just crossed
      • The live stats HUD (top-left corner)

    Args:
        frame:    the current video frame (modified in-place)
        vehicles: list of vehicle dicts from tracker.get_tracks()
    """
    h, w = frame.shape[:2]

    # ── 1. Counting line ──────────────────────────────────────────────────────
    if config.LINE_ORIENTATION == "horizontal":
        pt1 = (0,  config.COUNT_LINE_Y)
        pt2 = (w,  config.COUNT_LINE_Y)
    else:
        pt1 = (config.COUNT_LINE_X, 0)
        pt2 = (config.COUNT_LINE_X, h)

    cv2.line(frame, pt1, pt2, config.COLOR_LINE, 2)

    # Label next to the line
    label_pos = (pt1[0] + 6, pt1[1] - 8) if config.LINE_ORIENTATION == "horizontal" \
                else (pt1[0] + 6, 20)
    cv2.putText(frame, "COUNT LINE", label_pos,
                config.FONT, 0.45, config.COLOR_LINE, 1)

    # ── 2. Centre dot on each vehicle ─────────────────────────────────────────
    for v in vehicles:
        cv2.circle(frame, (v["cx"], v["cy"]), 4, config.COLOR_DOT, -1)

    # ── 3. Stats HUD (top-left) ───────────────────────────────────────────────
    total = sum(counts.values())
    hud_lines = [
        f"  VEHICLE COUNT  ",
        f"  Total     : {total}",
    ] + [f"  {lbl:<11}: {cnt}" for lbl, cnt in counts.items()]

    # Compute HUD background size
    line_h   = 22
    hud_h    = len(hud_lines) * line_h + 10
    hud_w    = 190
    hud_x, hud_y = 8, 8

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (hud_x, hud_y),
                  (hud_x + hud_w, hud_y + hud_h),
                  config.COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, text in enumerate(hud_lines):
        y = hud_y + line_h * (i + 1)
        # Header line in a brighter colour
        color = config.COLOR_LINE if i == 0 else config.COLOR_TEXT
        cv2.putText(frame, text, (hud_x + 4, y),
                    config.FONT, 0.48, color, 1)
