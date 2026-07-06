# wrong_side.py
# Detects vehicles travelling in the wrong direction on a TWO-WAY road.
# No AI training — purely compares movement direction against expected
# direction for each side of the road (split by ROAD_CENTRE_X).
#
# Left  half (cx < ROAD_CENTRE_X) → correct direction is UP   (delta Y < 0)
# Right half (cx ≥ ROAD_CENTRE_X) → correct direction is DOWN (delta Y > 0)

import config

# ── State ─────────────────────────────────────────────────────────────────────
prev_cy: dict[int, int] = {}          # last known centre-Y per track_id
wrong_side_ids: set[int] = set()      # currently flagged track_ids
counted_wrong: set[int] = set()       # ids already added to the total count
wrong_side_count: int = 0             # total unique wrong-side vehicles


def _expected_direction(cx: int) -> str:
    """
    Return the expected direction of travel based on which side of the
    road the vehicle is on.
        cx < ROAD_CENTRE_X  →  vehicle is on left lanes  →  should go "up"
        cx ≥ ROAD_CENTRE_X  →  vehicle is on right lanes →  should go "down"
    """
    return "up" if cx < config.ROAD_CENTRE_X else "down"


def update(vehicles: list) -> tuple[set[int], int]:
    """
    Evaluate each vehicle's direction and flag wrong-side movers.

    Returns:
        wrong_side_ids   – set of track_ids currently moving the wrong way
        wrong_side_count – cumulative unique wrong-side count
    """
    global wrong_side_count

    for v in vehicles:
        tid = v["track_id"]
        cx  = v["cx"]
        cy  = v["cy"]

        if tid in prev_cy:
            delta = cy - prev_cy[tid]   # +ve = moving down, -ve = moving up

            # Only judge if vehicle moved enough to rule out jitter
            if abs(delta) >= config.MIN_MOVEMENT_PX:
                expected  = _expected_direction(cx)
                moving_down = delta > 0

                is_wrong = (expected == "down" and not moving_down) or \
                           (expected == "up"   and moving_down)

                if is_wrong:
                    wrong_side_ids.add(tid)
                    if tid not in counted_wrong:
                        counted_wrong.add(tid)
                        wrong_side_count += 1
                else:
                    # Vehicle is back on correct course — remove live flag
                    wrong_side_ids.discard(tid)

        prev_cy[tid] = cy

    return wrong_side_ids, wrong_side_count


def build_extra_info(vehicles: list, existing: dict = None) -> dict:
    """
    Inject wrong-side flags into the extra_info dict used by
    tracker.draw_boxes().  Merges with existing data (e.g. speed lines).
    """
    info = existing if existing is not None else {}

    for v in vehicles:
        tid = v["track_id"]
        if tid not in info:
            info[tid] = {"wrong_side": False, "lines": []}

        if tid in wrong_side_ids:
            info[tid]["wrong_side"] = True
            if "⚠ WRONG SIDE" not in info[tid]["lines"]:
                info[tid]["lines"].append("⚠ WRONG SIDE")
        else:
            info[tid]["wrong_side"] = False
            # Remove stale wrong-side text if vehicle corrected course
            info[tid]["lines"] = [
                l for l in info[tid]["lines"] if "WRONG SIDE" not in l
            ]

    return info


def draw_wrong_side_count(frame) -> None:
    """
    Display the cumulative wrong-side count at the bottom-left of the frame.
    """
    import cv2
    h = frame.shape[0]
    text = f"Wrong Side: {wrong_side_count}"
    cv2.putText(frame, text,
                (10, h - 16),
                config.FONT, config.FONT_SCALE,
                config.COLOR_WRONG_SIDE, config.THICKNESS)
