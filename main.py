# main.py
# Entry point for TrafficSurveyAI.
# Pipeline per frame:
#   get_tracks → counter → speed → draw trails → draw boxes → HUD
# Run with:  python main.py

import cv2
import config
import tracker
import counter
import speed_estimator as speed


def main():
    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {config.VIDEO_PATH}")
        return

    # Use forced FPS if configured, otherwise read from the file
    fps    = config.FORCE_FPS or cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Video writer ──────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(config.OUTPUT_PATH, fourcc, fps, (width, height))

    print(f"[INFO] Processing '{config.VIDEO_PATH}'  |  {width}x{height}  |  FPS: {fps:.1f}")
    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Step 1 : Detect & track ───────────────────────────────────────────
        vehicles = tracker.get_tracks(frame)

        # ── Step 2 : Count vehicles crossing the line ─────────────────────────
        counter.update(vehicles)
        counter.draw(frame, vehicles)

        # ── Step 3 : Speed estimation ──────────────────────────────────────────
        speed.update_pixel_trail(vehicles)          # store pixel trail
        speed_stats = speed.update(vehicles, fps)   # compute speeds
        extra_info  = speed.build_extra_info(vehicles, speed_stats)

        # ── Step 4 : Draw trails (under boxes) ────────────────────────────────
        speed.draw_pixel_trails(frame, vehicles)

        # ── Step 5 : Draw bounding boxes + labels ─────────────────────────────
        tracker.draw_boxes(frame, vehicles, extra_info)

        # ── Step 6 : Speed stats HUD (bottom-right) ───────────────────────────
        speed.draw_speed_stats(frame, vehicles, speed_stats)

        # ── Write & display ────────────────────────────────────────────────────
        writer.write(frame)
        cv2.imshow("TrafficSurveyAI", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit by user.")
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Output saved to '{config.OUTPUT_PATH}'")

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n── Traffic Summary ──────────────────────────")
    for label, cnt in counter.counts.items():
        print(f"  {label:<12}: {cnt}")
    print(f"  {'Total':<12}: {sum(counter.counts.values())}")
    print("─────────────────────────────────────────────")

    # ── Per-vehicle speed log ──────────────────────────────────────────────────
    if speed.departed_log:
        print(f"\n── Per-Vehicle Speed Log ({len(speed.departed_log)} vehicles) ──")
        print(f"  {'#':<6} {'Type':<12} {'Avg (km/h)':>10} {'Max (km/h)':>10}  Flag")
        print(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*10}  ----")
        for e in speed.departed_log:
            flag = "OVERSPEED" if e["max"] > config.SPEED_LIMIT_KMH else ""
            print(f"  {e['id']:<6} {e['label']:<12} {e['avg']:>10.1f} {e['max']:>10.1f}  {flag}")
        print("─────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
