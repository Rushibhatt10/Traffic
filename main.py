# main.py
# TrafficSurveyAI — Government Traffic Survey System
#
# Offline video analysis pipeline. Processes EVERY frame sequentially.
# Uses original video FPS for all speed calculations — never wall-clock time.
# Output video has identical FPS, resolution, and duration as the input.
#
# Run:  python main.py

import time
import cv2
import config
import tracker
import counter
import speed_estimator as speed


def main():

    # =========================================================================
    # OPEN VIDEO — read all metadata before touching frames
    # =========================================================================
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {config.VIDEO_PATH}")
        return

    # Read FPS directly from the video container.
    # This value is passed to speed_estimator so all speed calculations are
    # based on real video time, NOT on how fast this machine processes frames.
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = config.FORCE_FPS if config.FORCE_FPS else video_fps
    if not fps or fps <= 0:
        fps = 30.0
        print(f"[WARN] FPS unreadable from video. Using default {fps:.1f}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur_s  = total / fps if fps > 0 else 0

    print()
    print("═" * 55)
    print("  TrafficSurveyAI — Offline Video Survey")
    print("═" * 55)
    print(f"  Input       : {config.VIDEO_PATH}")
    print(f"  Resolution  : {width} × {height}")
    print(f"  Video FPS   : {fps:.3f}")
    print(f"  Total frames: {total}  ({dur_s:.1f} sec)")
    print(f"  Model       : {config.MODEL_PATH}  |  imgsz={config.IMGSZ}")
    print(f"  Output      : {config.OUTPUT_PATH}")
    print(f"  Preview     : {'ON' if config.SHOW_PREVIEW else 'OFF (headless)'}")
    print("═" * 55)
    print()

    # =========================================================================
    # VIDEO WRITER
    # Exact same FPS and resolution as the input — duration is preserved.
    # =========================================================================
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(config.OUTPUT_PATH, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"[ERROR] Cannot open output writer: {config.OUTPUT_PATH}")
        cap.release()
        return

    # =========================================================================
    # FRAME LOOP
    # Every frame is processed. No skipping. No dropping.
    # =========================================================================
    frame_idx       = 0
    t_start         = time.perf_counter()   # wall-clock only for progress display
    t_last_report   = t_start
    inference_times = []   # per-frame inference durations (seconds)

    if config.SHOW_PREVIEW:
        print("  [Preview ON]  Press 'q' in the window to stop early.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break   # clean end of video

        frame_idx += 1

        # ── Step 1 : YOLO11x + ByteTrack detection ────────────────────────────
        # Measures only the inference time so we can report avg inference FPS.
        t0       = time.perf_counter()
        vehicles = tracker.get_tracks(frame)
        inference_times.append(time.perf_counter() - t0)

        # ── Step 2 : Vehicle counting (virtual line) ──────────────────────────
        counter.update(vehicles)
        counter.draw(frame, vehicles)

        # ── Step 3 : Speed estimation ──────────────────────────────────────────
        # fps passed here is the VIDEO fps — speed calculation uses real video
        # timestamps, completely independent of processing speed.
        speed.update_pixel_trail(vehicles)
        speed_stats = speed.update(vehicles, fps)
        extra_info  = speed.build_extra_info(vehicles, speed_stats)

        # ── Step 4 : Draw trajectory trails (rendered under boxes) ────────────
        speed.draw_pixel_trails(frame, vehicles)

        # ── Step 5 : Draw bounding boxes + speed labels ───────────────────────
        tracker.draw_boxes(frame, vehicles, extra_info)

        # ── Step 6 : Speed stats HUD (bottom-right) ───────────────────────────
        speed.draw_speed_stats(frame, vehicles, speed_stats)

        # ── Step 7 : Frame info HUD (top-right) ───────────────────────────────
        # Shows frame number and INFERENCE fps (not display fps)
        avg_inf_fps = 1.0 / (sum(inference_times[-30:]) / min(len(inference_times), 30)) \
                      if inference_times else 0.0
        tracker.draw_frame_info(frame, frame_idx, avg_inf_fps, len(vehicles))

        # ── Step 8 : Write annotated frame to output video ────────────────────
        writer.write(frame)

        # ── Step 9 : Optional live preview ────────────────────────────────────
        if config.SHOW_PREVIEW:
            cv2.imshow("TrafficSurveyAI", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n  [INFO] Stopped early by user.")
                break

        # ── Terminal progress — printed every 25 frames ───────────────────────
        if frame_idx % 25 == 0 or frame_idx == total:
            pct     = (frame_idx / total * 100) if total > 0 else 0.0
            elapsed = time.perf_counter() - t_start
            eta_s   = (elapsed / frame_idx) * (total - frame_idx) if frame_idx > 0 else 0
            print(f"  Frame {frame_idx:>5} / {total}  ({pct:5.1f}%)  "
                  f"inf={avg_inf_fps:5.1f} fps  "
                  f"ETA {eta_s:6.0f}s  "
                  f"vehicles={len(vehicles)}")

    # =========================================================================
    # CLEANUP
    # =========================================================================
    cap.release()
    writer.release()
    if config.SHOW_PREVIEW:
        cv2.destroyAllWindows()

    # =========================================================================
    # FINAL REPORT
    # =========================================================================
    total_time   = time.perf_counter() - t_start
    avg_inf      = 1.0 / (sum(inference_times) / len(inference_times)) \
                   if inference_times else 0.0

    print()
    print("═" * 55)
    print("  PROCESSING COMPLETE")
    print("═" * 55)
    print(f"  Frames processed    : {frame_idx}")
    print(f"  Total time          : {total_time:.1f} s  ({total_time/60:.1f} min)")
    print(f"  Avg inference FPS   : {avg_inf:.2f} fps")
    print(f"  Output video        : {config.OUTPUT_PATH}")
    print("═" * 55)

    # ── Vehicle count summary ─────────────────────────────────────────────────
    print()
    print("  ── Vehicle Count ─────────────────────────")
    for label, cnt in counter.counts.items():
        print(f"    {label:<14}: {cnt:>5}")
    print(f"    {'TOTAL':<14}: {sum(counter.counts.values()):>5}")

    # ── Per-vehicle speed log ─────────────────────────────────────────────────
    if speed.departed_log:
        print()
        print(f"  ── Per-Vehicle Speed Log  ({len(speed.departed_log)} vehicles) ──")
        print(f"    {'ID':<6} {'Type':<14} {'Avg km/h':>9} {'Max km/h':>9}  Note")
        print(f"    {'-'*6} {'-'*14} {'-'*9} {'-'*9}  ----")
        for e in speed.departed_log:
            note = "OVERSPEED" if e["max"] > config.SPEED_LIMIT_KMH else ""
            print(f"    {e['id']:<6} {e['label']:<14} "
                  f"{e['avg']:>9.1f} {e['max']:>9.1f}  {note}")
    print()


if __name__ == "__main__":
    main()
