"""
main.py — MotionScope with SORT Tracking + Trajectory Trails
Novelties added:
  1. SORT multi-object tracker → persistent IDs across frames
  2. Trajectory trails → colour-coded motion paths per object
"""

import cv2
import numpy as np
import collections
from sort_tracker import SORTTracker

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_VIDEO      = "sample/car.mp4"
OUTPUT_VIDEO     = "output_recorded.mp4"
MIN_CONTOUR_AREA = 500
TRAIL_LENGTH     = 60       # frames of trail history per object
TRAIL_THICKNESS  = 2
BOX_THICKNESS    = 2
FONT             = cv2.FONT_HERSHEY_SIMPLEX
# ────────────────────────────────────────────────────────────────────────────

# Deterministic colour palette (one colour per object ID, cycles at 30)
_PALETTE = [
    (255, 56,  56 ), (255, 157, 56 ), (255, 254, 56 ), (56,  255, 56 ),
    (56,  255, 157), (56,  255, 254), (56,  157, 255), (56,  56,  255),
    (157, 56,  255), (254, 56,  255), (200, 100, 50 ), (50,  200, 100),
    (100, 50,  200), (200, 50,  100), (50,  100, 200), (100, 200, 50 ),
    (230, 180, 30 ), (30,  230, 180), (180, 30,  230), (180, 230, 30 ),
    (30,  180, 230), (230, 30,  180), (120, 220, 120), (220, 120, 120),
    (120, 120, 220), (255, 128, 0  ), (0,   255, 128), (128, 0,   255),
    (255, 0,   128), (0,   128, 255),
]

def get_color(obj_id: int):
    return _PALETTE[obj_id % len(_PALETTE)]


def draw_trail(frame: np.ndarray, trail: list, color: tuple) -> np.ndarray:
    """Draw a fading polyline trail on frame."""
    n = len(trail)
    for i in range(1, n):
        alpha = i / n           # older points are more transparent
        thickness = max(1, int(TRAIL_THICKNESS * alpha))
        # Blend color toward black based on age
        c = tuple(int(v * alpha) for v in color)
        cv2.line(frame, trail[i - 1], trail[i], c, thickness, cv2.LINE_AA)
    return frame


def vid_inf(vid_path: str = INPUT_VIDEO):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print("Error opening video file:", vid_path)
        return

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
    out          = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

    # ── Background subtractor ──────────────────────────────────────────────
    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=True
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # ── SORT tracker ───────────────────────────────────────────────────────
    tracker = SORTTracker(max_age=10, min_hits=3, iou_threshold=0.3)

    # ── Trail storage: obj_id → deque of (cx, cy) ─────────────────────────
    trails: dict[int, collections.deque] = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Background subtraction
        fg_mask = back_sub.apply(frame)

        # 2. Threshold (remove shadows, which MOG2 labels as 127)
        _, mask_thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)

        # 3. Morphological opening to kill noise
        mask_clean = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.dilate(mask_clean, kernel, iterations=2)

        # 4. Contour detection → bounding boxes
        contours, _ = cv2.findContours(
            mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        detections = []
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, x + w, y + h])

        dets_np = np.array(detections, dtype=float) if detections else np.empty((0, 4))

        # 5. SORT update → [x1, y1, x2, y2, id]
        tracked = tracker.update(dets_np)

        # 6. Draw trails + bounding boxes
        frame_out = frame.copy()
        for obj in tracked:
            x1, y1, x2, y2, obj_id = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            color  = get_color(obj_id)

            # Update trail
            if obj_id not in trails:
                trails[obj_id] = collections.deque(maxlen=TRAIL_LENGTH)
            trails[obj_id].append((cx, cy))

            # Draw fading trail
            trail_pts = list(trails[obj_id])
            draw_trail(frame_out, trail_pts, color)

            # Draw bounding box
            cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, BOX_THICKNESS, cv2.LINE_AA)

            # Draw ID label with background
            label    = f"ID {obj_id}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
            cv2.rectangle(frame_out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame_out, label, (x1 + 2, y1 - 4), FONT, 0.55,
                        (0, 0, 0), 1, cv2.LINE_AA)

        # 7. Overlay stats
        cv2.putText(frame_out, f"Active objects: {len(tracked)}", (10, 30),
                    FONT, 0.8, (0, 255, 200), 2, cv2.LINE_AA)

        out.write(frame_out)
        cv2.imshow("MotionScope — Tracking + Trails", frame_out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved → {OUTPUT_VIDEO}")


if __name__ == "__main__":
    vid_inf()
