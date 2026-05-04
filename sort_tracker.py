"""
Enhanced SORT Tracker
=====================
Base: SORT (Bewley et al. 2016) — Kalman Filter + Hungarian matching

Novelty B — Trajectory Anomaly Scoring (Mahalanobis residual)
  Each Kalman tracker accumulates its innovation residuals over time.
  A per-track anomaly score is computed as the Mahalanobis distance of the
  current residual from the track's own residual distribution. Tracks with
  scores above a threshold are flagged as anomalous — no training data needed.

Novelty C — Geometric Re-ID Buffer
  When a track dies (max_age exceeded), its last known state (bbox + velocity)
  is stored in a re-ID buffer. When new unmatched detections appear, they are
  graph-matched against the buffer using a combined IoU + velocity-extrapolated
  position score. Matched detections resume the original track ID, eliminating
  spurious ID switches after short occlusions.

Novelty D — Dwell-Time & Speed Estimation
  Each active track accumulates:
    - dwell_frames: total frames the object has been tracked (persistence metric)
    - speed_px:     estimated pixel-speed this frame (Euclidean distance of
                    centre-point between consecutive predictions), smoothed with
                    an exponential moving average (alpha=0.3).
  Both values are exposed in the update() output columns 7 and 8, making
  downstream analysis (e.g. loitering detection, fast-mover alerts) trivial
  without any extra computation.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque


# ── IoU helpers ───────────────────────────────────────────────────────────────

def iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt   = (bb_gt[2]   - bb_gt[0])   * (bb_gt[3]   - bb_gt[1])
    union = area_test + area_gt - inter
    return inter / union if union > 0 else 0.0


def associate_detections(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), list(range(len(detections))), []

    iou_matrix = np.zeros((len(detections), len(trackers)))
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices  = np.stack([row_ind, col_ind], axis=1)

    unmatched_dets = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trks = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m)

    return np.array(matches, dtype=int).reshape(-1, 2), unmatched_dets, unmatched_trks


# ── Novelty B: Anomaly Scorer ─────────────────────────────────────────────────

class AnomalyScorer:
    """
    Online Mahalanobis anomaly detector built from Kalman innovation residuals.

    Each frame the tracker produces an innovation vector:
        r_t = z_t - H @ x_t|t-1   (measurement minus prediction)

    We maintain a rolling window of past residuals and compute:
        mu    = mean of residuals over window
        Sigma = covariance of residuals over window
        score = sqrt( (r - mu)^T @ Sigma^-1 @ (r - mu) )   [Mahalanobis distance]

    A high score means the object moved in a way that is unusual *for that
    specific track* — no global training required.
    """

    def __init__(self, window: int = 30, threshold: float = 3.5):
        self.window    = window
        self.threshold = threshold
        self._residuals: deque = deque(maxlen=window)
        self.score     = 0.0
        self.anomalous = False

    def update(self, residual: np.ndarray):
        """residual: shape (4,) — [cx, cy, w, h] innovation"""
        r = residual.flatten()[:4]
        self._residuals.append(r)

        if len(self._residuals) < 6:          # need minimum samples
            self.score     = 0.0
            self.anomalous = False
            return

        R = np.array(self._residuals)          # (N, 4)
        mu    = R.mean(axis=0)
        diff  = R - mu
        Sigma = (diff.T @ diff) / max(len(R) - 1, 1)

        # Regularise to keep Sigma invertible
        Sigma += np.eye(4) * 1e-4

        try:
            Sigma_inv = np.linalg.inv(Sigma)
            d = r - mu
            self.score = float(np.sqrt(d @ Sigma_inv @ d))
        except np.linalg.LinAlgError:
            self.score = 0.0

        self.anomalous = self.score > self.threshold


# ── Kalman Tracker ────────────────────────────────────────────────────────────

class KalmanBoxTracker:
    """
    State: [cx, cy, w, h, vx, vy, vw]
    Augmented with AnomalyScorer (Novelty B).
    Augmented with dwell_frames + speed_px (Novelty D).
    """
    count = 0

    def __init__(self, bbox, restore_id: int = -1):
        self.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=float)
        self.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=float)
        self.R = np.eye(4) * 4.0
        self.Q = np.eye(7) * 0.01
        self.Q[4:, 4:] *= 10
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 100

        x0 = self._bbox_to_z(bbox)
        self.x = np.zeros((7, 1))
        self.x[:4] = x0

        # Assign ID — restore old ID if re-identified (Novelty C)
        if restore_id >= 0:
            self.id = restore_id
        else:
            self.id = KalmanBoxTracker.count
            KalmanBoxTracker.count += 1

        self.hits              = 1
        self.hit_streak        = 1
        self.age               = 0
        self.time_since_update = 0

        # Novelty B — anomaly scorer per track
        self.anomaly = AnomalyScorer(window=30, threshold=3.5)

        # Novelty D — dwell time & speed
        self.dwell_frames = 0
        self.speed_px     = 0.0          # EMA-smoothed pixel speed
        self._prev_cx     = float(x0[0])
        self._prev_cy     = float(x0[1])
        self._speed_alpha = 0.3          # EMA smoothing factor

    # ── coordinate helpers ────────────────────────────────────────────────────
    @staticmethod
    def _bbox_to_z(bbox):
        w  = bbox[2] - bbox[0]
        h  = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2
        cy = bbox[1] + h / 2
        return np.array([[cx], [cy], [w], [h]], dtype=float)

    @staticmethod
    def _z_to_bbox(z):
        cx, cy, w, h = float(z[0]), float(z[1]), float(z[2]), float(z[3])
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

    # ── Kalman predict / update ───────────────────────────────────────────────
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age               += 1
        self.time_since_update += 1
        return self._z_to_bbox(self.x[:4])

    def update(self, bbox):
        z          = self._bbox_to_z(bbox)
        innovation = z - self.H @ self.x          # ← residual (Novelty B)
        S  = self.H @ self.P @ self.H.T + self.R
        K  = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.hits              += 1
        self.hit_streak        += 1
        self.time_since_update = 0

        # Feed residual into anomaly scorer
        self.anomaly.update(innovation)

        # Novelty D — update dwell & speed
        self.dwell_frames += 1
        cx = float(self.x[0])
        cy = float(self.x[1])
        raw_speed = float(np.sqrt((cx - self._prev_cx)**2 + (cy - self._prev_cy)**2))
        self.speed_px = (self._speed_alpha * raw_speed
                         + (1 - self._speed_alpha) * self.speed_px)
        self._prev_cx = cx
        self._prev_cy = cy

    def get_state(self):
        """Returns a plain Python list of 4 Python floats — never nested."""
        z = self.x[:4]
        cx, cy, w, h = float(z[0]), float(z[1]), float(z[2]), float(z[3])
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

    def extrapolate_bbox(self, dt: int):
        """Project bbox forward by dt frames using current velocity (for re-ID)."""
        F_dt = self.F.copy()
        F_dt[0, 4] = dt
        F_dt[1, 5] = dt
        F_dt[2, 6] = dt
        x_proj = F_dt @ self.x
        return self._z_to_bbox(x_proj[:4])


# ── Novelty C: Re-ID Buffer ───────────────────────────────────────────────────

class ReIDBuffer:
    """
    Stores recently-died tracks and attempts to re-identify them when new
    unmatched detections appear.

    Matching score combines:
      - IoU between detection and velocity-extrapolated bbox of dead track
      - Size similarity (aspect ratio consistency)

    No appearance features, no neural network — pure geometry.
    """

    def __init__(self, buffer_age: int = 30, reid_iou_threshold: float = 0.15):
        self.buffer_age       = buffer_age   # frames to keep dead track in buffer
        self.reid_iou_thresh  = reid_iou_threshold
        # entries: {track_id: {state, velocity, bbox, died_at, anomaly_scorer}}
        self._buffer: dict = {}

    def store(self, tracker: KalmanBoxTracker, frame: int):
        """Called when a tracker is about to be pruned."""
        vx = float(tracker.x[4])
        vy = float(tracker.x[5])
        self._buffer[tracker.id] = {
            "tracker":  tracker,
            "died_at":  frame,
            "velocity": (vx, vy),
        }

    def purge_old(self, current_frame: int):
        dead = [k for k, v in self._buffer.items()
                if current_frame - v["died_at"] > self.buffer_age]
        for k in dead:
            del self._buffer[k]

    def match(self, detections: list, current_frame: int):
        """
        Try to match each unmatched detection to a dead track.
        Returns: dict {det_index -> track_id} for successful re-IDs.
        """
        if not self._buffer or not detections:
            return {}

        self.purge_old(current_frame)
        if not self._buffer:
            return {}

        buf_ids   = list(self._buffer.keys())
        score_mat = np.zeros((len(detections), len(buf_ids)))

        for di, det in enumerate(detections):
            for bi, tid in enumerate(buf_ids):
                entry = self._buffer[tid]
                dt    = current_frame - entry["died_at"]
                trk   = entry["tracker"]
                proj  = trk.extrapolate_bbox(dt)   # velocity-extrapolated position
                overlap = iou(det, proj)

                # Size similarity (penalise large scale changes)
                dw = det[2] - det[0];  dh = det[3] - det[1]
                pw = proj[2] - proj[0]; ph = proj[3] - proj[1]
                size_sim = min(dw*dh, pw*ph) / max(dw*dh, pw*ph + 1e-6)

                score_mat[di, bi] = overlap * 0.7 + size_sim * 0.3

        # Hungarian matching on score matrix
        row_ind, col_ind = linear_sum_assignment(-score_mat)
        reid_map = {}
        for r, c in zip(row_ind, col_ind):
            if score_mat[r, c] >= self.reid_iou_thresh:
                tid = buf_ids[c]
                reid_map[r] = tid
                del self._buffer[tid]   # consumed

        return reid_map


# ── Enhanced SORT Tracker ─────────────────────────────────────────────────────

class SORTTracker:
    """
    Enhanced SORT with:
      - Novelty B: per-track Mahalanobis anomaly scoring
      - Novelty C: geometric re-ID buffer for occlusion recovery
      - Novelty D: dwell-time & EMA-smoothed speed estimation

    update() returns Mx9 array:
      [x1, y1, x2, y2, id, anomaly_score, is_anomalous, dwell_frames, speed_px]
    """

    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3,
                 reid_buffer_age=30, anomaly_threshold=3.5):
        self.max_age           = max_age
        self.min_hits          = min_hits
        self.iou_threshold     = iou_threshold
        self.trackers          = []
        self.frame_count       = 0
        self.reid_buffer       = ReIDBuffer(reid_buffer_age, reid_iou_threshold=0.15)
        self.anomaly_threshold = anomaly_threshold
        KalmanBoxTracker.count = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        detections: Nx4 [x1,y1,x2,y2]
        returns:    Mx9 [x1,y1,x2,y2, id, anomaly_score, is_anomalous,
                         dwell_frames, speed_px]
        """
        self.frame_count += 1

        # 1. Predict all active trackers
        predicted_boxes = [trk.predict() for trk in self.trackers]

        # 2. Match detections to active trackers
        det_list = detections.tolist() if len(detections) else []
        matched, unmatched_dets, _ = associate_detections(
            det_list, predicted_boxes, self.iou_threshold
        )

        # 3. Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(detections[m[0]])

        # 4. Novelty C — attempt re-ID for unmatched detections
        unmatched_det_boxes = [detections[i].tolist() for i in unmatched_dets]
        reid_map = self.reid_buffer.match(unmatched_det_boxes, self.frame_count)

        still_unmatched = []
        for local_idx, global_idx in enumerate(unmatched_dets):
            if local_idx in reid_map:
                # Resurrect old track with original ID
                old_id  = reid_map[local_idx]
                old_trk = self.reid_buffer._buffer.get(old_id, {}).get("tracker")
                new_trk = KalmanBoxTracker(detections[global_idx], restore_id=old_id)
                if old_trk is not None:
                    new_trk.anomaly      = old_trk.anomaly       # carry over anomaly history
                    new_trk.dwell_frames = old_trk.dwell_frames  # carry over dwell (Novelty D)
                    new_trk.speed_px     = old_trk.speed_px
                self.trackers.append(new_trk)
            else:
                still_unmatched.append(global_idx)

        # 5. Create new trackers for truly new detections
        for d in still_unmatched:
            self.trackers.append(KalmanBoxTracker(detections[d]))

        # 6. Collect results; store dying trackers in re-ID buffer
        results = []
        keep    = []
        for trk in self.trackers:
            if trk.time_since_update > self.max_age:
                # Novelty C — save to re-ID buffer before pruning
                self.reid_buffer.store(trk, self.frame_count)
                continue
            if trk.hits >= self.min_hits or self.frame_count <= self.min_hits:
                box = trk.get_state()   # guaranteed plain list of 4 Python floats
                # Build row as explicit Python floats — prevents inhomogeneous array
                row = [
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                    float(trk.id),
                    float(trk.anomaly.score),
                    float(1.0 if trk.anomaly.anomalous else 0.0),
                    float(trk.dwell_frames),   # Novelty D
                    float(trk.speed_px),       # Novelty D
                ]
                results.append(row)
            keep.append(trk)
        self.trackers = keep

        if results:
            return np.array(results, dtype=float)   # guaranteed Mx9, all floats
        return np.empty((0, 9))
