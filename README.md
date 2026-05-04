# 🎯 MotionScope — Moving Object Detection with Tracking + Trajectory Trails

> **Base**: OpenCV MOG2 background subtraction  
> **Novelty 1**: SORT multi-object tracker with persistent IDs (Kalman Filter + Hungarian Algorithm)  
> **Novelty 2**: Colour-coded fading trajectory trails per tracked object  
> **Deployment**: NVIDIA CUDA 12.4 GPU Docker container

---

## Architecture

```
Video Frame
    │
    ▼
MOG2 Background Subtraction      ← separates moving foreground
    │
    ▼
Threshold + Morphological Open   ← removes shadows & noise
    │
    ▼
Contour Detection                ← finds moving blobs
    │
    ▼
Bounding Box Extraction          ← converts contours → [x1,y1,x2,y2]
    │
    ▼
┌─────────────────────────────┐
│  SORT Tracker               │  ◄── Novelty 1
│  Kalman Filter (predict)    │
│  Hungarian Matching (assign)│
│  Persistent ID per object   │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Trajectory Trail Renderer  │  ◄── Novelty 2
│  Deque of (cx,cy) per ID    │
│  Fading alpha polyline      │
└─────────────────────────────┘
    │
    ▼
Annotated Frame → Video Writer + Gradio Preview
```

---

## Tech Novelties

### 1. SORT Multi-Object Tracker (`sort_tracker.py`)

**SORT** (Simple Online and Realtime Tracking) assigns a persistent integer ID to
each detected object and maintains it across frames even through brief occlusions.

| Component | Role |
|-----------|------|
| `KalmanBoxTracker` | 7-state Kalman Filter per object `[cx, cy, w, h, dx, dy, dw]` |
| `associate_detections` | IoU cost matrix + Hungarian algorithm (scipy `linear_sum_assignment`) |
| `SORTTracker` | Manages tracker lifecycle; prunes dead tracks after `max_age` missed frames |

**Parameters** (tunable in `app.py` sliders):
- `max_age=10` — frames before a lost track is deleted  
- `min_hits=3` — confirmations before a track is shown  
- `iou_threshold=0.3` — minimum IoU for detection↔track assignment  

### 2. Trajectory Trails

Each tracked object accumulates a `collections.deque(maxlen=60)` of its centre
points `(cx, cy)`. On every frame, the trail is rendered as a **fading polyline**:

- **Alpha** scales from `0` (oldest point) → `1` (newest point)  
- **Colour** is ID-deterministic from a 30-colour palette, so the same object
  always gets the same colour regardless of restart  
- **Thickness** also scales with alpha, giving a tapered brush stroke effect  

---

## Local Setup (no Docker)

```bash
pip install -r requirements.txt

# Run CLI (saves output_recorded.mp4)
python main.py

# Run Gradio web app
python app.py
# → open http://localhost:7860
```

---

## Docker (GPU)

### Prerequisites

- NVIDIA GPU with driver ≥ 525  
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed

```bash
# Verify GPU is visible to Docker
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Build & Run

```bash
# Option A — Docker Compose (recommended)
docker compose up --build

# Option B — Raw Docker
docker build -t motionscope:gpu .
docker run --gpus all \
  -p 7860:7860 \
  -v $(pwd)/sample:/app/sample \
  -v $(pwd)/outputs:/app/outputs \
  motionscope:gpu
```

→ Open **http://localhost:7860**

### Output Videos

When running via Compose, processed videos are written to `./outputs/` on the host
via the volume mount. You can swap input videos by dropping `.mp4` files into
`./sample/` — no rebuild needed.

---

## File Structure

```
motionscope/
├── sort_tracker.py      ← SORT: Kalman Filter + Hungarian matching
├── main.py              ← CLI inference (cv2.imshow)
├── app.py               ← Gradio web app with live preview
├── requirements.txt
├── Dockerfile           ← CUDA 12.4 + cuDNN 9 + Ubuntu 22.04
├── docker-compose.yml   ← GPU deploy with volume mounts
└── sample/
    └── car.mp4
```

---

## Gradio UI Controls

| Control | Default | Description |
|---------|---------|-------------|
| Min Contour Area | 2000 px² | Filters out small noise blobs |
| Trail Length | 60 frames | How long trajectory history is kept |

---

## References

- Bewley et al. (2016) — [SORT: Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763)  
- [OpenCV MOG2 Background Subtractor](https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html)  
- [LearnOpenCV — Moving Object Detection](https://learnopencv.com/moving-object-detection-with-opencv/)
