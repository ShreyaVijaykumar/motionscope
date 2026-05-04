# 🎯 MotionScope

> Real-time multi-object motion tracking with anomaly detection, occlusion-resilient re-identification, and dwell-time/speed analytics — containerised with GPU acceleration.

[![Docker](https://img.shields.io/badge/Docker-shrevj%2Fmotionscope-blue?logo=docker)](https://hub.docker.com/r/shrevj/motionscope)
[![CUDA](https://img.shields.io/badge/CUDA-12.4.1-green?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.11-yellow?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.3-lightgrey?logo=flask)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Novelties](#novelties)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Docker Hub](#docker-hub)
- [Configuration](#configuration)
- [Output](#output)
- [Project Structure](#project-structure)

---

## Overview

MotionScope is a GPU-accelerated video analytics pipeline that combines classical computer vision (MOG2 background subtraction) with an enhanced SORT tracker to deliver persistent multi-object tracking, per-track anomaly scoring, occlusion-recovery re-identification, and real-time dwell-time/speed estimation — all served through a browser-based live-stream interface.

---

## Features

| Feature | Description |
|---|---|
| 🎥 **Live MJPEG Stream** | Real-time annotated preview in browser at `localhost:7860` |
| 🔲 **MOG2 Background Subtraction** | Adaptive Gaussian mixture model for foreground segmentation |
| 🏷️ **Persistent Track IDs** | SORT Kalman + Hungarian matching maintains IDs across frames |
| ⚠️ **Anomaly Detection** | Per-track Mahalanobis distance scoring; no training data required |
| 🔁 **Re-ID Recovery** | Geometric re-ID buffer resurrects original IDs after occlusion |
| 📊 **Dwell & Speed Metrics** | EMA-smoothed pixel speed + frame-level dwell counter per track |
| 🎨 **Trajectory Trails** | Colour-coded fading polyline trails per object |
| ⬇️ **Video Download** | Annotated output `.mp4` downloadable after processing |

---

## Architecture

```
Video Input
    │
    ▼
┌─────────────────────────────┐
│  MOG2 Background Subtractor │  ← Adaptive Gaussian mixture, history=500
└─────────────┬───────────────┘
              │  Foreground Mask
              ▼
┌─────────────────────────────┐
│  Morphological Cleanup      │  ← Threshold → Open → Dilate (kernel 3×3)
└─────────────┬───────────────┘
              │  Clean Mask
              ▼
┌─────────────────────────────┐
│  Contour Detection          │  ← findContours → boundingRect → filter area
└─────────────┬───────────────┘
              │  Detections [x1,y1,x2,y2]
              ▼
┌─────────────────────────────────────────────────────────┐
│                   SORTTracker.update()                  │
│                                                         │
│  ┌─────────────────┐    ┌──────────────────────────┐   │
│  │ KalmanBoxTracker│    │  Hungarian Association   │   │
│  │  predict()      │───▶│  (IoU cost matrix)       │   │
│  └─────────────────┘    └──────────┬───────────────┘   │
│                                    │                    │
│               ┌────────────────────┼──────────────┐    │
│               ▼                    ▼               ▼    │
│         [Matched]          [Unmatched Dets]  [Unmatched │
│        update KF           ReIDBuffer.match()  Trks]    │
│        AnomalyScorer        ↓ resurrect or              │
│        DwellSpeed           ↓ new tracker               │
└──────────────────────────────┬──────────────────────────┘
                               │  Results [x1,y1,x2,y2,id,
                               │           score,flag,dwell,speed]
                               ▼
                   Annotation & Rendering
                   (trails, HUD, labels, dwell bar)
                               │
                               ▼
                   MJPEG Stream + MP4 Write
```

---

## Novelties

### Novelty A — Mahalanobis Anomaly Scoring
Each Kalman tracker accumulates its innovation residuals `r_t = z_t − H x̂_{t|t-1}` in a rolling window. The Mahalanobis distance of the current residual from the track's own distribution flags unusual motion — **no labelled data required**.

### Novelty B — Geometric Re-ID Buffer
Dead tracks are stored with their extrapolated bounding boxes. Unmatched detections are matched against the buffer via IoU + size similarity, restoring original IDs after occlusion without any appearance features.

### Novelty C — Dwell-Time & EMA Speed
Each track maintains a `dwell_frames` counter (frames actively updated) and an EMA-smoothed pixel speed `v_t = α·d_t + (1−α)·v_{t-1}`, enabling loitering detection and fast-mover alerts with zero extra compute.

---

## Prerequisites

- **NVIDIA GPU** with driver ≥ 525
- **Docker** + **NVIDIA Container Toolkit**
  - Install guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- **Docker Compose** (included with Docker Desktop)

Verify GPU is accessible to Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## Quick Start

### Option A — Pull from Docker Hub (recommended)
```bash
docker pull shrevj/motionscope:latest
docker run --gpus all -p 7860:7860 shrevj/motionscope:latest
```

### Option B — Build locally
```bash
git clone https://github.com/shrevj/motionscope.git
cd motionscope
docker compose up --build
```

Open your browser at **http://localhost:7860**

---

## Usage

1. Open **http://localhost:7860**
2. Click **📥 Choose File** → select any `.mp4` video  
   *(a sample `car.mp4` is bundled at `/app/sample/car.mp4`)*
3. Adjust sliders:
   - **Min contour area** — filters noise (default: 2000 px²)
   - **Trail length** — frames of motion history shown (default: 60)
4. Click **▶ Analyse**
5. Watch the live annotated stream on the right panel
6. Monitor stats in real-time:

| Stat | Description |
|---|---|
| Active Tracks | Currently visible objects |
| Frame | Current processing frame |
| ⚠ Anomalies | Objects with high Mahalanobis score this frame |
| 🔁 Re-IDs | Total identity recoveries so far |
| Avg Speed | Mean pixel speed across all tracks |
| Max Dwell | Longest-running track in frames |

7. When processing completes → click **⬇ Download output video**

---

## Docker Hub

```bash
# Pull
docker pull shrevj/motionscope:latest

# Run with GPU
docker run --gpus all -p 7860:7860 shrevj/motionscope:latest

# Run without GPU (CPU fallback — slower)
docker run -p 7860:7860 shrevj/motionscope:latest
```

Image URL: https://hub.docker.com/r/shrevj/motionscope

---

## Configuration

All parameters are exposed via the UI sliders. For programmatic control, edit environment or pass form values to `/process`:

| Parameter | Default | Range | Description |
|---|---|---|---|
| `min_area` | 2000 | 100–5000 | Minimum contour area in px² |
| `trail_len` | 60 | 10–120 | Trail history length in frames |
| `max_age` | 15 | — | Frames before a track is pruned |
| `min_hits` | 3 | — | Minimum hits before track is confirmed |
| `iou_threshold` | 0.3 | — | IoU threshold for Hungarian matching |
| `reid_buffer_age` | 40 | — | Frames to keep dead track in re-ID buffer |
| `anomaly_threshold` | 3.5 | — | Mahalanobis score to flag as anomalous |

---

## Output

Each processed frame is annotated with:

- **Coloured bounding box** per track (unique palette, 30 colours cycling)
- **🔴 Red box + glow ring** for anomalous tracks
- **Label** `IDn [⚠] s=X.X v=Y.Y` — ID, anomaly flag, score, speed
- **Fading trail** — polyline with opacity proportional to recency
- **Green dwell bar** — thin stripe below bbox, width ∝ dwell time
- **HUD overlay** — Active / Anomalies / Re-IDs / Avg Speed / Max Dwell

Output video saved to `/tmp/output.mp4` inside container; downloadable via `/download` endpoint.  
Mount `./outputs:/app/outputs` in `docker-compose.yml` to persist on host.

---

## Project Structure

```
motionscope/
├── app.py              # Flask app — routes, MJPEG stream, processing thread
├── sort_tracker.py     # Enhanced SORT — Kalman, Anomaly, ReID, Dwell/Speed
├── main.py             # Standalone CLI runner (no Flask)
├── requirements.txt    # Python dependencies
├── Dockerfile          # CUDA 12.4.1 + Python 3.11 image
├── docker-compose.yml  # GPU compose config
├── build_and_push.sh   # DockerHub build & push script
├── sample/
│   └── car.mp4         # Bundled test video
└── outputs/            # Host-mounted output directory
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Base image | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` |
| Language | Python 3.11 |
| Web framework | Flask 3.0.3 |
| Computer vision | OpenCV 4.9.0 (headless) |
| Numerical computing | NumPy 1.26.4 |
| Optimisation | SciPy 1.13.0 (Hungarian algorithm) |
| Streaming | MJPEG over HTTP (multipart/x-mixed-replace) |
| GPU | CUDA 12.4.1 + cuDNN |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
