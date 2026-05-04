"""
MotionScope — Flask MJPEG app
Novelty B: Mahalanobis anomaly scoring per track
Novelty C: Geometric re-ID buffer for occlusion recovery
Novelty D: Dwell-time & EMA-smoothed speed estimation per track
"""
import os, collections, threading, time
import cv2
import numpy as np
from flask import Flask, Response, request, jsonify, render_template_string, send_file
from sort_tracker import SORTTracker

app = Flask(__name__)

_lock       = threading.Lock()
_frame_buf  = None
_processing = False
_stats      = {"active": 0, "frame": 0, "done": False,
               "anomalies": 0, "reids": 0, "avg_speed": 0.0, "max_dwell": 0}

# ── colour palette ────────────────────────────────────────────────────────────
_PALETTE = [
    (255,56,56),(255,157,56),(255,254,56),(56,255,56),(56,255,157),
    (56,255,254),(56,157,255),(56,56,255),(157,56,255),(254,56,255),
    (200,100,50),(50,200,100),(100,50,200),(200,50,100),(50,100,200),
    (100,200,50),(230,180,30),(30,230,180),(180,30,230),(180,230,30),
    (30,180,230),(230,30,180),(120,220,120),(220,120,120),(120,120,220),
    (255,128,0),(0,255,128),(128,0,255),(255,0,128),(0,128,255),
]
def palette(oid): return _PALETTE[int(oid) % len(_PALETTE)]

def draw_trail(frame, trail, c):
    n = len(trail)
    for i in range(1, n):
        a = i / n
        cv2.line(frame, trail[i-1], trail[i],
                 tuple(int(v*a) for v in c), max(1, int(2*a)), cv2.LINE_AA)

# ── processing thread ─────────────────────────────────────────────────────────
def process_video(path, min_area, trail_len):
    global _frame_buf, _processing, _stats
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        _processing = False
        return

    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter("/tmp/output.mp4", fourcc, fps, (w, h))

    back_sub = cv2.createBackgroundSubtractorMOG2(500, 50, True)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    tracker  = SORTTracker(max_age=15, min_hits=3, iou_threshold=0.3,
                           reid_buffer_age=40, anomaly_threshold=3.5)
    trails   = {}
    total_reids    = 0
    prev_track_ids = set()
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        fg = back_sub.apply(frame)
        _, thr = cv2.threshold(fg, 180, 255, cv2.THRESH_BINARY)
        msk = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
        msk = cv2.dilate(msk, kernel, iterations=2)

        cnts, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in cnts:
            if cv2.contourArea(c) < min_area: continue
            x,y,ww,hh = cv2.boundingRect(c)
            dets.append([x, y, x+ww, y+hh])

        dets_np = np.array(dets, dtype=float) if dets else np.empty((0,4))

        # tracker returns [x1,y1,x2,y2, id, anomaly_score, is_anomalous,
        #                  dwell_frames, speed_px]
        tracked = tracker.update(dets_np)

        # Count re-IDs (IDs reappearing after being gone)
        cur_ids = set(int(t[4]) for t in tracked) if len(tracked) else set()
        reids_this_frame = len(cur_ids & (prev_track_ids - cur_ids))
        total_reids += reids_this_frame
        prev_track_ids = cur_ids

        out_frame    = frame.copy()
        n_anomalous  = 0
        speeds       = []
        max_dwell    = 0

        for obj in tracked:
            x1,y1,x2,y2 = int(obj[0]),int(obj[1]),int(obj[2]),int(obj[3])
            oid          = int(obj[4])
            ascore       = float(obj[5])
            is_anom      = bool(obj[6])
            dwell        = int(obj[7])    # Novelty D
            speed        = float(obj[8])  # Novelty D
            cx,cy        = (x1+x2)//2, (y1+y2)//2

            if is_anom:
                n_anomalous += 1
            speeds.append(speed)
            max_dwell = max(max_dwell, dwell)

            # Box colour: red flash for anomalous, palette otherwise
            box_color = (0, 0, 255) if is_anom else palette(oid)

            # Trail
            if oid not in trails:
                trails[oid] = collections.deque(maxlen=int(trail_len))
            trails[oid].append((cx,cy))
            trail_color = (0,80,255) if is_anom else palette(oid)
            draw_trail(out_frame, list(trails[oid]), trail_color)

            # Bounding box
            cv2.rectangle(out_frame, (x1,y1), (x2,y2), box_color, 2, cv2.LINE_AA)

            # Anomaly glow ring
            if is_anom:
                cv2.rectangle(out_frame, (x1-3,y1-3), (x2+3,y2+3),
                              (0,0,200), 1, cv2.LINE_AA)

            # Label: ID + anomaly score + speed (Novelty D)
            label = f"ID{oid} {'⚠' if is_anom else ''} s={ascore:.1f} v={speed:.1f}"
            (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out_frame, (x1, y1-th-6), (x1+tw+4, y1), box_color, -1)
            cv2.putText(out_frame, label, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)

            # Dwell bar: thin green bar along top edge of bbox (Novelty D)
            dwell_width = min(x2 - x1, int((x2 - x1) * min(dwell, 300) / 300))
            cv2.rectangle(out_frame, (x1, y2+2), (x1+dwell_width, y2+5),
                          (0,255,100), -1)

        avg_speed = float(np.mean(speeds)) if speeds else 0.0

        # HUD overlay
        hud_lines = [
            f"Active:    {len(tracked)}",
            f"Anomalies: {n_anomalous}",
            f"Re-IDs:    {total_reids}",
            f"Avg speed: {avg_speed:.1f} px/f",   # Novelty D
            f"Max dwell: {max_dwell} frm",          # Novelty D
        ]
        for i, line in enumerate(hud_lines):
            color = (0,80,255) if (i==1 and n_anomalous>0) else (0,255,200)
            cv2.putText(out_frame, line, (10, 30 + i*28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        out.write(out_frame)

        if count % 3 == 0:
            _, jpg = cv2.imencode(".jpg", out_frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            with _lock:
                _frame_buf = jpg.tobytes()
                _stats.update({
                    "active":     int(len(tracked)),
                    "frame":      count,
                    "anomalies":  int(n_anomalous),
                    "reids":      int(total_reids),
                    "avg_speed":  round(avg_speed, 1),
                    "max_dwell":  int(max_dwell),
                })
        count += 1

    cap.release()
    out.release()
    with _lock:
        _stats["done"] = True
        _processing    = False


# ── HTML UI ───────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MotionScope</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0f1e;color:#cdd6f4;font-family:'Courier New',monospace}
h1{text-align:center;padding:22px 0 6px;color:#00ffe7;letter-spacing:.15em;font-size:1.9rem}
p.sub{text-align:center;color:#6c7086;margin-bottom:18px;font-size:.82rem}
.wrap{display:flex;gap:18px;padding:0 20px 24px;flex-wrap:wrap}
.panel{background:#1e1e2e;border-radius:12px;padding:18px;flex:1;min-width:270px}
label{display:block;margin-bottom:5px;color:#89b4fa;font-size:.83rem}
input[type=file]{width:100%;margin-bottom:14px}
input[type=range]{width:100%;margin-bottom:14px;accent-color:#00ffe7}
.val{color:#cba6f7;font-size:.78rem}
button{width:100%;padding:11px;border:none;border-radius:8px;
  background:#00ffe7;color:#0a0f1e;font-weight:bold;font-size:.95rem;cursor:pointer}
button:hover{background:#94e2d5}
#stream{width:100%;border-radius:8px;background:#11111b;min-height:220px;display:block}
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}
.stat-box{background:#11111b;border-radius:8px;padding:10px;text-align:center}
.stat-box .num{font-size:1.4rem;font-weight:bold;color:#00ffe7}
.stat-box .lbl{font-size:.72rem;color:#6c7086;margin-top:2px}
.stat-box.warn .num{color:#f38ba8}
.stat-box.speed .num{color:#a6e3a1}
#legend{margin-top:14px;font-size:.75rem;color:#6c7086;line-height:1.8}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}
#dlbtn{display:none;margin-top:12px;text-align:center}
#dlbtn a{color:#f38ba8;font-size:.9rem}
</style>
</head>
<body>
<h1>🎯 MotionScope</h1>
<p class="sub">MOG2 &middot; SORT + Re-ID Buffer &middot; Mahalanobis Anomaly &middot; Dwell &amp; Speed Tracking &middot; Trajectory Trails</p>
<div class="wrap">
  <div class="panel">
    <label>📥 Video file</label>
    <input type="file" id="vid" accept="video/*">
    <label>Min contour area: <span class="val" id="aval">2000</span></label>
    <input type="range" id="area" min="100" max="5000" value="2000" step="100"
           oninput="document.getElementById('aval').textContent=this.value">
    <label>Trail length (frames): <span class="val" id="tval">60</span></label>
    <input type="range" id="trail" min="10" max="120" value="60" step="5"
           oninput="document.getElementById('tval').textContent=this.value">
    <button onclick="start()">▶ Analyse</button>

    <div class="stat-grid" style="margin-top:16px">
      <div class="stat-box"><div class="num" id="s-active">—</div><div class="lbl">Active Tracks</div></div>
      <div class="stat-box"><div class="num" id="s-frame">—</div><div class="lbl">Frame</div></div>
      <div class="stat-box warn"><div class="num" id="s-anom">—</div><div class="lbl">⚠ Anomalies</div></div>
      <div class="stat-box"><div class="num" id="s-reid">—</div><div class="lbl">🔁 Re-IDs</div></div>
      <div class="stat-box speed"><div class="num" id="s-speed">—</div><div class="lbl">Avg Speed px/f</div></div>
      <div class="stat-box speed"><div class="num" id="s-dwell">—</div><div class="lbl">Max Dwell (frm)</div></div>
    </div>

    <div id="legend">
      <span class="dot" style="background:#00ffe7"></span>Normal track<br>
      <span class="dot" style="background:#ff3838"></span>Anomalous track (high Mahalanobis score)<br>
      <span class="dot" style="background:#cba6f7"></span>Re-identified track (ID recovered after occlusion)<br>
      <span class="dot" style="background:#a6e3a1"></span>Dwell bar (green stripe = time tracked)
    </div>
    <div id="dlbtn"><a id="dlink" href="/download">⬇ Download output video</a></div>
  </div>

  <div class="panel">
    <img id="stream" src="/stream" alt="Live preview">
  </div>
</div>

<script>
let poll;
function start(){
  const f=document.getElementById('vid').files[0];
  if(!f){alert('Please select a video file first.');return;}
  const fd=new FormData();
  fd.append('video',f);
  fd.append('min_area',document.getElementById('area').value);
  fd.append('trail_len',document.getElementById('trail').value);
  document.getElementById('dlbtn').style.display='none';
  fetch('/process',{method:'POST',body:fd})
    .then(r=>r.json()).then(d=>{
      if(d.ok){
        document.getElementById('stream').src='/stream?'+Date.now();
        clearInterval(poll);
        poll=setInterval(checkDone,1200);
      }
    });
}
function set(id,v){document.getElementById(id).textContent=v;}
function checkDone(){
  fetch('/status').then(r=>r.json()).then(d=>{
    set('s-active', d.active);
    set('s-frame',  d.frame);
    set('s-anom',   d.anomalies);
    set('s-reid',   d.reids);
    set('s-speed',  d.avg_speed);
    set('s-dwell',  d.max_dwell);
    if(d.done){
      clearInterval(poll);
      document.getElementById('dlbtn').style.display='block';
    }
  });
}
</script>
</body>
</html>"""

# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/process", methods=["POST"])
def process():
    global _processing, _stats, _frame_buf
    if _processing:
        return jsonify(ok=False, msg="Already processing")
    vid = request.files.get("video")
    if not vid:
        return jsonify(ok=False, msg="No file")
    path = "/tmp/input_video"
    vid.save(path)
    min_area  = float(request.form.get("min_area", 2000))
    trail_len = float(request.form.get("trail_len", 60))
    with _lock:
        _frame_buf  = None
        _processing = True
        _stats      = {"active":0,"frame":0,"done":False,"anomalies":0,
                       "reids":0,"avg_speed":0.0,"max_dwell":0}
    t = threading.Thread(target=process_video,
                         args=(path, min_area, trail_len), daemon=True)
    t.start()
    return jsonify(ok=True)

@app.route("/status")
def status():
    with _lock:
        return jsonify(**_stats)

def gen_frames():
    blank = np.zeros((240, 426, 3), dtype=np.uint8)
    cv2.putText(blank, "Waiting for input...", (30,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,200), 2)
    _, j = cv2.imencode(".jpg", blank)
    blank_jpg = j.tobytes()
    while True:
        with _lock:
            frame = _frame_buf
        data = frame if frame else blank_jpg
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        time.sleep(0.04)

@app.route("/stream")
def stream():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/download")
def download():
    return send_file("/tmp/output.mp4", as_attachment=True,
                     download_name="motionscope_output.mp4")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, threaded=True)
