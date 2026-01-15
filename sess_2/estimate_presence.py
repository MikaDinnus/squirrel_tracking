import json, os
import cv2 as cv

DETECTIONS_JSON = "sess_2/detections_report.json"
VIDEO_PATH      = "sess_2/videos/sqr_dattsosib_2.mp4"

cap = cv.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Konnte Video nicht öffnen: {VIDEO_PATH}")
fps = cap.get(cv.CAP_PROP_FPS) or 30.0
cap.release()
print(f"FPS: {fps:.3f}")

with open(DETECTIONS_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)
data = sorted(data, key=lambda d: (d.get("frame", -1), d.get("image","")))

segments = []
current_start = None

def end_segment(end_frame):
    global segments, current_start
    if current_start is not None and end_frame is not None and end_frame >= current_start:
        segments.append((current_start, end_frame))
    current_start = None

prev_state = None
for item in data:
    fr = item.get("frame")
    state = bool(item.get("squirrel_detected", False))
    if prev_state is None:
        if state: 
            current_start = fr
        prev_state = state
        continue

    if (not prev_state) and state:
        current_start = fr
    elif prev_state and (not state):
        end_segment(fr)

    prev_state = state

if prev_state and current_start is not None:
    last_true_frame = max([d["frame"] for d in data if d.get("squirrel_detected", False)], default=current_start)
    end_segment(last_true_frame)

total_sec = 0.0
print("\nGefundene Präsenz-Segmente:")
for i, (a, b) in enumerate(segments, 1):
    dur = (b - a) / fps
    total_sec += dur
    print(f"  {i:02d}: Frames {a}–{b}  |  Dauer ~ {dur:.2f} s")

print(f"\nGesamtdauer (geschätzt): ~ {total_sec:.2f} s")
