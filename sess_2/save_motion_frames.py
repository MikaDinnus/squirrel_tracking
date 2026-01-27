import sys, time, os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import csv 

# Use a non-interactive backend for matplotlib to prevent crashes
plt.switch_backend('Agg') 

print("################ START DETECTING MOTION FRAMES (HEADLESS) ################")

# --- INPUT HANDLING ---
if len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]
else:
    VIDEO_PATH = "sess_2/videos/sqr_dattsosib_1.mp4"
    print(f"[WARN] No argument provided. Using default: {VIDEO_PATH}")

print(f"Processing Video: {VIDEO_PATH}")

METHOD = "mog2"
BIN_THRESH = 25
# DISPLAY_WIDTH = 360  <-- Not needed in headless
# DRAW_EVERY_MS = 10   <-- Not needed in headless
LOG_EVERY_N = 100
PLOT_OUT = "final.png" # You might want to make this dynamic if running multiple clips, e.g. based on video name

SERIES_CSV = "sess_2/changed_series.csv"
series_rows = [("frame","changed_percent")]

EVENT_THRESHOLD = 2.5
EVENT_DIR = "sess_2/motion_frames"
os.makedirs(EVENT_DIR, exist_ok=True)

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")

cap = cv.VideoCapture(VIDEO_PATH)
cap_seek = cv.VideoCapture(VIDEO_PATH)

ok, frame = cap.read()
if not ok:
    raise RuntimeError(f"Konnte erstes Frame nicht lesen aus {VIDEO_PATH}.")

gray_prev = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

if METHOD == "mog2":
    bg = cv.createBackgroundSubtractorMOG2(history=1000, varThreshold=16, detectShadows=True)

# Setup Plot (but do not show it)
fig, ax = plt.subplots(figsize=(8, 3))
line, = ax.plot([], [], linewidth=1.5)
ax.set_xlabel("Frame")
ax.set_ylabel("Changed pixels [%]")
ax.set_ylim(0, 15)

xdata = []
ydata = []

frame_idx = 0
prev_changed = None
pre_event_saved = False
PRE_OFFSET = 200

# --- REMOVED WINDOW CREATION ---
# cv.namedWindow("frame", cv.WINDOW_NORMAL)
# cv.namedWindow("motion", cv.WINDOW_NORMAL)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if METHOD == "diff":
        diff = cv.absdiff(gray_prev, gray)
        _, mask = cv.threshold(diff, BIN_THRESH, 255, cv.THRESH_BINARY)
        gray_prev = gray
    else:
        fg = bg.apply(gray)
        mask = np.where(fg == 255, 255, 0).astype(np.uint8)

    changed = float(np.count_nonzero(mask)) / mask.size * 100.0
    series_rows.append((frame_idx, changed))

    # --- REMOVED IMAGE RESIZING AND SHOWING ---
    # cv.imshow("frame", _small(frame))
    # cv.imshow("motion", _small(mask))

    frame_idx += 1
    xdata.append(frame_idx)
    ydata.append(changed)

    # Event Detection Logic
    if prev_changed is None:
        prev_changed = changed
    else:
        crossed_up   = (prev_changed < EVENT_THRESHOLD) and (changed >= EVENT_THRESHOLD)
        crossed_down = (prev_changed >= EVENT_THRESHOLD) and (changed < EVENT_THRESHOLD)
        if crossed_up or crossed_down:
            out_path = os.path.join(EVENT_DIR, f"event_{frame_idx:06d}_{changed:.2f}pct.jpg")
            cv.imwrite(out_path, frame)

            if not pre_event_saved:
                target_idx = frame_idx - PRE_OFFSET
                if target_idx >= 0:
                    cap_seek.set(cv.CAP_PROP_POS_FRAMES, target_idx)
                    ok2, pre_frame = cap_seek.read()
                    if ok2 and pre_frame is not None:
                        pre_path = os.path.join(
                            EVENT_DIR, f"event_{target_idx:06d}_pre{PRE_OFFSET}.jpg"
                        )
                        cv.imwrite(pre_path, pre_frame)
                    pre_event_saved = True
        prev_changed = changed

    if frame_idx % LOG_EVERY_N == 0:
        # print(f"Processing frame {frame_idx}...") # Optional log
        pass

    # --- REMOVED LIVE PLOTTING AND WAITKEY ---
    # if cv.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cap_seek.release()
cv.destroyAllWindows() # Safe to keep, does nothing if no windows exist

# Update plot data once at the end and save
line.set_data(xdata, ydata)
ax.set_xlim(0, frame_idx)
ax.relim() 
ax.autoscale_view()

# Generate a unique filename for the plot so they don't overwrite each other
plot_filename = f"plot_{os.path.basename(VIDEO_PATH)}.png"
plot_path = os.path.join("sess_2", plot_filename)
fig.savefig(plot_path, dpi=150)
print(f"Gespeichert: {plot_path}")

# plt.show()  <-- CRITICAL: REMOVED

with open(SERIES_CSV, "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerows(series_rows)

print("Motion frames saved successfully.")