import sys, time, os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import csv 

print("################ START DETECTING MOTION FRAMES ################")

METHOD = "mog2"
BIN_THRESH = 25
DISPLAY_WIDTH = 360
DRAW_EVERY_MS = 10
LOG_EVERY_N = 100
PLOT_OUT = "final.png"

SERIES_CSV = "sess_2/changed_series.csv"
series_rows = [("frame","changed_percent")]

EVENT_THRESHOLD = 2.5
EVENT_DIR = "sess_2/motion_frames"
os.makedirs(EVENT_DIR, exist_ok=True)

VIDEO_PATH = "sess_2/videos/sqr_dattsosib_1.mp4"
cap = cv.VideoCapture(VIDEO_PATH)
cap_seek = cv.VideoCapture(VIDEO_PATH)

ok, frame = cap.read()
if not ok:
    raise RuntimeError("Konnte erstes Frame nicht lesen.")

gray_prev = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

if METHOD == "mog2":
    bg = cv.createBackgroundSubtractorMOG2(history=1000, varThreshold=16, detectShadows=True)

plt.ion()
fig, ax = plt.subplots(figsize=(8, 3))
line, = ax.plot([], [], linewidth=1.5)
ax.set_xlabel("Frame")
ax.set_ylabel("Changed pixels [%]")
ax.set_ylim(0, 15)

xdata = []
ydata = []
last_draw = 0.0

log_frames = []
log_values = []

frame_idx = 0
cv.namedWindow("frame", cv.WINDOW_NORMAL)
cv.namedWindow("motion", cv.WINDOW_NORMAL)
cv.resizeWindow("frame", DISPLAY_WIDTH, int(DISPLAY_WIDTH * 9/16))
cv.resizeWindow("motion", DISPLAY_WIDTH, int(DISPLAY_WIDTH * 9/16))

prev_changed = None

pre_event_saved = False
PRE_OFFSET = 200

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

    def _small(img):
        h, w = img.shape[:2]
        scale = DISPLAY_WIDTH / float(w)
        nh = max(1, int(h * scale))
        return cv.resize(img, (DISPLAY_WIDTH, nh), interpolation=cv.INTER_AREA)

    cv.imshow("frame", _small(frame))
    cv.imshow("motion", _small(mask))

    frame_idx += 1
    xdata.append(frame_idx)
    ydata.append(changed)

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
                    else:
                        print(f"[WARN] Konnte Vor-Event-Frame {target_idx} nicht lesen.")
                else:
                    print(f"[INFO] Weniger als {PRE_OFFSET} Frames seit Start – Vor-Event entfällt.")
                pre_event_saved = True
        prev_changed = changed

    if frame_idx % LOG_EVERY_N == 0:
        log_frames.append(frame_idx)
        log_values.append(changed)

    now = time.time()
    if (now - last_draw) * 1000 >= DRAW_EVERY_MS:
        line.set_data(xdata, ydata)
        ax.set_xlim(max(0, frame_idx - 600), frame_idx)
        fig.canvas.draw()
        fig.canvas.flush_events()
        last_draw = now

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap_seek.release()
cv.destroyAllWindows()

plt.ioff()
fig.savefig(PLOT_OUT, dpi=150)
print(f"Gespeichert: {PLOT_OUT}")
plt.show()

with open(SERIES_CSV, "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerows(series_rows)
