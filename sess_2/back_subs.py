import sys, time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

METHOD = "mog2"
BIN_THRESH = 25
DISPLAY_WIDTH = 360
DRAW_EVERY_MS = 10
LOG_EVERY_N = 100
PLOT_OUT = "final.png"

cap = cv.VideoCapture("sess_2/videos/sqr_dattsosib_2.mp4")

ok, frame = cap.read()
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
cv.destroyAllWindows()

plt.ioff()
fig.savefig(PLOT_OUT, dpi=150)
print(f"Gespeichert: {PLOT_OUT}")
plt.show()
