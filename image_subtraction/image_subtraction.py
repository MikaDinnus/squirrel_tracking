import cv2
import numpy as np
import matplotlib.pyplot as plt

video_path = "snippet1.mp4"
difference_threshold = 30

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Error: Cannot open video.")

ret, prev_frame = cap.read()
if not ret:
    raise IOError("Error: Cannot read first frame.")

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
changed_counts = []

# Zielbreite für die Anzeige
display_width = 600

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_gray)
    _, thresh = cv2.threshold(diff, difference_threshold, 255, cv2.THRESH_BINARY)

    changed_pixels = int(np.sum(thresh > 0))
    changed_counts.append(changed_pixels)

    diff_display = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # In 3-Kanal-Bilder umwandeln
    gray_col  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    diff_col  = cv2.cvtColor(diff_display, cv2.COLOR_GRAY2BGR)

    # *** BEIDE BILDER AUF SELBE BREITE RESIZEN ***
    height = int(frame.shape[0] * (display_width / frame.shape[1]))
    gray_col = cv2.resize(gray_col, (display_width, height))
    diff_col = cv2.resize(diff_col, (display_width, height))

    # Labels darauf schreiben
    cv2.putText(gray_col, "GRAY", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(diff_col, "DIFF", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    combined = np.hstack((gray_col, diff_col))
    cv2.imshow("Left = Grayscale | Right = Difference", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_gray = gray

cap.release()
cv2.destroyAllWindows()

plt.figure(figsize=(12, 4))
plt.plot(changed_counts)
plt.xlabel("Frame Index")
plt.ylabel("Changed Pixels")
plt.title(f"Movement Intensity (Threshold {difference_threshold})")
plt.grid(True)
# plt.show()
# export as PNG
plt.savefig(f"movement_intensity_threshold_{difference_threshold}.png", dpi=300)
