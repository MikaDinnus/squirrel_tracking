import cv2
import numpy as np
import matplotlib.pyplot as plt

video_path = "./snippet1.mp4"
output_path = "processed_video.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Cannot open video.")

# Read the first frame to get original dimensions
ret, frame = cap.read()
if not ret:
    raise IOError("Cannot read the first frame.")

original_height, original_width = frame.shape[:2]

# Desired display width
display_width = 600
display_height = int(original_height * (display_width / original_width))

# VideoWriter (width*2 because we stack two frames side by side)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, fourcc, fps, (display_width*2, display_height))

backSub = cv2.createBackgroundSubtractorMOG2(history=4000, varThreshold=50, detectShadows=True)

changed_counts = []
min_area = 2000

# Reset capture to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg_mask = backSub.apply(gray)

    # noise reduction
    fg_mask = cv2.medianBlur(fg_mask, 5)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    changed_pixels = np.sum(fg_mask > 0)
    changed_counts.append(changed_pixels)

    # find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # bounding box for largest contour
    largest_cnt = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area and area > max_area:
            max_area = area
            largest_cnt = cnt

    # convert to color for display
    gray_col = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    fg_col = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

    filled_mask = np.zeros_like(fg_mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=-1)
    fg_col = cv2.cvtColor(filled_mask, cv2.COLOR_GRAY2BGR)


    # draw bounding box on both frames
    if largest_cnt is not None:
        x, y, w, h = cv2.boundingRect(largest_cnt)
        cv2.rectangle(gray_col, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(fg_col, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Resize frames for display
    gray_col = cv2.resize(gray_col, (display_width, display_height))
    fg_col = cv2.resize(fg_col, (display_width, display_height))

    # Add labels
    cv2.putText(gray_col, "GRAY", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(fg_col, "MOG2 BS", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Combine side by side
    combined = np.hstack((gray_col, fg_col))

    # Show and write frame
    cv2.imshow("squirrel tracking", combined)
    out.write(combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Plot movement intensity
plt.figure(figsize=(12, 4))
plt.plot(changed_counts, linewidth=1)
plt.xlabel("Frame Index")
plt.ylabel("Changed Pixels")
plt.title("Movement Intensity per Frame (Background Subtraction)")
plt.grid(True)
plt.savefig("movement_intensity_background_subtraction.png", dpi=300)
