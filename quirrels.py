import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime

# --- Settings ---
video_path = '20241023_TrepS_01_in (2).MOV'
video_path = 'video.mp4'
threshold = 30
scale = 1
window_size = 5         # moving average window
display_every = 1     # update display every N frames
pixel_threshold = 10000  # update if changed pixels exceed this

# --- Open video ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
    
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
prev_gray = None
motion_counts = []
frame_idx = 0
first_frame = True  # Flag to force display on first frame
start_time = datetime.datetime.now()
time = start_time

# --- Setup Matplotlib for plotting ---
plt.ion()
fig, ax = plt.subplots(figsize=(8, 3))
line_raw, = ax.plot([], [], color='lightgray', lw=1, label='Raw Motion')
line_smooth, = ax.plot([], [], color='tab:red', lw=2, label=f'Smoothed ({window_size})')
ax.set_title("Motion Intensity Over Time")
ax.set_xlabel("Frame Index")
ax.set_ylabel("Changed Pixel Count")
ax.grid(True)
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.legend()
fig.canvas.draw()
fig.show()

print("Press 'q' in the video window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Optional resize
    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale)

    rgb = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        # --- Motion detection ---
        diff = cv2.absdiff(gray, prev_gray)
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        count_t = np.sum(mask > 0)
        motion_counts.append(count_t)

        # --- Apply moving average ---
        if len(motion_counts) >= window_size:
            smooth = np.convolve(motion_counts, np.ones(window_size)/window_size, mode='valid')
            x_smooth = np.arange(window_size-1, len(motion_counts))
        else:
            smooth = []
            x_smooth = []

        # --- Decide if we should update display ---
        update_display = first_frame or frame_idx % display_every == 0 or count_t > pixel_threshold

        if update_display:
            perc = frame_idx/total_frames*100
            print(f"Frame {frame_idx}/{total_frames}")
            print(f"{perc:.2f}% done")
            print(f"Working for {(datetime.datetime.now() - start_time).total_seconds()} seconds already")
            last_interval = (datetime.datetime.now() - time).total_seconds()
            print(f"Last interval took {last_interval:.2f} seconds")
            if frame_idx > 0:
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                est_total = elapsed / frame_idx * total_frames
                est_remaining = max(0, est_total - elapsed)
                print(f"Estimated time remaining: {est_remaining/60:.2f} minutes")
            else:
                print("Estimated time remaining: calculating...")
            time = datetime.datetime.now()
            first_frame = False  # Reset flag after first display

            # Convert gray & mask to 3-channel
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Combine RGB | Gray | Mask
            combined = np.hstack((rgb, gray_bgr, mask_bgr))

            # Overlay text
            cv2.putText(combined, f"Frame: {frame_idx}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(combined, f"Changed Pixels: {count_t}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Show video
            cv2.imshow(f"Motion Detection (RGB | Gray | Mask) Threshold: {threshold}", combined)

            # Update plot
            xdata = np.arange(len(motion_counts))
            line_raw.set_data(xdata, motion_counts)
            if len(smooth) > 0:
                line_smooth.set_data(x_smooth, smooth)

            ax.set_xlim(0, max(100, len(motion_counts)))
            ax.set_ylim(0, max(10, np.max(motion_counts)*1.1))
            fig.canvas.draw()
            fig.canvas.flush_events()

            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("Stopped by user.")
                break

    prev_gray = gray.copy()
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()

