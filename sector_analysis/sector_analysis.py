import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

BOX_CENTER = None
ENTRY_RADIUS = 350
N_SECTORS = 12

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_TO_VIDEO = os.path.join(BASE_DIR, "sam_outside.mp4")


def click_event(event, x, y, flags, param):
    global BOX_CENTER
    if event ==cv2.EVENT_LBUTTONDOWN:
        BOX_CENTER = (x, y)
        print(f"Box center set to: {BOX_CENTER}")

def contour_centroid(contour):
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)

def get_sector(point, center, n_sectors):
    px, py = point
    cx, cy = center

    angle = np.arctan2(py - cy, px - cx)
    angle_deg = (np.degrees(angle) + 360) % 360

    sector_size = 360 / n_sectors
    sector = int(angle_deg // sector_size)
    return sector, angle_deg

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def draw_sectors(frame, center, radius, n_sectors):
    cx, cy = center
    
    for i in range(n_sectors):
        angle = 2 * np.pi * i / n_sectors
        
        x = int(cx + radius * np.cos(angle))
        y = int(cy + radius * np.sin(angle))
        
        cv2.line(frame, center, (x, y), (255, 255, 0), 2)
        
        # Draw sector numbers at the edge
        label_x = int(cx + (radius + 30) * np.cos(angle))
        label_y = int(cy + (radius + 30) * np.sin(angle))
        cv2.putText(frame, str(i), (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

def highlight_sector(frame, center, radius, sector, n_sectors):
    if sector is None:
        return
    
    # Ensure sector is within valid range
    sector = sector % n_sectors
    
    cx, cy = center

    angle1 = 2 * np.pi * sector / n_sectors
    angle2 = 2 * np.pi * (sector + 1) / n_sectors

    points = [center]

    for a in np.linspace(angle1, angle2, 30):
        x = int(cx + radius * np.cos(a))
        y = int(cy + radius * np.sin(a))
        points.append((x, y))

    points = np.array(points, dtype=np.int32)

    try:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], (0, 255, 255))
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    except Exception as e:
        print(f"Error highlighting sector: {e}")

def draw_direction(frame, center, point):
    cv2.line(frame, center, point, (0, 255, 255), 2)



cap = cv2.VideoCapture(PATH_TO_VIDEO)

if not cap.isOpened():
    print("ERROR: failed to open video")
    exit()

# Read first frame to set box center
ret, first_frame = cap.read()
if not ret:
    print("ERROR: failed to read first frame")
    exit()

WINDOW_NAME = "Click to set box center"

cv2.imshow(WINDOW_NAME, first_frame)
cv2.setMouseCallback(WINDOW_NAME, click_event)

print("Please click on the video frame to set the box center.")

while True:
    cv2.imshow(WINDOW_NAME, first_frame)

    key = cv2.waitKey(1)

    if BOX_CENTER is not None:
        break


cv2.destroyAllWindows()

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame after setting center

current_sector = None
sector_counts = {i: 0 for i in range(N_SECTORS)}  # Track how many frames in each sector

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        tolerance = 30  # Increased from 10 for better color detection

        lower = np.array([168 - tolerance, 93 - tolerance, 247 - tolerance])
        upper = np.array([168 + tolerance, 93 + tolerance, 247 + tolerance])

        mask = cv2.inRange(frame, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            centroid = contour_centroid(largest)

            if centroid is not None:

                dist = distance(centroid, BOX_CENTER)

                draw_direction(frame, BOX_CENTER, centroid)

                if dist < ENTRY_RADIUS:
                    # Get the sector of the centroid every frame
                    current_sector, current_angle = get_sector(centroid, BOX_CENTER, N_SECTORS)
                    sector_counts[current_sector] += 1
                    print(f"Centroid in sector {current_sector} at angle {current_angle:.2f} degrees")

                if current_sector is not None:
                        cv2.putText(frame,
                                    f"Current sector: {current_sector}",
                                    (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0,255,255),
                                    2)

                cv2.circle(frame, centroid, 5, (0, 255, 0), -1)
        
        cv2.circle(frame, BOX_CENTER, ENTRY_RADIUS, (255, 0, 0), 2)

        draw_sectors(frame, BOX_CENTER, ENTRY_RADIUS, N_SECTORS)

        highlight_sector(frame, BOX_CENTER, ENTRY_RADIUS, current_sector, N_SECTORS)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    except Exception as e:
        print(f"Error processing frame: {e}")
        continue

cap.release()
cv2.destroyAllWindows()

# Display statistics in circular heatmap
print("\nSector statistics:")
for sector, count in sector_counts.items():
    if count > 0:
        print(f"Sector {sector}: {count} frames")

# Create circular heatmap
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Normalize counts for color mapping (0 to 1)
max_count = max(sector_counts.values()) if max(sector_counts.values()) > 0 else 1
normalized_counts = [sector_counts[i] / max_count for i in range(N_SECTORS)]

# Color map from white to red
from matplotlib.colors import LinearSegmentedColormap
colors_list = ['white', 'red']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('white_red', colors_list, N=n_bins)

# Draw sectors as pie slices
theta = np.linspace(0, 2 * np.pi, N_SECTORS, endpoint=False)
width = 2 * np.pi / N_SECTORS
radius = 1

for i in range(N_SECTORS):
    color = cmap(normalized_counts[i])
    ax.bar(theta[i], radius, width=width, bottom=0, color=color, edgecolor='black', linewidth=0.5)
    
    # Add sector number at the sector boundary (to match video frame positioning)
    ax.text(theta[i], radius + 0.2, str(i), ha='center', va='center', fontsize=8, fontweight='bold')

ax.set_ylim(0, 1.3)
ax.set_yticks([])
ax.set_xticks(theta)
ax.set_xticklabels([])
ax.set_theta_offset(0)
ax.set_theta_direction(-1)  # Clockwise direction to match video frame
ax.set_title('Squirrel Distribution by Sector\n(White: No appearance, Red: High appearance)', fontsize=14, pad=20)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_count))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.1, fraction=0.046)
cbar.set_label('Number of Frames', fontsize=12)

plt.tight_layout()
plt.show()