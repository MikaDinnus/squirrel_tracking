import json, csv
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

print("################ MARKING GROUNDING DINO DETECTIONS ################")

SERIES_CSV = "sess_2/changed_series.csv"
DETECTIONS_JSON = "sess_2/detections_report.json"
OUT_PNG = "sess_2/final_marked.png"

frames, values = [], []
with open(SERIES_CSV, "r", encoding="utf-8") as f:
    r = csv.reader(f)
    header = next(r, None)
    for row in r:
        frames.append(int(row[0]))
        values.append(float(row[1]))

with open(DETECTIONS_JSON, "r", encoding="utf-8") as f:
    detections = json.load(f)

plt.figure(figsize=(8, 3))
plt.plot(frames, values, linewidth=1.5)
plt.xlabel("Frame")
plt.ylabel("Changed pixels [%]")
plt.ylim(0, 15)

for det in detections:
    fr = det.get("frame")
    if fr is None:
        continue
    color = "g" if det.get("squirrel_detected") else "r"
    plt.axvline(fr, color=color, linewidth=1, alpha=0.6)

leg_true = mlines.Line2D([], [], color='g', label='squirrel_detected=True')
leg_false = mlines.Line2D([], [], color='r', label='squirrel_detected=False')
plt.legend(handles=[leg_true, leg_false], loc="upper right", frameon=False)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"Gespeichert: {OUT_PNG}")
