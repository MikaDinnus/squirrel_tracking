import os, glob, json, re, csv
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from gradio_client import Client, handle_file
import textwrap

print("################ START RUNNING LLAMA DESCRIPTION & PLOTTING ################")

INPUT_DIR       = "sess_2/motion_frames"
GD_REPORT_JSON  = "sess_2/detections_report.json"
FINAL_REPORT    = "sess_2/combined_report.json"
SERIES_CSV      = "sess_2/changed_series.csv"
OUT_PNG         = "sess_2/final_marked_described.png"

PROMPT_TEXT = ("""
    Return a classification of the picture in the following schema:
               
    squirrel_present: <True/False>
    squirrel_position: <Inside/Door/Outside/Unknown>
    squirrel_activity: <Eating/Climbing/Running/Resting/Unknown>
    prop_present: <Disco ball/Nut/Cups>
    prop_used: <Cup flipped/Nut eaten/Disco ball spinning/Unknown>

""")

client = Client("huggingface-projects/llama-3.2-vision-11B")

if not os.path.exists(GD_REPORT_JSON):
    raise FileNotFoundError(f"Bitte zuerst run_gd.py ausführen! {GD_REPORT_JSON} fehlt.")

with open(GD_REPORT_JSON, "r", encoding="utf-8") as f:
    gd_data = json.load(f)

gd_lookup = {item['frame']: item for item in gd_data}

image_paths = sorted([p for ext in ("*.jpg","*.jpeg","*.png") 
                      for p in glob.glob(os.path.join(INPUT_DIR, ext))])

final_report = []
frame_re = re.compile(r"(\d+)")

print(f"Verarbeite Bilder basierend auf GD-Ergebnissen...")

for p in image_paths:
    base = os.path.splitext(os.path.basename(p))[0]
    m = frame_re.search(base)
    frame_num = int(m.group(1)) if m else None
    
    gd_info = gd_lookup.get(frame_num)
    
    description = None
    squirrel_detected = False

    if gd_info:
        squirrel_detected = gd_info.get("squirrel_detected", False)
        
        if squirrel_detected:
            try:
                print(f"Frame {frame_num}: Squirrel erkannt (GD). Frage Llama nach Beschreibung...")
                result = client.predict(
                    message={"text": PROMPT_TEXT, "files": [handle_file(p)]},
                    max_new_tokens=60,
                    api_name="/chat"
                )
                description = result.strip()
                print(f" -> {description}")
            except Exception as e:
                print(f"Fehler bei Frame {frame_num}: {e}")
                description = "Error generating description."
    
    final_report.append({
        "frame": frame_num,
        "image": os.path.basename(p),
        "squirrel_detected": squirrel_detected,
        "description": description
    })

with open(FINAL_REPORT, "w", encoding="utf-8") as f:
    json.dump(final_report, f, ensure_ascii=False, indent=2)


print("Erstelle erweitertes Diagramm...")

frames, values = [], []
if os.path.exists(SERIES_CSV):
    with open(SERIES_CSV, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            frames.append(int(row[0]))
            values.append(float(row[1]))

plt.figure(figsize=(12, 6)) 
plt.plot(frames, values, linewidth=1.5, label="Motion Intensity", color='blue', alpha=0.3)
plt.xlabel("Frame")
plt.ylabel("Changed pixels [%]")
plt.ylim(0, 20)

text_height_cycle = [12, 14, 16, 18] 
text_idx = 0

for item in final_report:
    fr = item.get("frame")
    if fr is None:
        continue
    
    is_squirrel = item.get("squirrel_detected")
    desc = item.get("description")

    color = "g" if is_squirrel else "r"
    alpha = 0.6 if is_squirrel else 0.2
    plt.axvline(fr, color=color, linewidth=1, alpha=alpha)

    if desc:
        wrapped_desc = "\n".join(textwrap.wrap(desc, width=30))
        
        y_pos = text_height_cycle[text_idx % len(text_height_cycle)]
        
        plt.annotate(
            wrapped_desc, 
            xy=(fr, 10),
            xytext=(fr, y_pos), 
            arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.5),
            fontsize=7,
            rotation=0,
            horizontalalignment='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8)
        )
        text_idx += 1

leg_motion = mlines.Line2D([], [], color='blue', label='Motion', alpha=0.3)
leg_true = mlines.Line2D([], [], color='g', label='GD: Squirrel Found')
leg_false = mlines.Line2D([], [], color='r', label='GD: No Squirrel')
plt.legend(handles=[leg_motion, leg_true, leg_false], loc="upper left", frameon=True)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"Fertig. Grafik mit Beschreibungen gespeichert unter: {OUT_PNG}")
