import os, glob, json, re, csv
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from gradio_client import Client, handle_file
import textwrap

print("################ START RUNNING LLAMA DESCRIPTION & PLOTTING ################")

INPUT_DIR = "sess_2/motion_frames"
GD_REPORT_JSON = "sess_2/detections_report.json"
FINAL_REPORT = "sess_2/combined_report.json"
SERIES_CSV = "sess_2/changed_series.csv"
OUT_PNG = "sess_2/final_marked_described.png"

PRE_PROMPT_TEXT = ("""
    Return a classification of the picture in the following schema: Only return the values, no extra text. Only choose one value per field.
    No ** or other markdown formatting. Not title or anything adding, just the following:

    squirrel_present: <True/False>
    prop_research_type: <disco ball/nut/cups/none> 
    view: <inside/outside>
""")

PROMPT_TEXT = ("""
    Return a classification of the picture in the following schema: Only return the values, no extra text. Only choose one value per field.
    No ** or other markdown formatting. Not title or anything adding, just the following:
               
    squirrel_present: <True/False>
    squirrel_position: <Inside/Door/Outside/Unknown>
    squirrel_activity: <Eating/Climbing/Running/Resting/Unknown>
    prop_present: <Disco ball/Nut/Cups>
    prop_used: <Cup flipped/Nut eaten/None>

""")

client = Client("huggingface-projects/llama-3.2-vision-11B")

# 1. Load existing GD Report
if os.path.exists(GD_REPORT_JSON):
    with open(GD_REPORT_JSON, "r", encoding="utf-8") as f:
        gd_data = json.load(f)
else:
    gd_data = []

report_map = {item['frame']: item for item in gd_data}

image_paths = sorted([p for ext in ("*.jpg","*.jpeg","*.png") 
                      for p in glob.glob(os.path.join(INPUT_DIR, ext))])

frame_re = re.compile(r"(\d+)")

print(f"Processing images and updating report...")

for p in image_paths:
    base = os.path.splitext(os.path.basename(p))[0]
    m = frame_re.search(base)
    frame_num = int(m.group(1)) if m else None
    
    if frame_num is None: 
        continue

    is_pre = "_pre" in base
    
    # Ensure entry exists
    if frame_num not in report_map:
        report_map[frame_num] = {"frame": frame_num}
    
    entry = report_map[frame_num]
    entry["image_name"] = os.path.basename(p)
    entry["is_pre_event"] = is_pre

    # --- PATH A: Pre-Event Image ---
    if is_pre:
        print(f"Frame {frame_num} (Pre-Event): Asking Llama...")
        try:
            result = client.predict(
                message={"text": PRE_PROMPT_TEXT, "files": [handle_file(p)]},
                max_new_tokens=60,
                api_name="/chat"
            )
            desc = result.strip()
            print(f" -> {desc}")
            entry["pre_description"] = desc
        except Exception as e:
            print(f"Error Pre-Frame {frame_num}: {e}")
            entry["pre_description"] = "Error"

    # --- PATH B: Standard Event ---
    else:
        squirrel_detected = entry.get("squirrel_detected", False)
        
        if squirrel_detected:
            print(f"Frame {frame_num}: Squirrel detected. Asking Llama...")
            try:
                result = client.predict(
                    message={"text": PROMPT_TEXT, "files": [handle_file(p)]},
                    max_new_tokens=60,
                    api_name="/chat"
                )
                raw_output = result.strip()
                print(f" -> {raw_output}")
                
                # Save raw output as fallback
                entry["vlm_raw"] = raw_output
                
                # Robust Parsing
                lines = raw_output.split('\n')
                for line in lines:
                    if ':' in line:
                        key, val = line.split(':', 1)
                        # Clean key (remove ** or spaces)
                        clean_key = key.replace('*', '').strip()
                        clean_val = val.strip()
                        entry[clean_key] = clean_val
                        
            except Exception as e:
                print(f"Error Frame {frame_num}: {e}")

# 2. Save Updated Report
final_report_list = sorted(report_map.values(), key=lambda x: x['frame'])

with open(GD_REPORT_JSON, "w", encoding="utf-8") as f:
    json.dump(final_report_list, f, ensure_ascii=False, indent=2)
    print(f"Updated {GD_REPORT_JSON} with VLM data.")

# --- PLOTTING ---
print("Creating chart...")

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

display_keys = [
    "squirrel_present", 
    "squirrel_position", 
    "squirrel_activity", 
    "prop_present", 
    "prop_used"
]

for item in final_report_list:
    fr = item.get("frame")
    if fr is None: continue
    
    is_squirrel = item.get("squirrel_detected")
    is_pre = item.get("is_pre_event", False)
    
    # Construct Label
    label_text = ""
    if is_pre:
        label_text = item.get("pre_description", "")
    else:
        # Try to build from keys first
        lines = []
        found_keys = False
        for k in display_keys:
            if k in item:
                lines.append(f"{k}: {item[k]}")
                found_keys = True
        
        if found_keys:
            label_text = "\n".join(lines)
        else:
            # Fallback to raw output if keys missing but data exists
            label_text = item.get("vlm_raw", "")

    # Determine Color
    if is_pre:
        color = "orange"
        alpha = 0.8
        box_ec = "orange"
    else:
        color = "g" if is_squirrel else "r"
        alpha = 0.6 if is_squirrel else 0.2
        box_ec = "green"

    plt.axvline(fr, color=color, linewidth=1, alpha=alpha)

    # Draw Text Box
    if label_text:
        wrapped_desc = "\n".join(textwrap.wrap(label_text, width=30))
        y_pos = text_height_cycle[text_idx % len(text_height_cycle)]
        
        plt.annotate(
            wrapped_desc,
            xy=(fr, 10),
            xytext=(fr, y_pos),
            arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.5),
            fontsize=7,
            rotation=0,
            horizontalalignment='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=box_ec, alpha=0.8)
        )
        text_idx += 1

leg_motion = mlines.Line2D([], [], color='blue', label='Motion', alpha=0.3)
leg_true = mlines.Line2D([], [], color='g', label='GD: Squirrel Found')
leg_pre = mlines.Line2D([], [], color='orange', label='Pre-Event')

plt.legend(handles=[leg_motion, leg_true, leg_pre], loc="upper left", frameon=True)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"Done. Chart saved to: {OUT_PNG}")