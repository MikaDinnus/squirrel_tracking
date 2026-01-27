import sys
import os

if len(sys.argv) < 2:
    sys.exit(1)

video_input = sys.argv[1]
print(f"--- Pipeline started for: {video_input} ---")

with open("sess_2/save_motion_frames.py") as file:
    exec(file.read())

with open("sess_2/run_gd.py") as file:
    exec(file.read())

with open("sess_2/run_llama.py") as file:
    exec(file.read())
    
with open("sess_2/estimate_presence.py") as file:
    exec(file.read())