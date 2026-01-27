import os
import sys
import glob
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fast_video_binary import FastVideoSearcher

RAW_VIDEO_PATH = "C:/Users/itsmi/OneDrive/Desktop/DATTSOSIB/videos/raw/Rahn_05_7.mov"  # Or "C:/.../Rahn_05_7.mov"
OUTPUT_DIR = "sess_3/output"
SESS_2_MAIN = "sess_2/__main__.py"

def run_pipeline():
    print(f"=== STEP 1: Running Fast Binary Search on {RAW_VIDEO_PATH} ===")
    
    searcher = FastVideoSearcher(RAW_VIDEO_PATH, OUTPUT_DIR)
    searcher.run()

    print("\n=== STEP 2: Processing generated clips with sess_2 pipeline ===")
    
    clip_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")))
    
    if not clip_files:
        print("No clips were generated. Exiting.")
        return

    for clip_path in clip_files:
        print(f"\n>>> Processing clip: {clip_path}")
        
        cmd = [sys.executable, SESS_2_MAIN, clip_path]
        
        subprocess.run(cmd, check=True)
        print(f">>> Finished processing {clip_path}")

if __name__ == "__main__":
    run_pipeline()