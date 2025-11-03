import os, glob, json, re
import cv2 as cv
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate
from PIL import Image

print("################ START RUNNING GROUNDING DINO ################")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using cuda' if torch.cuda.is_available() else 'Using cpu')

GROUNDING_DINO_WEIGHTS = "sess_2/weights/groundingdino_swint_ogc.pth"
GROUNDING_DINO_CONFIG  = "sess_2/weights/GroundingDINO_SwinT_OGC.py"

INPUT_DIR   = "sess_2/motion_frames"
OUTPUT_DIR  = "sess_2/frames_out"
CAPTION     = "squirrel . animal"
BOX_THRESH  = 0.40
TEXT_THRESH = 0.25
REPORT_JSON = "sess_2/detections_report.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = load_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_WEIGHTS, device=device)

image_paths = sorted([p for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff")
                      for p in glob.glob(os.path.join(INPUT_DIR, ext))])

report = []
frame_re = re.compile(r"(\d+)")

for p in image_paths:
    image_source, image = load_image(p)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=CAPTION,
        box_threshold=BOX_THRESH,
        text_threshold=TEXT_THRESH,
        device=str(device)
    )

    rgb_annot = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    base = os.path.splitext(os.path.basename(p))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}_gdino.jpg")
    Image.fromarray(rgb_annot).save(out_path)

    m = frame_re.search(base)
    frame_num = int(m.group(1)) if m else None

    dets = [{"label": str(ph), "score": float(sc)} for ph, sc in zip(phrases, logits)]

    squirrel_detected = any("squirrel" in str(ph).lower() for ph in phrases)

    report.append({
        "frame": frame_num,
        "image": os.path.basename(p),
        "detections": dets,
        "squirrel_detected": squirrel_detected
    })

with open(REPORT_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"done. JSON: {REPORT_JSON}")
