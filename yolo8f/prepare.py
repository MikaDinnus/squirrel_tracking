import json
import os
import cv2
import yaml
import shutil
import random
from tqdm import tqdm

# --- KONFIGURATION ---
JSON_FILE_PATH = './annotations.ndjson'     # Pfad zur Labelbox JSON
SCIEBO_VIDEO_PATH = './videos' # Ordner wo die Videos liegen
OUTPUT_DIR = './yolo_video_dataset'

# Deine Klassen exakt wie in Labelbox (Reihenfolge ist wichtig für ID!)
CLASSES = ["squirrel","nut","cup_empty", "cup_full", "disco_ball"]

VAL_SPLIT = 0.2  # 20% Validierung
# ---------------------

def create_dir_structure():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)

def convert_bbox(img_w, img_h, bbox):
    """Konvertiert Labelbox (top, left, h, w) zu YOLO (cx, cy, w, h)"""
    # Labelbox Video Export hat oft: top, left, width, height
    x_center = bbox['left'] + (bbox['width'] / 2.0)
    y_center = bbox['top'] + (bbox['height'] / 2.0)
    
    # Normalisieren (0-1)
    x = x_center / img_w
    y = y_center / img_h
    w = bbox['width'] / img_w
    h = bbox['height'] / img_h
    
    return (x, y, w, h)

def process_video_data():
    create_dir_structure()

    print("Lade JSON (NDJSON Modus)...")
    data = []
    with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Leere Zeilen überspringen
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print("Warnung: Eine Zeile konnte nicht gelesen werden (Skipped).")

    # Wir sammeln alle validen (Frame, Label) Paare erst, um sie später zu splitten
    all_samples = []

    print(f"Verarbeite {len(data)} Video-Einträge...")

    for entry in tqdm(data):
        # 1. Video Datei finden
        try:
            external_id = entry['data_row']['external_id']
            # Versuchen, das Video im Ordner zu finden
            video_path = os.path.join(SCIEBO_VIDEO_PATH, external_id)
            
            if not os.path.exists(video_path):
                print(f"Video nicht gefunden: {video_path} (Überspringe)")
                continue
                
        except KeyError:
            continue

        # 2. Labels extrahieren
        # Labelbox Video JSON ist oft tief verschachtelt
        try:
            # Projekt ID holen (erste verfügbare)
            project_id = list(entry['projects'].keys())[0]
            labels = entry['projects'][project_id]['labels']
            
            if not labels: continue

            # Iteriere über alle Label-Instanzen (falls Video mehrmals gelabelt wurde)
            for label_instance in labels:
                frames_data = label_instance['annotations']['frames']
                
                # Wir merken uns, welche Frames wir extrahieren müssen
                # frames_data ist ein Dict: "1": {objects...}, "2": {objects...}
                for frame_idx_str, frame_content in frames_data.items():
                    frame_number = int(frame_idx_str)
                    objects = frame_content.get('objects', {})
                    
                    if not objects: continue

                    # Speichere Infos für spätere Verarbeitung
                    all_samples.append({
                        'video_path': video_path,
                        'frame_number': frame_number, # Labelbox startet oft bei Frame 1
                        'objects': objects,
                        'base_filename': external_id
                    })

        except (KeyError, IndexError) as e:
            # print(f"Fehler bei Struktur von {external_id}: {e}")
            pass

    print(f"Insgesamt {len(all_samples)} gelabelte Frames gefunden. Starte Extraktion...")
    
    # Random Split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * (1 - VAL_SPLIT))

    # Cache für Video Capture Objekte, um nicht ständig neu zu öffnen
    cap_cache = {} 

    for i, sample in enumerate(tqdm(all_samples)):
        subset = 'train' if i < split_idx else 'val'
        
        vid_path = sample['video_path']
        f_num = sample['frame_number']
        
        # Video öffnen (Caching Strategie einfach)
        if vid_path not in cap_cache:
            cap_cache[vid_path] = cv2.VideoCapture(vid_path)
        
        cap = cap_cache[vid_path]
        
        # Zu Frame springen
        # ACHTUNG: Labelbox Frame 1 ist oft OpenCV Frame 0. 
        # Wir ziehen sicherheitshalber 1 ab. Prüfe das im Zweifel!
        opencv_frame_idx = f_num - 1 
        cap.set(cv2.CAP_PROP_POS_FRAMES, opencv_frame_idx)
        
        ret, frame = cap.read()
        if not ret:
            print(f"Konnte Frame {f_num} aus {os.path.basename(vid_path)} nicht lesen.")
            continue
            
        h_img, w_img, _ = frame.shape
        
        # Dateinamen generieren: VideoName_FrameNr
        base_name = os.path.splitext(sample['base_filename'])[0]
        out_name = f"{base_name}_frame{f_num}"
        
        img_out_path = os.path.join(OUTPUT_DIR, subset, 'images', f"{out_name}.jpg")
        txt_out_path = os.path.join(OUTPUT_DIR, subset, 'labels', f"{out_name}.txt")
        
        # Bild speichern
        cv2.imwrite(img_out_path, frame)
        
        # Labels verarbeiten
        yolo_lines = []
        # Objects ist ein Dict: { "UUID": { name: "car", bbox: {...} } }
        for obj_id, obj_data in sample['objects'].items():
            cls_name = obj_data['name']
            
            if cls_name in CLASSES:
                cls_id = CLASSES.index(cls_name)
                bbox = obj_data['bounding_box'] # top, left, width, height
                
                # Konvertieren
                y_bbox = convert_bbox(w_img, h_img, bbox)
                yolo_lines.append(f"{cls_id} {y_bbox[0]:.6f} {y_bbox[1]:.6f} {y_bbox[2]:.6f} {y_bbox[3]:.6f}")
        
        # Label Datei schreiben
        with open(txt_out_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

    # Cleanup
    for cap in cap_cache.values():
        cap.release()

    # YAML erstellen
    yaml_content = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'names': {i: name for i, name in enumerate(CLASSES)}
    }
    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print("\nFertig!")

if __name__ == "__main__":
    process_video_data()