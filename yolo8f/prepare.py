import json
import os
import cv2
import yaml
import shutil
import random
from tqdm import tqdm
from collections import defaultdict

# --- KONFIGURATION ---
JSON_FILE_PATH = './annotations.ndjson'
SCIEBO_VIDEO_PATH = './videos'
OUTPUT_DIR = './yolo_video_dataset'

CLASSES = ["squirrel", "nut", "cup_empty", "cup_full", "disco_ball"]

VAL_SPLIT = 0.2
# ---------------------

def sanitize_filename(name):
    """Ersetzt Umlaute und Sonderzeichen für Windows/YOLO Kompatibilität."""
    replacements = {
        'ä': 'ae', 'ö': 'oe', 'ü': 'ue',
        'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue',
        'ß': 'ss', ' ': '_' 
    }
    for char, replacement in replacements.items():
        name = name.replace(char, replacement)
    
    # Entferne alles, was nicht Buchstabe, Zahl, _ oder - ist
    return "".join([c for c in name if c.isalnum() or c in ['_', '-']])

def create_dir_structure():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)

def convert_bbox(img_w, img_h, bbox):
    """Konvertiert Labelbox (top, left, h, w) zu YOLO (cx, cy, w, h)"""
    x_center = bbox['left'] + (bbox['width'] / 2.0)
    y_center = bbox['top'] + (bbox['height'] / 2.0)
    
    x = x_center / img_w
    y = y_center / img_h
    w = bbox['width'] / img_w
    h = bbox['height'] / img_h
    return (x, y, w, h)

def process_video_data():
    create_dir_structure()

    print("Lade JSON (NDJSON Modus)...")
    data = []
    # Set für schnellen Abgleich erstellen
    referenced_filenames = set()

    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        data.append(entry)
                        
                        # Sammle alle Dateinamen, die im JSON vorkommen
                        if 'data_row' in entry and 'external_id' in entry['data_row']:
                            referenced_filenames.add(entry['data_row']['external_id'])
                            
                    except json.JSONDecodeError:
                        print("Warnung: Eine Zeile konnte nicht gelesen werden.")
    except FileNotFoundError:
        print(f"FEHLER: Datei nicht gefunden: {JSON_FILE_PATH}")
        return

    # --- SCHRITT 1: Ungenutzte Videos löschen ---
    print(f"\n--- Phase 1: Bereinigung des Video-Ordners ---")
    if os.path.exists(SCIEBO_VIDEO_PATH):
        all_files = os.listdir(SCIEBO_VIDEO_PATH)
        deleted_count = 0
        
        for filename in all_files:
            file_path = os.path.join(SCIEBO_VIDEO_PATH, filename)
            
            # Überspringe Ordner, lösche nur Dateien
            if not os.path.isfile(file_path):
                continue
                
            # Wenn Datei NICHT im JSON steht -> Löschen
            if filename not in referenced_filenames:
                try:
                    os.remove(file_path)
                    print(f"Gelöscht (nicht im JSON): {filename}")
                    deleted_count += 1
                except OSError as e:
                    print(f"Fehler beim Löschen von {filename}: {e}")
        
        print(f"-> {deleted_count} ungenutzte Videos wurden gelöscht.")
    else:
        print(f"Warnung: Video Ordner '{SCIEBO_VIDEO_PATH}' existiert nicht.")
        return

    # --- SCHRITT 2: Verarbeitung der genutzten Videos ---
    print(f"\n--- Phase 2: Extraktion und Verarbeitung ---")
    
    all_samples = []
    for entry in tqdm(data, desc="Suche Labels"):
        try:
            external_id = entry['data_row']['external_id']
            video_path = os.path.join(SCIEBO_VIDEO_PATH, external_id)
            
            project_id = list(entry['projects'].keys())[0]
            labels = entry['projects'][project_id]['labels']
            if not labels: continue

            for label_instance in labels:
                frames_data = label_instance['annotations']['frames']
                for frame_idx_str, frame_content in frames_data.items():
                    frame_number = int(frame_idx_str)
                    objects = frame_content.get('objects', {})
                    if not objects: continue

                    all_samples.append({
                        'video_path': video_path,
                        'frame_number': frame_number,
                        'objects': objects,
                        'base_filename': external_id
                    })
        except (KeyError, IndexError):
            pass

    # Random Split zuweisen
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * (1 - VAL_SPLIT))
    
    for i, sample in enumerate(all_samples):
        sample['subset'] = 'train' if i < split_idx else 'val'

    # Gruppieren nach Video
    video_groups = defaultdict(list)
    for sample in all_samples:
        video_groups[sample['video_path']].append(sample)

    print(f"Verarbeite {len(video_groups)} gelabelte Videos...")

    # Abarbeiten und Löschen
    for vid_path, samples in tqdm(video_groups.items(), desc="Processing Videos"):
        
        if not os.path.exists(vid_path):
            print(f"Video nicht gefunden (wurde evtl. schon gelöscht?): {vid_path}")
            continue

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"Konnte Video nicht öffnen: {vid_path}")
            continue
            
        samples.sort(key=lambda x: x['frame_number'])

        for sample in samples:
            f_num = sample['frame_number']
            subset = sample['subset']
            
            opencv_frame_idx = f_num - 1 
            cap.set(cv2.CAP_PROP_POS_FRAMES, opencv_frame_idx)
            ret, frame = cap.read()
            
            if not ret: continue
                
            h_img, w_img, _ = frame.shape
            
            # Dateinamen
            base_name_raw = os.path.splitext(sample['base_filename'])[0]
            base_name = sanitize_filename(base_name_raw)
            out_name = f"{base_name}_frame{f_num}"
            
            img_out_path = os.path.join(OUTPUT_DIR, subset, 'images', f"{out_name}.jpg")
            txt_out_path = os.path.join(OUTPUT_DIR, subset, 'labels', f"{out_name}.txt")
            
            cv2.imwrite(img_out_path, frame)
            
            yolo_lines = []
            for obj_id, obj_data in sample['objects'].items():
                cls_name = obj_data['name']
                if cls_name in CLASSES:
                    cls_id = CLASSES.index(cls_name)
                    bbox = obj_data['bounding_box']
                    y_bbox = convert_bbox(w_img, h_img, bbox)
                    yolo_lines.append(f"{cls_id} {y_bbox[0]:.6f} {y_bbox[1]:.6f} {y_bbox[2]:.6f} {y_bbox[3]:.6f}")
            
            with open(txt_out_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
        
        cap.release()
        
        # --- LÖSCHEN NACH VERARBEITUNG ---
        try:
            os.remove(vid_path)
            print(f"Gelöscht nach Verarbeitung: {vid_path}")
        except OSError as e:
            print(f"FEHLER beim Löschen von {vid_path}: {e}")

    # YAML erstellen
    yaml_content = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'names': {i: name for i, name in enumerate(CLASSES)}
    }
    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print("\nFertig! Alle Videos wurden verarbeitet und gelöscht.")

if __name__ == "__main__":
    process_video_data()