import cv2
import numpy as np
from collections import deque
import os
import sys
from tqdm import tqdm  # Falls nicht installiert: pip install tqdm

# ================= KONFIGURATION =================
VIDEO_PATH = '20241023_TrepS_01_in (2).MOV'
OUTPUT_PATH = 'Smart_Cut_Final.mp4'

# --- Performance & Analyse ---
RESIZE_RATE = 0.5          # 0.5 = Bild für Analyse halbieren (Output bleibt Original!)
ANALYSIS_SKIP_RATE = 3     # Nur jedes 3. Frame prüfen (höher = schneller, aber ungenauer)

# --- Bewegungserkennung (Image Subtraction) ---
PIXEL_DIFF_THRESH = 30     # Wie stark muss sich die Farbe ändern? (0-255)
MIN_AREA_PIXELS = 50       # Wie viele Pixel müssen sich ändern?
BLUR_SIZE = (21, 21)       # Gegen Rauschen (ungerade Zahlen!)

# --- Zeitsteuerung ---
BUFFER_SECONDS = 5         # Sekunden VOR der Bewegung speichern (Pre-Roll)
COOLDOWN_SECONDS = 5       # Sekunden NACH der Bewegung weiter aufnehmen (Post-Roll)

# --- Debugging ---
SHOW_PREVIEW = False       # Setze auf True, um ein Fenster mit der Analyse zu sehen (langsamer!)

# =================================================

def main():
    # 1. Video Check
    if not os.path.exists(VIDEO_PATH):
        print(f"FEHLER: Datei nicht gefunden: {VIDEO_PATH}")
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("FEHLER: Konnte Video nicht öffnen.")
        sys.exit(1)

    # 2. Video Infos lesen
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"--- Video Analyse Start ---")
    print(f"Input: {VIDEO_PATH}")
    print(f"Auflösung: {width}x{height} | FPS: {fps:.2f}")
    print(f"Frames gesamt: {total_frames}")
    print(f"Analyse-Rate: Prüfe jedes {ANALYSIS_SKIP_RATE}. Frame")
    
    # 3. Output Setup
    # mp4v ist kompatibel, für bessere Kompression könnte man 'avc1' (H.264) probieren, falls installiert
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    # 4. Buffer & Variablen Setup
    buffer_frames_count = int(BUFFER_SECONDS * fps)
    cooldown_frames_count = int(COOLDOWN_SECONDS * fps)
    
    frame_buffer = deque(maxlen=buffer_frames_count)
    
    prev_gray = None
    recording = False
    cooldown_counter = 0
    saved_frames_count = 0
    motion_detected = False

    # 5. Hauptschleife
    # tqdm sorgt für den Ladebalken im Terminal
    pbar = tqdm(total=total_frames, unit="fr", desc="Verarbeite")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- A. Analyse (Nur alle X Frames) ---
        should_analyze = (frame_idx % ANALYSIS_SKIP_RATE == 0)

        if should_analyze:
            # Bild verkleinern & Grau
            frame_small = cv2.resize(frame, (0, 0), fx=RESIZE_RATE, fy=RESIZE_RATE)
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, BLUR_SIZE, 0)

            motion_detected = False # Reset für diesen Check

            if prev_gray is not None:
                # Differenz berechnen
                frame_diff = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(frame_diff, PIXEL_DIFF_THRESH, 255, cv2.THRESH_BINARY)
                thresh = cv2.dilate(thresh, None, iterations=2)
                
                # Zählen
                if np.count_nonzero(thresh) > MIN_AREA_PIXELS:
                    motion_detected = True
                
                # Preview Fenster (Optional)
                if SHOW_PREVIEW:
                    cv2.imshow("Bewegungsmaske", thresh)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            prev_gray = gray

        # --- B. Aufnahme-Logik (State Machine) ---
        
        # Wenn Bewegung erkannt wurde (oder der Status vom letzten Check noch gilt)
        if motion_detected:
            if not recording:
                # Start Event: Puffer schreiben
                # (tqdm.write nutzt man statt print, um den Balken nicht zu zerstören)
                # tqdm.write(f"Bewegung bei Frame {frame_idx}! Starte Aufnahme...")
                while frame_buffer:
                    out.write(frame_buffer.popleft())
                    saved_frames_count += 1
                recording = True
            
            # Timer zurücksetzen
            cooldown_counter = cooldown_frames_count

        if recording:
            out.write(frame)
            saved_frames_count += 1
            
            # Cooldown runterzählen
            cooldown_counter -= 1
            
            if cooldown_counter <= 0:
                # tqdm.write(f"Stop Aufnahme bei Frame {frame_idx}.")
                recording = False
                frame_buffer.clear()
        else:
            # Nur puffern
            frame_buffer.append(frame)

        frame_idx += 1
        pbar.update(1)

    # 6. Aufräumen
    pbar.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if saved_frames_count > 0:
        duration_saved = saved_frames_count / fps
        print(f"\n--- FERTIG ---")
        print(f"Gespeichert: {OUTPUT_PATH}")
        print(f"Dauer neu: {duration_saved:.2f} Sekunden ({saved_frames_count} Frames)")
        print(f"Reduktion: {100 - (saved_frames_count/total_frames*100):.1f}% kleiner als Original.")
    else:
        print("\nWARNUNG: Keine Bewegung erkannt. Das Video ist leer.")

if __name__ == "__main__":
    main()