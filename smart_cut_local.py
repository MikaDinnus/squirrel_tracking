import cv2
import numpy as np
from collections import deque
import os
import sys
import imageio.v2 as imageio # v2 API ist einfacher für Streaming-Writes
from tqdm import tqdm

# ================= EINSTELLUNGEN =================
# Analyse-Einstellungen
RESIZE_RATE = 0.25          # Bild halbieren für Analyse (macht es schneller)
ANALYSIS_SKIP_RATE = 10     # Nur jedes 10. Frame prüfen

OUTPUT_RESIZE_RATE = 0.5    # Ausgabegröße (1.0 = Originalgröße)

# Empfindlichkeit (Image Subtraction)
PIXEL_DIFF_THRESH = 30     # Wie stark muss sich die Farbe ändern? (0-255)
MIN_AREA_PIXELS = 50       # Wie viele Pixel müssen sich ändern?
BLUR_SIZE = (21, 21)       # Weichzeichner gegen Rauschen

# Zeitsteuerung (Sekunden)
BUFFER_SECONDS = 3         # Pre-Roll: Zeit VOR dem Event
COOLDOWN_SECONDS = 3       # Post-Roll: Zeit NACH dem Event
# =================================================

def shorten(VIDEO_PATH, OUTPUT_PATH):
    # 1. Datei Check
    if not os.path.exists(VIDEO_PATH):
        print(f"FEHLER: Video nicht gefunden: {VIDEO_PATH}")
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Metadaten lesen
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0 # Fallback
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"--- Starte Lokale Analyse ---")
    print(f"Input: {VIDEO_PATH}")
    print(f"Original Auflösung: {width}x{height} @ {fps} FPS")
    print(f"Analyse Auflösung: {int(width*RESIZE_RATE)}x{int(height*RESIZE_RATE)} @ {fps/ANALYSIS_SKIP_RATE:.2f} FPS")
    print(f"Output Auflösung: {int(width*OUTPUT_RESIZE_RATE)}x{int(height*OUTPUT_RESIZE_RATE)} @ {fps} FPS")
    print(f"Gesamtanzahl Frames: {total_frames}")
    print(f"Encoder: H.264 (via imageio/ffmpeg) für kleine Dateigröße")
    print(f"output: {OUTPUT_PATH}\n")

    # 2. Writer Setup (Der Schlüssel für kleine Dateien!)
    # Wir nutzen 'libx264', das ist der Standard für MP4 Komprimierung
    try:
        writer = imageio.get_writer(
            OUTPUT_PATH, 
            fps=fps, 
            codec='libx264',  # <--- DAS macht die Datei klein
            quality=6,        # 5-7 ist optimal. (entspricht CRF ~23)
            pixelformat='yuv420p', # Maximale Kompatibilität (Windows/Mac/Android)
            macro_block_size=None,  # Verhindert Fehler bei krummen Auflösungen
        )
    except Exception as e:
        print(f"\nFEHLER beim Starten des Writers: {e}")
        print("Hast du 'pip install imageio[ffmpeg]' ausgeführt?")
        sys.exit(1)

    # 3. Puffer Setup
    buffer_frames_count = int(BUFFER_SECONDS * fps)
    cooldown_frames_count = int(COOLDOWN_SECONDS * fps)
    frame_buffer = deque(maxlen=buffer_frames_count)

    # Variablen
    prev_gray = None
    recording = False
    cooldown_counter = 0
    saved_frames_count = 0
    motion_detected = False

    # 4. Hauptschleife
    pbar = tqdm(total=total_frames, unit="fr", desc="Verarbeite")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- A. Analyse (Nur alle X Frames um CPU zu sparen) ---
        should_analyze = (frame_idx % ANALYSIS_SKIP_RATE == 0)

        if should_analyze:
            # Bild verkleinern & Grau
            frame_small = cv2.resize(frame, (0, 0), fx=RESIZE_RATE, fy=RESIZE_RATE)
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, BLUR_SIZE, 0)

            motion_detected = False 
            if prev_gray is not None:
                # Mathe: Differenz -> Schwellenwert -> Löcher stopfen
                diff = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(diff, PIXEL_DIFF_THRESH, 255, cv2.THRESH_BINARY)
                thresh = cv2.dilate(thresh, None, iterations=2)
                
                # Wenn genug weiße Pixel da sind -> Bewegung!
                if np.count_nonzero(thresh) > MIN_AREA_PIXELS:
                    motion_detected = True
            
            prev_gray = gray

        # --- B. Aufnahme-Logik ---
        
        # Entscheidung: Bewegung erkannt?
        if motion_detected:
            if not recording:
                # Start-Event! Puffer (Vergangenheit) schreiben
                while frame_buffer:
                    buf_frame = frame_buffer.popleft()
                    buf_frame = cv2.resize(buf_frame, (0, 0), fx=OUTPUT_RESIZE_RATE, fy=OUTPUT_RESIZE_RATE)
                    # WICHTIG: OpenCV ist BGR, imageio braucht RGB
                    rgb_frame = cv2.cvtColor(buf_frame, cv2.COLOR_BGR2RGB)
                    writer.append_data(rgb_frame)
                    saved_frames_count += 1
                recording = True
            
            # Timer resetten (Verlängerung der Aufnahme)
            cooldown_counter = cooldown_frames_count

        # Entscheidung: Schreiben oder Puffern?
        if recording:
            # Wir sind im Aufnahmemodus
            frame = cv2.resize(frame, (0, 0), fx=OUTPUT_RESIZE_RATE, fy=OUTPUT_RESIZE_RATE)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_frame)
            saved_frames_count += 1
            
            # Cooldown runterzählen
            cooldown_counter -= 1
            if cooldown_counter <= 0:
                recording = False
                frame_buffer.clear() # Puffer leeren
        else:
            # Wir sind im Überwachungsmodus -> Nur in den RAM puffern
            frame_buffer.append(frame)

        frame_idx += 1
        pbar.update(1)

    # 5. Aufräumen
    pbar.close()
    cap.release()
    writer.close() # Schließt die Datei sauber ab

    # Statistik
    if saved_frames_count > 0:
        duration = saved_frames_count / fps
        orig_size = os.path.getsize(VIDEO_PATH) / (1024*1024)
        new_size = os.path.getsize(OUTPUT_PATH) / (1024*1024)
        
        print(f"\n--- FERTIG ---")
        print(f"Gespeichert als: {OUTPUT_PATH}")
        print(f"Dauer: {duration:.1f} Sekunden")
        print(f"Dateigröße: {new_size:.1f} MB (Original war {orig_size:.1f} MB)")
    else:
        print("\nWarnung: Keine Bewegung erkannt, kein Video gespeichert.")

if __name__ == "__main__":
    for filename in os.listdir('original_videos'):
        shorten(os.path.join('original_videos', filename), os.path.join('shortened_videos', filename.replace(".MOV", ".mp4")))