import cv2
from ultralytics import YOLO

def detect_objects_in_video(video_path):
    # 1. Lade das YOLOv8 Modell
    # 'yolov8n.pt' ist das Nano-Modell (klein & schnell).
    # Alternativen: yolov8s.pt (small), yolov8m.pt (medium) für höhere Genauigkeit.
    # Das Modell wird beim ersten Start automatisch heruntergeladen.
    print("Lade Modell...")
    model = YOLO('yolov8n.pt')

    # 2. Öffne die Videoquelle
    # Nutze '0' für die Webcam oder den Dateipfad für ein Video (z.B. "mein_video.mp4")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Fehler: Konnte Video '{video_path}' nicht öffnen.")
        return

    print("Starte Analyse. Drücke 'q', um abzubrechen.")

    while True:
        # 3. Einzelnen Frame lesen
        success, frame = cap.read()

        if not success:
            print("Video zu Ende oder Fehler beim Lesen des Frames.")
            break

        # 4. YOLOv8 Inferenz auf dem Frame ausführen
        # conf=0.5 bedeutet, nur Objekte mit >50% Sicherheit anzeigen
        results = model(frame, conf=0.1, verbose=False)

        # 5. Ergebnisse visualisieren
        # YOLO hat eine eingebaute plot() Funktion, die Bounding Boxes und Labels zeichnet
        annotated_frame = results[0].plot()

        # 6. Ergebnis anzeigen
        cv2.imshow("YOLOv8 Objekt-Erkennung", annotated_frame)

        # Drücke 'q' auf der Tastatur, um die Schleife zu beenden
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Aufräumen
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # HIER den Pfad zu deinem Video eintragen oder 0 für Webcam
    VIDEO_SOURCE = "shortened_videos/20241031_Rahn_05_In (4).mp4" 
    # VIDEO_SOURCE = 0  # <--- Einkommentieren für Webcam
    
    detect_objects_in_video(VIDEO_SOURCE)