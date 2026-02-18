import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import os
import numpy as np

# Deine definierten Klassen
TARGET_CLASSES = ["squirrel", "cup_empty", "cup_full", "disco_ball", "nut"]

class YoloDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Feingranulare Einstellungen")
        self.root.geometry("650x850") # Etwas breiter für die 4. Spalte

        # --- Variablen ---
        self.model_path = tk.StringVar(value="")
        self.video_path = tk.StringVar(value="")
        self.conf_threshold = tk.DoubleVar(value=0.5)
        self.trace_length = tk.IntVar(value=60)
        self.move_threshold = tk.IntVar(value=15)
        
        # Dictionary für Settings: { "squirrel": { "box": Var, "status": Var, "heatmap": Var, "trace": Var } }
        self.class_settings = {}

        self.create_widgets()

    def create_widgets(self):
        # --- 1. Setup ---
        tk.Label(self.root, text="1. Setup", font=("Arial", 11, "bold")).pack(pady=(10, 5))
        frame_files = tk.Frame(self.root)
        frame_files.pack(fill="x", padx=20)
        tk.Button(frame_files, text="Modell (.pt)", command=self.select_model).grid(row=0, column=0, padx=5)
        tk.Entry(frame_files, textvariable=self.model_path, width=40).grid(row=0, column=1, padx=5)
        tk.Button(frame_files, text="Video", command=self.select_video).grid(row=1, column=0, padx=5, pady=5)
        tk.Entry(frame_files, textvariable=self.video_path, width=40).grid(row=1, column=1, padx=5, pady=5)

        # --- 2. Parameter ---
        frame_global = tk.LabelFrame(self.root, text="Parameter", padx=10, pady=5)
        frame_global.pack(fill="x", padx=20, pady=10)
        
        tk.Label(frame_global, text="Confidence:").grid(row=0, column=0, sticky="w")
        tk.Scale(frame_global, from_=0.1, to=1.0, resolution=0.05, orient="horizontal", variable=self.conf_threshold, length=150).grid(row=0, column=1)

        tk.Label(frame_global, text="Bewegungs-Schwelle (px):").grid(row=1, column=0, sticky="w")
        tk.Scale(frame_global, from_=1, to=100, orient="horizontal", variable=self.move_threshold, length=150).grid(row=1, column=1)

        tk.Label(frame_global, text="Trace Länge (Frames):").grid(row=2, column=0, sticky="w")
        tk.Scale(frame_global, from_=10, to=200, orient="horizontal", variable=self.trace_length, length=150).grid(row=2, column=1)

        # --- 3. MATRIX ---
        tk.Label(self.root, text="3. Anzeige-Konfiguration", font=("Arial", 11, "bold")).pack(pady=(10, 5))
        
        frame_matrix = tk.Frame(self.root, bd=2, relief="groove")
        frame_matrix.pack(padx=20, pady=5)

        # Tabellen-Header
        tk.Label(frame_matrix, text="Klasse", font=("Arial", 9, "bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        tk.Label(frame_matrix, text="Box (Rahmen)", font=("Arial", 9, "bold"), fg="blue").grid(row=0, column=1, padx=5)
        tk.Label(frame_matrix, text="Status (Move)", font=("Arial", 9, "bold"), fg="green").grid(row=0, column=2, padx=5)
        tk.Label(frame_matrix, text="Heatmap", font=("Arial", 9, "bold"), fg="red").grid(row=0, column=3, padx=5)
        tk.Label(frame_matrix, text="Trace", font=("Arial", 9, "bold"), fg="darkcyan").grid(row=0, column=4, padx=5)

        # Zeilen generieren
        for idx, class_name in enumerate(TARGET_CLASSES):
            row = idx + 1
            
            tk.Label(frame_matrix, text=class_name.capitalize()).grid(row=row, column=0, padx=10, sticky="w")
            
            # Variablen
            var_box = tk.BooleanVar(value=True)     # Rahmen
            var_status = tk.BooleanVar(value=True)  # "IDLE/MOVE" Text
            var_heat = tk.BooleanVar(value=True)    # Heatmap
            var_trace = tk.BooleanVar(value=False)  # Trace

            self.class_settings[class_name] = {
                "box": var_box,
                "status": var_status,
                "heatmap": var_heat,
                "trace": var_trace
            }

            # Checkboxen
            tk.Checkbutton(frame_matrix, variable=var_box).grid(row=row, column=1)
            tk.Checkbutton(frame_matrix, variable=var_status).grid(row=row, column=2)
            tk.Checkbutton(frame_matrix, variable=var_heat).grid(row=row, column=3)
            tk.Checkbutton(frame_matrix, variable=var_trace).grid(row=row, column=4)

        # --- Start ---
        self.btn_start = tk.Button(self.root, text="STARTEN", command=self.start_analysis, 
                                   bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), height=2)
        self.btn_start.pack(fill="x", padx=50, pady=20)
        
        self.lbl_status = tk.Label(self.root, text="Bereit", fg="grey")
        self.lbl_status.pack(pady=5)

    def select_model(self):
        f = filedialog.askopenfilename(filetypes=[("YOLO Weights", "*.pt")])
        if f: self.model_path.set(f)

    def select_video(self):
        f = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mov")])
        if f: self.video_path.set(f)

    def start_analysis(self):
        if not self.model_path.get() or not self.video_path.get():
            messagebox.showerror("Fehler", "Dateien fehlen!")
            return
        
        self.lbl_status.config(text="Analysiere...", fg="blue")
        self.root.update()
        try:
            self.run_logic()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(e)
        finally:
            self.lbl_status.config(text="Fertig", fg="grey")

    def run_logic(self):
        model = YOLO(self.model_path.get())
        cap = cv2.VideoCapture(self.video_path.get())
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        heatmap_acc = np.zeros((h, w), dtype=np.float32)
        track_history = {} 
        
        max_trace = self.trace_length.get()
        move_thresh = self.move_threshold.get()
        LOOKBACK_FRAMES = 10 

        window_name = "YOLO Fine-Grained Analysis"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, w, h)

        while True:
            success, frame = cap.read()
            if not success: break

            results = model.track(frame, conf=self.conf_threshold.get(), persist=True, verbose=False, tracker="bytetrack.yaml")
            annotated_frame = frame.copy() 

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                cls_ids = results[0].boxes.cls.int().cpu().tolist()
                names_map = model.names

                for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                    
                    class_name_raw = names_map[cls_id]
                    class_name = class_name_raw.strip().lower()

                    if class_name in self.class_settings:
                        settings = self.class_settings[class_name]
                        
                        # Variablen aus GUI abfragen
                        show_box = settings["box"].get()
                        show_status = settings["status"].get()
                        show_heat = settings["heatmap"].get()
                        show_trace = settings["trace"].get()

                        # Wenn NICHTS ausgewählt ist für diese Klasse, überspringen wir sie komplett
                        if not (show_box or show_status or show_heat or show_trace):
                            continue
                            
                        x, y, w_box, h_box = box
                        center = (float(x), float(y))
                        
                        # --- History Update (brauchen wir für Status & Trace) ---
                        track = track_history.get(track_id, [])
                        track.append(center)
                        if len(track) > max_trace: track.pop(0)
                        track_history[track_id] = track

                        # --- Heatmap Logic ---
                        if show_heat:
                            radius = int((w_box + h_box) / 8)
                            temp = np.zeros_like(heatmap_acc)
                            cv2.circle(temp, (int(x), int(y)), max(5, radius), (1), -1)
                            heatmap_acc += temp * 2.0

                        # --- Trace Logic ---
                        if show_trace and len(track) > 1:
                            pts = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(annotated_frame, [pts], False, (255, 255, 0), 2)

                        # --- Status & Box Logic ---
                        
                        # Standardfarbe (Orange), falls Status-Erkennung aus ist
                        color = (0, 165, 255) 
                        status_text = ""

                        # Wenn Status aktiviert ist, berechnen wir Bewegung und ändern Farbe/Text
                        if show_status:
                            dist = 0.0
                            if len(track) > LOOKBACK_FRAMES:
                                dist = np.linalg.norm(np.array(center) - np.array(track[-LOOKBACK_FRAMES]))
                            elif len(track) > 1:
                                dist = np.linalg.norm(np.array(center) - np.array(track[0]))
                            
                            is_moving = dist > move_thresh
                            
                            # Grün für Bewegung, Rot für Stillstand
                            color = (0, 255, 0) if is_moving else (0, 0, 255)
                            status_text = f" | MOVE" if is_moving else f" | IDLE"

                        # Zeichnen
                        x1, y1 = int(x - w_box/2), int(y - h_box/2)
                        x2, y2 = int(x + w_box/2), int(y + h_box/2)

                        # 1. Rahmen zeichnen (nur wenn Box an)
                        if show_box:
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                        # 2. Text zeichnen (Wenn Box ODER Status an ist)
                        # Wenn nur Status an ist, sieht man schwebenden Text ohne Box.
                        if show_box or show_status:
                            label = f"{class_name.upper()} {track_id}{status_text}"
                            # Text etwas oberhalb der Box platzieren
                            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Heatmap Overlay
            if np.max(heatmap_acc) > 0:
                hm_norm = cv2.normalize(heatmap_acc, None, 0, 255, cv2.NORM_MINMAX)
                hm_color = cv2.applyColorMap(hm_norm.astype(np.uint8), cv2.COLORMAP_JET)
                mask = (hm_norm / 255.0)[:, :, None]
                annotated_frame = (annotated_frame * (1.0 - mask * 0.6) + hm_color * mask * 0.6).astype(np.uint8)

            cv2.imshow(window_name, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloDetectorApp(root)
    root.mainloop()