import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import os

class YoloDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Objekt-Erkennung")
        self.root.geometry("500x450")

        # Standardwerte
        self.model_path = tk.StringVar(value="")
        self.video_path = tk.StringVar(value="")
        self.threshold = tk.DoubleVar(value=0.5)

        self.create_widgets()

    def create_widgets(self):
        # --- Modell Auswahl ---
        tk.Label(self.root, text="1. YOLO Modell (.pt Datei)", font=("Arial", 10, "bold")).pack(pady=(10, 5))
        
        frame_model = tk.Frame(self.root)
        frame_model.pack(fill="x", padx=20)
        
        self.entry_model = tk.Entry(frame_model, textvariable=self.model_path, state="readonly")
        self.entry_model.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        btn_model = tk.Button(frame_model, text="Durchsuchen", command=self.select_model)
        btn_model.pack(side="right")

        # --- Video Auswahl ---
        tk.Label(self.root, text="2. Videoquelle", font=("Arial", 10, "bold")).pack(pady=(20, 5))
        
        frame_video = tk.Frame(self.root)
        frame_video.pack(fill="x", padx=20)
        
        self.entry_video = tk.Entry(frame_video, textvariable=self.video_path, state="readonly")
        self.entry_video.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        btn_video = tk.Button(frame_video, text="Durchsuchen", command=self.select_video)
        btn_video.pack(side="right")

        # --- Threshold Slider ---
        tk.Label(self.root, text="3. Confidence Threshold (Vertrauen)", font=("Arial", 10, "bold")).pack(pady=(20, 5))
        
        slider = tk.Scale(self.root, from_=0.1, to=1.0, resolution=0.05, 
                          orient="horizontal", variable=self.threshold, length=400)
        slider.pack(pady=5)

        # --- Trennlinie ---
        tk.Frame(self.root, height=2, bd=1, relief="sunken").pack(fill="x", padx=20, pady=20)

        # --- Start Button ---
        self.btn_start = tk.Button(self.root, text="ANALYSE STARTEN", 
                                   command=self.start_analysis, 
                                   bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), height=2)
        self.btn_start.pack(fill="x", padx=50, pady=10)

        # --- Status Label ---
        self.lbl_status = tk.Label(self.root, text="Bereit", fg="grey")
        self.lbl_status.pack(pady=5)

    def select_model(self):
        filename = filedialog.askopenfilename(
            title="Wähle YOLO Modell",
            filetypes=[("YOLO Weights", "*.pt"), ("All Files", "*.*")]
        )
        if filename:
            self.model_path.set(filename)

    def select_video(self):
        filename = filedialog.askopenfilename(
            title="Wähle Videodatei",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if filename:
            self.video_path.set(filename)

    def start_analysis(self):
        model_file = self.model_path.get()
        video_file = self.video_path.get()
        conf_thresh = self.threshold.get()

        if not model_file or not video_file:
            messagebox.showerror("Fehler", "Bitte Modell und Video auswählen!")
            return
        
        if not os.path.exists(model_file):
             messagebox.showerror("Fehler", "Modelldatei nicht gefunden!")
             return

        self.lbl_status.config(text="Starte Video...", fg="blue")
        self.root.update()

        try:
            self.run_yolo(model_file, video_file, conf_thresh)
        except Exception as e:
            messagebox.showerror("Fehler", f"Ein Fehler ist aufgetreten:\n{e}")
        finally:
            self.lbl_status.config(text="Analyse beendet.", fg="grey")

    def run_yolo(self, model_file, video_file, conf_thresh):
        print(f"Lade Modell: {model_file}")
        model = YOLO(model_file)
        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            raise IOError("Konnte Videoquelle nicht öffnen.")

        # --- UPDATE START: Video-Eigenschaften lesen ---
        # Wir holen uns die originalen Dimensionen des Videos
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video Dimensionen: {video_width}x{video_height}")

        window_name = "YOLO Analyse"
        
        # 1. Fenster erstellen, das skalierbar ist (WINDOW_NORMAL)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # 2. Fenstergröße exakt auf die Videogröße setzen
        # Damit startet es im korrekten Seitenverhältnis (z.B. 16:9)
        # Falls das Video riesig ist (z.B. 4K), könnte man hier auch durch 2 teilen.
        cv2.resizeWindow(window_name, video_width, video_height)
        # --- UPDATE ENDE ---

        while True:
            success, frame = cap.read()
            if not success:
                break

            # YOLO Inferenz
            results = model(frame, conf=conf_thresh, verbose=False)
            annotated_frame = results[0].plot()

            cv2.imshow(window_name, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloDetectorApp(root)
    root.mainloop()