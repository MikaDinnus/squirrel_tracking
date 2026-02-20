from ultralytics import YOLO

def main():
    # 1. Modell laden
    model = YOLO('yolov11n.pt') 

    # 2. Training starten
    results = model.train(
        data='yolo_video_dataset/data.yaml', 
        epochs=50, 
        imgsz=640,
        batch=16,        # Wichtig für GTX 970
        device=0,       # GPU nutzen
        workers=2       # Datenlader-Prozesse
    )

if __name__ == '__main__':
    # Dieser Block wird NUR vom Hauptprozess ausgeführt,
    # nicht von den Workern!
    main()
