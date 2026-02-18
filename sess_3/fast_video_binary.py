import cv2
import numpy as np
import os
import subprocess
import sys
import imageio_ffmpeg


class FastVideoSearcher:
    def __init__(self, video_path, output_dir, headless=False, on_progress=None, on_frame=None):
        self.video_path = video_path
        self.output_dir = output_dir
        self.headless = headless
        self.on_progress = on_progress
        self.on_frame = on_frame  # callback(numpy_bgr_image) for GUI visualization
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.total_frames / self.fps

        self.search_granularity = 30.0  # Grobe Binärsuche bis 30s
        self.fine_scan_step = 2.0       # Feinanalyse prüft nur alle 2 Sekunden

        self.pixel_threshold = 30       # Pixel-Unterschied (0-255)
        self.change_threshold = 2.0     # % des Bildes geändert
        self.samples_per_check = 5      # Stichproben für Binärsuche
        self.min_actual_action = 2.0    # Min. Event-Länge

        self.vis_width = 1000
        self.vis_height = 200
        self.vis_img = np.zeros((self.vis_height, self.vis_width, 3), dtype=np.uint8)

        if not self.headless:
            self.window_name = "Smart Event Searcher"

        os.makedirs(output_dir, exist_ok=True)

    def _log(self, msg):
        print(msg)
        if self.on_progress:
            self.on_progress(msg)

    def get_frame_at_sec(self, seconds):
        frame_idx = int(seconds * self.fps)
        if frame_idx >= self.total_frames: frame_idx = self.total_frames - 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret: return None, None
        small = cv2.resize(frame, (320, 180)) 
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return frame, gray

    def get_change_score(self, frame_gray, ref_gray):
        if frame_gray is None or ref_gray is None: return 0.0
        diff = cv2.absdiff(ref_gray, frame_gray)
        _, thresh = cv2.threshold(diff, self.pixel_threshold, 255, cv2.THRESH_BINARY)
        non_zero = cv2.countNonZero(thresh)
        return (non_zero / frame_gray.size) * 100.0

    def check_interval_dirty(self, start_sec, end_sec, ref_gray):
        if end_sec <= start_sec: return False
        step = (end_sec - start_sec) / (self.samples_per_check + 1)
        check_points = [start_sec + step * i for i in range(1, self.samples_per_check + 1)]
        is_dirty = False
        for t in check_points:
            _, sample_gray = self.get_frame_at_sec(t)
            pct = self.get_change_score(sample_gray, ref_gray)
            self.draw_status(start_sec, end_sec, t, pct, is_dirty=pct > self.change_threshold)
            if pct > self.change_threshold:
                is_dirty = True
                break 
        return is_dirty

    def recursive_search(self, start_sec, end_sec, ref_gray):
        duration = end_sec - start_sec
        if duration <= self.search_granularity:
            if self.check_interval_dirty(start_sec, end_sec, ref_gray):
                return (start_sec, end_sec)
            else:
                return None
        mid_sec = start_sec + (duration / 2)
        if self.check_interval_dirty(start_sec, mid_sec, ref_gray):
            return self.recursive_search(start_sec, mid_sec, ref_gray)
        else:
            if self.check_interval_dirty(mid_sec, end_sec, ref_gray):
                return self.recursive_search(mid_sec, end_sec, ref_gray)
        return None

    def refine_and_export(self, coarse_start, coarse_end, ref_gray, event_idx):
        self._log(f" -> Feinanalyse (Schrittweite {self.fine_scan_step}s): {coarse_start:.1f}s - {coarse_end:.1f}s")
        
        scan_step = self.fine_scan_step 
        
        current = coarse_start
        real_start = None
        real_end = None
        
        while current < coarse_end + 5.0: 
            _, fr_gray = self.get_frame_at_sec(current)
            if fr_gray is None: break 
            
            score = self.get_change_score(fr_gray, ref_gray)
            self.draw_status(coarse_start, coarse_end, current, score, is_dirty=(score > self.change_threshold), color_override=(255, 0, 255))
            
            if score > self.change_threshold:
                if real_start is None: real_start = current
                real_end = current 
            else:
                if real_start is not None and (current - real_end) > 2.0: break
            current += scan_step

        if real_start is None:
            self._log(" -> Fehlalarm. Kein Export.")
            return None

        duration = real_end - real_start
        if duration < self.min_actual_action:
            self._log(f" -> Zu kurz ({duration:.1f}s). Ignoriere.")
            return None

        # Puffer für FFmpeg (etwas großzügiger, da wir nur alle 2s scannen)
        final_start = max(0, real_start - 4.0)
        final_end = min(self.duration_sec, real_end + 4.0)
        clip_duration = final_end - final_start
        
        self._log(f" -> ECHTES EVENT: {final_start:.1f}s - {final_end:.1f}s (FFmpeg Export...)")
        
        # 2. EXPORT MIT FFMPEG (TURBO MODE)
        out_path = os.path.join(self.output_dir, f"event_{event_idx:02d}_{int(final_start)}s.mp4")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        cmd = [
            ffmpeg_exe, "-y",
            "-ss", str(final_start),
            "-i", self.video_path,
            "-t", str(clip_duration),
            "-c", "copy",
            "-loglevel", "error",
            out_path
        ]
        
        try:
            subprocess.run(cmd, check=True)
            self._log(f"    [OK] Gespeichert: {out_path}")
        except subprocess.CalledProcessError as e:
            self._log(f"    [ERROR] FFmpeg Fehler: {e}")

        return final_end 

    def run(self):
        self._log(f"Starte Smart-Search auf: {self.video_path}")
        current_time = 0.0
        event_counter = 0
        try:
            while current_time < (self.duration_sec - 10):
                self._log(f"\n--- Referenzpunkt: {current_time/60:.2f} min ---")
                _, ref_gray = self.get_frame_at_sec(current_time)
                if ref_gray is None: break

                found_interval = self.recursive_search(current_time, self.duration_sec, ref_gray)
                if found_interval:
                    start, end = found_interval
                    self.draw_status(start, end, start, 100, True, (0, 165, 255))
                    real_end = self.refine_and_export(start, end, ref_gray, event_counter + 1)

                    if real_end:
                        current_time = real_end
                        event_counter += 1
                    else:
                        current_time = end
                else:
                    self._log("Keine weiteren Events.")
                    break
        except KeyboardInterrupt:
            self._log("Abbruch.")
        finally:
            self.cap.release()
            if not self.headless:
                cv2.destroyAllWindows()
                for _ in range(5): cv2.waitKey(1)
            self._log("Fertig.")

    def draw_status(self, start, end, curr, score, is_dirty, color_override=None):
        scale = self.vis_width / self.duration_sec
        x_start, x_end, x_curr = int(start * scale), int(end * scale), int(curr * scale)
        cv2.rectangle(self.vis_img, (0,0), (self.vis_width, self.vis_height), (0,0,0), -1)
        cv2.line(self.vis_img, (0, 100), (self.vis_width, 100), (50, 50, 50), 2)
        col = color_override if color_override else ((0, 0, 255) if is_dirty else (255, 255, 0))
        cv2.rectangle(self.vis_img, (x_start, 80), (x_end, 120), col, 2)
        cv2.circle(self.vis_img, (x_curr, 100), 5, (0, 255, 0), -1)
        txt = f"Time: {curr:.0f}s | Diff: {score:.1f}%"
        cv2.putText(self.vis_img, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)

        if self.on_frame:
            self.on_frame(self.vis_img.copy())
        if not self.headless:
            cv2.imshow(self.window_name, self.vis_img)
            cv2.waitKey(1)

VIDEO = "C:/Users/itsmi/OneDrive/Desktop/DATTSOSIB/videos/raw/Rahn_05_7.mov" 
OUT = "sess_3/output"

if __name__ == "__main__":
    if os.path.exists(VIDEO):
        searcher = FastVideoSearcher(VIDEO, OUT)
        searcher.run()
    else:
        print("Video not found.")