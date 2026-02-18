import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import glob

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QTextEdit,
    QProgressBar, QSizePolicy, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap

from fast_video_binary import FastVideoSearcher

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".wmv")


class ProcessingWorker(QThread):
    """Runs FastVideoSearcher on all videos in a folder (headless, sends frames via signal)."""
    log_message = pyqtSignal(str)
    video_progress = pyqtSignal(int, int)
    vis_frame = pyqtSignal(np.ndarray)  # visualization image from draw_status
    finished_all = pyqtSignal(str)

    def __init__(self, video_folder, output_dir, yolo_model_path=None):
        super().__init__()
        self.video_folder = video_folder
        self.output_dir = output_dir
        self.yolo_model_path = yolo_model_path

    def run(self):
        video_files = []
        for f in sorted(os.listdir(self.video_folder)):
            if f.lower().endswith(VIDEO_EXTENSIONS):
                video_files.append(os.path.join(self.video_folder, f))

        if not video_files:
            self.log_message.emit("No video files found in the selected folder.")
            return

        # Load YOLO once for all videos
        yolo_model = None
        if self.yolo_model_path:
            from ultralytics import YOLO
            self.log_message.emit(f"Loading YOLO model: {self.yolo_model_path}")
            yolo_model = YOLO(self.yolo_model_path)

        self.log_message.emit(f"Found {len(video_files)} video(s). Starting processing...\n")
        os.makedirs(self.output_dir, exist_ok=True)



        for i, video_path in enumerate(video_files):
            self.video_progress.emit(i + 1, len(video_files))
            self.log_message.emit(f"=== [{i+1}/{len(video_files)}] {os.path.basename(video_path)} ===")

            try:
                searcher = FastVideoSearcher(
                    video_path,
                    self.output_dir,
                    headless=True,
                    yolo_model=yolo_model,
                    on_progress=lambda msg: self.log_message.emit(msg),
                    on_frame=lambda img: self.vis_frame.emit(img),
                )
                searcher.run()
            except Exception as e:
                self.log_message.emit(f"ERROR processing {os.path.basename(video_path)}: {e}")

        self.log_message.emit(f"\nAll videos processed. Clips saved to: {self.output_dir}")
        self.finished_all.emit(self.output_dir)


class ProcessingTab(QWidget):
    """Tab 1: Select video folder + output folder, process videos, show progress + visualization."""
    processing_done = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.selected_folder = None
        self.output_folder = None
        self.yolo_model_path = None
        self.worker = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Video folder selection
        vid_row = QHBoxLayout()
        self.folder_label = QLabel("No video folder selected")
        self.folder_label.setStyleSheet("color: #888; font-size: 14px;")
        vid_row.addWidget(self.folder_label, 1)
        self.browse_btn = QPushButton("Select Video Folder")
        self.browse_btn.setMinimumHeight(40)
        self.browse_btn.setStyleSheet("font-size: 14px; padding: 8px 16px;")
        self.browse_btn.clicked.connect(self._browse_video_folder)
        vid_row.addWidget(self.browse_btn)
        layout.addLayout(vid_row)

        # Output folder selection
        out_row = QHBoxLayout()
        self.output_label = QLabel("No output folder selected")
        self.output_label.setStyleSheet("color: #888; font-size: 14px;")
        out_row.addWidget(self.output_label, 1)
        self.output_btn = QPushButton("Select Output Folder")
        self.output_btn.setMinimumHeight(40)
        self.output_btn.setStyleSheet("font-size: 14px; padding: 8px 16px;")
        self.output_btn.clicked.connect(self._browse_output_folder)
        out_row.addWidget(self.output_btn)
        layout.addLayout(out_row)

        # YOLO model selection
        yolo_row = QHBoxLayout()
        self.yolo_label = QLabel("No YOLO model selected")
        self.yolo_label.setStyleSheet("color: #888; font-size: 14px;")
        yolo_row.addWidget(self.yolo_label, 1)
        self.yolo_btn = QPushButton("Select YOLO Model (.pt)")
        self.yolo_btn.setMinimumHeight(40)
        self.yolo_btn.setStyleSheet("font-size: 14px; padding: 8px 16px;")
        self.yolo_btn.clicked.connect(self._browse_yolo_model)
        yolo_row.addWidget(self.yolo_btn)
        layout.addLayout(yolo_row)

        # Start button
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 12px;"
            "background-color: #4CAF50; color: white; border-radius: 6px;"
        )
        self.start_btn.clicked.connect(self._start_processing)
        layout.addWidget(self.start_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Visualization display (replaces the old cv2.imshow window)
        self.vis_label = QLabel()
        self.vis_label.setAlignment(Qt.AlignCenter)
        self.vis_label.setMinimumHeight(120)
        self.vis_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.vis_label)

        # Log area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        layout.addWidget(self.log_area, 1)

    def _browse_yolo_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO model", "", "YOLO Weights (*.pt);;All Files (*.*)"
        )
        if path:
            self.yolo_model_path = path
            self.yolo_label.setText(path)
            self.yolo_label.setStyleSheet("color: #222; font-size: 14px;")
            self._check_ready()

    def _browse_video_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with videos")
        if folder:
            self.selected_folder = folder
            self.folder_label.setText(folder)
            self.folder_label.setStyleSheet("color: #222; font-size: 14px;")
            self._check_ready()

    def _browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder for clips")
        if folder:
            self.output_folder = folder
            self.output_label.setText(folder)
            self.output_label.setStyleSheet("color: #222; font-size: 14px;")
            self._check_ready()

    def _check_ready(self):
        self.start_btn.setEnabled(
            self.selected_folder is not None and self.output_folder is not None and self.yolo_model_path is not None
        )

    def _start_processing(self):
        if not self.selected_folder or not self.output_folder:
            return

        self.start_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.output_btn.setEnabled(False)
        self.log_area.clear()
        self.progress_bar.setValue(0)

        self.worker = ProcessingWorker(self.selected_folder, self.output_folder, yolo_model_path=self.yolo_model_path)
        self.worker.log_message.connect(self._append_log)
        self.worker.video_progress.connect(self._update_progress)
        self.worker.vis_frame.connect(self._update_vis)
        self.worker.finished_all.connect(self._on_finished)
        self.worker.start()

    def _append_log(self, msg):
        self.log_area.append(msg)
        self.log_area.verticalScrollBar().setValue(
            self.log_area.verticalScrollBar().maximum()
        )

    def _update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"Video {current} / {total}")

    def _update_vis(self, bgr_img):
        """Display the visualization image from FastVideoSearcher in the GUI."""
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.vis_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.vis_label.setPixmap(scaled)

    def _on_finished(self, clips_dir):
        self.start_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.output_btn.setEnabled(True)
        self.processing_done.emit(clips_dir)


class VerificationTab(QWidget):
    """Tab 2: Review clips one-by-one using OpenCV playback in a QLabel."""

    def __init__(self):
        super().__init__()
        self.clips = []
        self.current_index = 0
        self.results_dir = ""
        self.kept_count = 0
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._next_frame)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Counter
        self.counter_label = QLabel("No clips to review")
        self.counter_label.setAlignment(Qt.AlignCenter)
        self.counter_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 8px;")
        layout.addWidget(self.counter_label)

        # Video display (QLabel with pixmap)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label, 1)

        # Clip filename label
        self.filename_label = QLabel("")
        self.filename_label.setAlignment(Qt.AlignCenter)
        self.filename_label.setStyleSheet("font-size: 13px; color: #555; padding: 4px;")
        layout.addWidget(self.filename_label)

        # Buttons
        btn_row = QHBoxLayout()

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setMinimumHeight(60)
        self.delete_btn.setStyleSheet(
            "font-size: 18px; font-weight: bold; padding: 16px;"
            "background-color: #f44336; color: white; border-radius: 8px;"
        )
        self.delete_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(self.delete_btn)

        self.keep_btn = QPushButton("Squirrel Present")
        self.keep_btn.setMinimumHeight(60)
        self.keep_btn.setStyleSheet(
            "font-size: 18px; font-weight: bold; padding: 16px;"
            "background-color: #4CAF50; color: white; border-radius: 8px;"
        )
        self.keep_btn.clicked.connect(self._on_keep)
        btn_row.addWidget(self.keep_btn)

        layout.addLayout(btn_row)

        # Replay button
        self.replay_btn = QPushButton("Replay")
        self.replay_btn.setMinimumHeight(35)
        self.replay_btn.setStyleSheet("font-size: 13px; padding: 6px;")
        self.replay_btn.clicked.connect(self._replay)
        layout.addWidget(self.replay_btn)

        self._set_buttons_enabled(False)

    def keyPressEvent(self, event):
        if not self.clips or self.current_index >= len(self.clips):
            return
        if event.key() in (Qt.Key_Right, Qt.Key_Return, Qt.Key_Enter):
            self._on_keep()
        elif event.key() in (Qt.Key_Left, Qt.Key_Delete, Qt.Key_Backspace):
            self._on_delete()
        elif event.key() == Qt.Key_R:
            self._replay()

    def load_clips(self, clips_dir):
        self.clips = sorted(glob.glob(os.path.join(clips_dir, "*.mp4")))
        self.clips_dir = clips_dir
        self.current_index = 0
        self.kept_count = 0
        self.to_delete = []

        if not self.clips:
            self.counter_label.setText("No clips were generated.")
            self._set_buttons_enabled(False)
            return

        self._set_buttons_enabled(True)
        self._play_clip()

    def _play_clip(self):
        if self.current_index >= len(self.clips):
            self._finish_review()
            return

        clip_path = self.clips[self.current_index]
        self.counter_label.setText(f"Clip {self.current_index + 1} / {len(self.clips)}")
        self.filename_label.setText(os.path.basename(clip_path))

        self._stop_playback()
        self.cap = cv2.VideoCapture(clip_path)
        if not self.cap.isOpened():
            self.counter_label.setText(f"ERROR: Could not open {os.path.basename(clip_path)}")
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25
        self.timer.start(int(1000 / fps))

    def _next_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            # Loop: restart from beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                self.timer.stop()
                return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    def _stop_playback(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _replay(self):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _on_keep(self):
        if self.current_index >= len(self.clips):
            return
        self.kept_count += 1
        self.current_index += 1
        self._play_clip()

    def _on_delete(self):
        if self.current_index >= len(self.clips):
            return
        self.to_delete.append(self.clips[self.current_index])
        self.current_index += 1
        self._play_clip()

    def _finish_review(self):
        self._stop_playback()
        self._set_buttons_enabled(False)

        # Delete rejected clips from the output folder
        deleted_count = 0
        for path in self.to_delete:
            try:
                os.remove(path)
                deleted_count += 1
            except OSError:
                pass

        self.counter_label.setText(
            f"Done! {self.kept_count} clip(s) kept, {deleted_count} deleted"
        )
        self.filename_label.setText(self.clips_dir)

        QMessageBox.information(
            self,
            "Verification Complete",
            f"Finished reviewing all clips.\n\n"
            f"Kept: {self.kept_count}\n"
            f"Deleted: {deleted_count}\n\n"
            f"Output folder:\n{self.clips_dir}",
        )

    def _set_buttons_enabled(self, enabled):
        self.keep_btn.setEnabled(enabled)
        self.delete_btn.setEnabled(enabled)
        self.replay_btn.setEnabled(enabled)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Squirrel Tracker")
        self.setMinimumSize(900, 650)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.processing_tab = ProcessingTab()
        self.tabs.addTab(self.processing_tab, "Processing")

        self.verification_tab = VerificationTab()
        self.tabs.addTab(self.verification_tab, "Verification")

        self.processing_tab.processing_done.connect(self._on_processing_done)

    def _on_processing_done(self, clips_dir):
        self.verification_tab.load_clips(clips_dir)
        self.tabs.setCurrentIndex(1)
        self.verification_tab.setFocus()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
