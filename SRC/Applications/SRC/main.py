

# --- BEGIN models/detector.py ---

import os
import sys
import cv2
import numpy as np

try:
    from ultralytics import YOLO, RTDETR
    ULTRA_AVAILABLE = True
except ImportError:
    ULTRA_AVAILABLE = False

try:
    import torch
    import torchvision
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class CancerDetector:
    def __init__(self, model_path="best.pt"):
        self.model_path = model_path
        self.model = None
        self.model_type = "yolo"
        self.is_loaded = False
        self.mode = "skin" # Default mode
        
        has_gpu = False
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except Exception:
            pass
        self.device = torch.device('cuda') if has_gpu else torch.device('cpu')
        
        self._load_model()


    def set_mode(self, mode):
        """Sets the diagnostic mode ('skin' or 'endoscopy')."""
        self.mode = mode
        print(f"Detector mode set to: {self.mode}")

    def load_new_model(self, model_path):
        """Dynamically loads a new model from the given path."""
        self.model_path = model_path
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        resolved_path = get_resource_path(self.model_path)

        if not os.path.exists(resolved_path):
            print(f"Model file not found at {resolved_path}")
            return
            
        try:
            lower_path = self.model_path.lower()
            
            if "faster" in lower_path:
                if not TORCHVISION_AVAILABLE:
                    print("torchvision not installed. Cannot load Faster R-CNN.")
                    return
                self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(num_classes=3, weights_backbone=None)
                state_dict = torch.load(resolved_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                self.model_type = "faster_rcnn"
                self.is_loaded = True
                print(f"Faster R-CNN model loaded: {self.model_path}")
                
            elif "ssd" in lower_path:
                if not TORCHVISION_AVAILABLE:
                    print("torchvision not installed. Cannot load SSD.")
                    return
                # Use strict=False since num_classes might not exactly align with default ssd300 configuration without modifying heads
                self.model = torchvision.models.detection.ssd300_vgg16(num_classes=4, weights_backbone=None)
                state_dict = torch.load(resolved_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.to(self.device)
                self.model.eval()
                self.model_type = "ssd"
                self.is_loaded = True
                print(f"SSD model loaded: {self.model_path}")
                
            else:
                if not ULTRA_AVAILABLE:
                    print("Ultralytics library not installed. Detection disabled.")
                    return
                if "rtdetr" in lower_path or "tr-detr" in lower_path:
                    self.model = RTDETR(resolved_path)
                    self.model_type = "rtdetr"
                elif "yolov11" in lower_path or "yolo11" in lower_path:
                    self.model = YOLO(resolved_path)
                    self.model_type = "yolov11"
                else:
                    self.model = YOLO(resolved_path)
                    self.model_type = "yolo"
                self.is_loaded = True
                print(f"Model loaded successfully: {self.model_path}")
                
        except Exception as e:
            print(f"Error loading model: {e}")

    def draw_detections(self, frame, detections):
        """
        Draws cached or given detections onto the frame.
        """
        processed_frame = frame.copy()
        for d in detections:
            class_name = d.get("class", "Unknown")
            conf = d.get("confidence", 0.0)
            box = d.get("box")
            if box is not None:
                x1, y1, x2, y2 = box
                color = (0, 0, 255) # Red for detection
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{class_name}: {conf:.2f}"
                cv2.putText(processed_frame, label_text, (x1, max(y1 - 10, 0)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return processed_frame

    def detect(self, frame):
        """
        Runs inference on a single frame.
        Returns:
        - processed_frame: The frame with bounding boxes drawn.
        - detections: List of dictionaries containing detection info.
        """
        if not self.is_loaded or self.model is None:
            # If model isn't loaded, just return the original frame
            return frame, []

        try:
            detections = []
            
            if self.model_type in ["faster_rcnn", "ssd"]:
                # torchvision models expect tensor images scaled [0, 1] (C, H, W)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.to(self.device)
                
                with torch.no_grad():
                    predictions = self.model([img_tensor])[0]
                
                boxes = predictions['boxes'].cpu().numpy()
                labels = predictions['labels'].cpu().numpy()
                scores = predictions['scores'].cpu().numpy()
                
                class_map = {0: "Background", 1: "Dysplasia", 2: "SCC", 3: "Other"}
                
                thresh = 0.20 if self.model_type == "ssd" else 0.5
                
                for i in range(len(boxes)):
                    conf = float(scores[i])
                    if conf > thresh: # threshold
                        x1, y1, x2, y2 = map(int, boxes[i])
                        cls_id = int(labels[i])
                        class_name = class_map.get(cls_id, f"Class_{cls_id}")
                        
                        detections.append({
                            "class": class_name,
                            "confidence": conf,
                            "box": [x1, y1, x2, y2]
                        })
                                    
            else:
                # Run inference for YOLO/RT-DETR
                results = self.model(frame, verbose=False, device=self.device)
                r = results[0]
                
                if r.boxes is not None and len(r.boxes) > 0:
                    names = self.model.names
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    cls = r.boxes.cls.cpu().numpy()
                    conf = r.boxes.conf.cpu().numpy()
                    
                    for i in range(len(r.boxes)):
                        class_name = names[int(cls[i])]
                        c = float(conf[i])
                        class_name_lower = class_name.lower()
                        
                        # Standardize name for display or keep original
                        if "dys" in class_name_lower:
                            display_name = "Dysplasia"
                        elif "scc" in class_name_lower:
                            display_name = "SCC"
                        elif class_name_lower == "ap":
                            display_name = "Adenomatous Polyp (AP)"
                        elif class_name_lower == "mp":
                            display_name = "Malignant Polyp (MP)"
                        elif class_name_lower == "mp-ap":
                            display_name = "MP-AP"
                        else:
                            display_name = class_name
                            
                        x1, y1, x2, y2 = map(int, xyxy[i])
                        detections.append({
                            "class": display_name,
                            "confidence": c,
                            "box": [x1, y1, x2, y2]
                        })
            
            processed_frame = self.draw_detections(frame, detections)
            return processed_frame, detections
        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            return frame, []


# --- END models/detector.py ---


# --- BEGIN utils/data_manager.py ---

import os
import sys
import cv2
import shutil
from datetime import datetime

class DataManager:
    def __init__(self, base_dir=None):
        if base_dir is None:
            # Prioritize the specific folder requested by the user
            preferred_dir = r"C:\Users\rugve\OneDrive\Documents\cancer detection-app\data"
            
            # Check write permissions in the preferred directory
            is_writable = False
            try:
                os.makedirs(preferred_dir, exist_ok=True)
                test_file = os.path.join(preferred_dir, ".write_test")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                is_writable = True
            except Exception:
                is_writable = False
                
            if is_writable:
                self.base_dir = preferred_dir
            else:
                # Fallback check for other machines/setups
                if hasattr(sys, 'frozen'):
                    app_dir = os.path.dirname(sys.executable)
                else:
                    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                
                local_data_dir = os.path.join(app_dir, "data")
                
                try:
                    os.makedirs(local_data_dir, exist_ok=True)
                    self.base_dir = local_data_dir
                except Exception:
                    # Final fallback to user documents
                    docs_path = os.path.join(os.path.expanduser("~"), "Documents")
                    self.base_dir = os.path.join(docs_path, "ClinicalCancerApp_Data")
        else:
            self.base_dir = base_dir
            
        if not os.path.exists(self.base_dir):
            try:
                os.makedirs(self.base_dir, exist_ok=True)
            except Exception as e:
                print(f"CRITICAL: Failed to create base data directory: {e}")

    def create_patient_directory(self, patient_id):
        """Creates the folder structure for a specific patient. Sanitizes ID for Windows."""
        import re
        # Remove invalid filename characters: < > : " / \\ | ? *
        clean_id = re.sub(r'[<>:"/\\|?*]', '_', str(patient_id))
        patient_path = os.path.join(self.base_dir, clean_id)
        
        # Define subdirectories
        dirs = {
            "root": patient_path,
            "original_video": os.path.join(patient_path, "original_video"),
            "detected_video": os.path.join(patient_path, "detected_video"),
            "snapshots": os.path.join(patient_path, "snapshots")
        }
        
        try:
            for name, dir_path in dirs.items():
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            print(f"Error creating patient directories: {e}")
            # Ensure we return the dictionary even if creation failed (caller should handle empty dirs)
                
        return dirs

    def save_snapshot(self, patient_id, frame, timestamp_str):
        """Saves a single snapshot frame."""
        dirs = self.create_patient_directory(patient_id)
        filename = f"snapshot_{timestamp_str}.jpg"
        filepath = os.path.join(dirs["snapshots"], filename)
        
        # cv2 uses BGR, assume frame is BGR
        cv2.imwrite(filepath, frame)
        return filepath

    def save_report(self, patient_id, detection_summary, mode="Skin Cancer"):
        """Generates and saves the report.txt."""
        dirs = self.create_patient_directory(patient_id)
        report_path = os.path.join(dirs["root"], "report.txt")
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        content = f"{mode} Analysis System - Patient Report\n"
        content += f"{'='*50}\n"
        content += f"Patient ID: {patient_id}\n"
        content += f"Examination Type: {mode}\n"
        content += f"Date of Scan: {current_time}\n\n"
        content += f"Detection Summary:\n"
        content += f"{'-'*20}\n"
        for item in detection_summary:
            content += f"- {item}\n"
            
        content += f"\n{'='*50}\n"
        content += f"Generated automatically by {mode} Analysis System.\n"
        
        with open(report_path, "w") as f:
            f.write(content)
            
        return report_path

    def copy_original_video(self, patient_id, source_video_path):
        """Copies the original video to the patient folder."""
        if not source_video_path or not os.path.exists(source_video_path):
            return None
            
        dirs = self.create_patient_directory(patient_id)
        filename = os.path.basename(source_video_path)
        dest_path = os.path.join(dirs["original_video"], filename)
        
        try:
            shutil.copy2(source_video_path, dest_path)
            return dest_path
        except Exception as e:
            print(f"Error copying video: {e}")
            return None

# --- END utils/data_manager.py ---


# --- BEGIN utils/video_thread.py ---

import cv2
import time
import os
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

class VideoThread(QThread):
    # Signals to communicate with the Main GUI
    # Emits: (wli_frame, enhanced_frame, detection_frame)
    change_pixmap_signal = pyqtSignal(object, object, object)
    snapshot_signal = pyqtSignal(object, str, str) # frame, timestamp, class_name
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    fps_signal = pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        self._pause_flag = False
        
        self.video_source = None
        self.is_camera = False
        self.detection_enabled = True
        
        # Initialize detector
        self.detector = CancerDetector(model_path="best.pt")
        
        # For saving processed videos
        self.record_flag = False
        self.save_path_root = None
        self.writers = {
            "original": None,
            "enhanced": None,
            "detected": None
        }
        
        # Performance monitoring
        self.current_fps = 30.0
        self.current_confidence = 0.0
        
        # Snapshot cooldown
        self.last_snapshot_time = 0
        self.snapshot_cooldown = 3.0 # seconds
        
        # Frame skipping variables
        self.frame_count = 0
        self.last_detections = []

    def apply_nbi_simulation(self, frame):
        """
        Simulates Narrow Band Imaging (NBI).
        NBI uses two specific wavelengths of light (415nm blue and 540nm green).
        In simulation: 
        - Dim the Red channel.
        - Enhance Green and Blue channels.
        - Increase contrast.
        """
        if frame is None: return None
        
        # Split channels
        b, g, r = cv2.split(frame)
        
        # NBI heuristic: Red is absorbed by hemoglobin, so we darken it
        r = cv2.multiply(r, 0.2) 
        # Green and Blue are reflected by surface vessels
        g = cv2.multiply(g, 1.2)
        b = cv2.multiply(b, 1.2)
        
        # Re-merge
        nbi = cv2.merge([b, g, r])
        
        # Increase contrast
        lab = cv2.cvtColor(nbi, cv2.COLOR_BGR2LAB)
        l, a, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b_chan))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return final

    def run(self):
        try:
            self._run_flag = True
            self.frame_count = 0
            self.last_detections = []
            if self.video_source is None:
                self.status_signal.emit("Error: No source")
                return
                
            if self.is_camera:
                # Prioritize default backend
                cap = cv2.VideoCapture(self.video_source)
                if not cap.isOpened():
                    # Fallback to DSHOW for Windows if needed
                    cap = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
                
                if not cap.isOpened():
                    self.status_signal.emit("Error: Camera unavailable")
                    return
            else:
                cap = cv2.VideoCapture(self.video_source)

            if not cap.isOpened():
                self.status_signal.emit("Error: Could not open source")
                return
                
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_camera else 0
            frame_count = 0
            fps_start_time = time.time()
            fps_frame_count = 0
            
            if not self.detector.is_loaded:
                self.status_signal.emit("Warning: AI Model Not Loaded")
            else:
                self.status_signal.emit("AI Model Loaded. Ready.")
                
            self.status_signal.emit("Examination Active")

            failed_reads = 0
            while self._run_flag:
                iteration_start = time.time()
                if self._pause_flag:
                    time.sleep(0.1)
                    continue
                    
                ret, frame = cap.read()
                if not ret or frame is None:
                    failed_reads += 1
                    if failed_reads > 50:  # Allow 50 retries (approx 1.5 seconds) for camera to warm up
                        break
                    time.sleep(0.03)
                    continue
                
                failed_reads = 0
                    
                # 1. Original (WLI)
                wli_frame = frame.copy()
                
                # 2. Enhanced (NBI)
                enhanced_frame = self.apply_nbi_simulation(wli_frame)
                
                # 3. Detection (Run AI on Original with frame skipping)
                if self.detection_enabled:
                    if self.detector.model_type in ["faster_rcnn", "ssd"]:
                        skip_interval = 12
                    else:
                        skip_interval = 3

                    if self.frame_count == 0 or (self.frame_count % skip_interval == 0):
                        detect_frame, detections = self.detector.detect(wli_frame)
                        self.last_detections = detections
                    else:
                        detections = self.last_detections
                        detect_frame = self.detector.draw_detections(wli_frame, detections)
                else:
                    detections = []
                    detect_frame = wli_frame.copy()
                
                self.frame_count += 1
                
                # Initialize VideoWriters if recording
                if self.record_flag and self.save_path_root and self.writers["original"] is None:
                    try:
                        h, w = wli_frame.shape[:2]
                        # Use a more compatible codec if possible
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                        
                        self.writers["original"] = cv2.VideoWriter(
                            os.path.join(self.save_path_root, "original_feed.mp4"), fourcc, fps, (w, h))
                        self.writers["enhanced"] = cv2.VideoWriter(
                            os.path.join(self.save_path_root, "enhanced_feed.mp4"), fourcc, fps, (w, h))
                        self.writers["detected"] = cv2.VideoWriter(
                            os.path.join(self.save_path_root, "detection_feed.mp4"), fourcc, fps, (w, h))
                    except Exception as e:
                        print(f"Failed to initialize video writers: {e}")
                        self.status_signal.emit("Warning: Recording disabled due to error")
                    
                # Write to videos
                try:
                    if self.writers["original"]:
                        self.writers["original"].write(wli_frame)
                    if self.writers["enhanced"]:
                        self.writers["enhanced"].write(enhanced_frame)
                    if self.writers["detected"]:
                        self.writers["detected"].write(detect_frame)
                except Exception as e:
                    print(f"Error writing frame: {e}")
                    
                # Auto-snapshot on detection
                if detections:
                    self.current_confidence = max(d["confidence"] for d in detections)
                    
                    # If high confidence and cooldown passed
                    if self.current_confidence > 0.5:
                        current_time = time.time()
                        if current_time - self.last_snapshot_time > self.snapshot_cooldown:
                            self.last_snapshot_time = current_time
                            timestamp = time.strftime("%H%M%S")
                            class_names = ", ".join(set(d["class"] for d in detections))
                            self.snapshot_signal.emit(detect_frame.copy(), timestamp, class_names)
                else:
                    self.current_confidence = 0.0

                # Emit all 3 frames
                self.change_pixmap_signal.emit(wli_frame, enhanced_frame, detect_frame)
                
                # Update progress and FPS
                frame_count += 1
                fps_frame_count += 1
                
                current_time_fps = time.time()
                elapsed = current_time_fps - fps_start_time
                if elapsed >= 1.0:
                    self.current_fps = fps_frame_count / elapsed
                    self.fps_signal.emit(self.current_fps)
                    fps_start_time = current_time_fps
                    fps_frame_count = 0
                    
                if not self.is_camera and total_frames > 0:
                    progress = int((frame_count / total_frames) * 100)
                    self.progress_signal.emit(progress)
                
                # Sync to FPS dynamically
                elapsed_processing = time.time() - iteration_start
                target_delay = 1.0 / fps
                sleep_time = target_delay - elapsed_processing
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    time.sleep(0.001) 
        except Exception as e:
            print(f"Thread Error: {e}")
            self.status_signal.emit(f"Critical Thread Error: {e}")
        finally:
            if 'cap' in locals() and cap:
                import threading
                threading.Thread(target=cap.release, daemon=True).start()
            for key in self.writers:
                if self.writers[key]:
                    self.writers[key].release()
                    self.writers[key] = None
            self.status_signal.emit("Stopped")

    def stop(self):
        self._run_flag = False
        self.wait()

    def set_detection_enabled(self, enabled):
        """Enable or disable the AI detection model."""
        self.detection_enabled = enabled
        
    def toggle_pause(self):
        self._pause_flag = not self._pause_flag
        return self._pause_flag

    def change_model(self, model_path):
        """Changes the AI model being used for detection."""
        import threading
        
        def _load_task():
            was_running = self._run_flag and not self._pause_flag
            if was_running:
                self._pause_flag = True
                time.sleep(0.2) # Give loop time to pause
    
            self.status_signal.emit(f"Loading new model: {os.path.basename(model_path)}...")
            
            try:
                self.detector.load_new_model(model_path)
                self.frame_count = 0
                self.last_detections = []
                if self.detector.is_loaded:
                    self.status_signal.emit(f"Model changed to {os.path.basename(model_path)}")
                else:
                    self.status_signal.emit("Warning: Failed to load new model")
            except Exception as e:
                self.status_signal.emit(f"Error loading model: {e}")
    
            if was_running:
                self._pause_flag = False
                
        threading.Thread(target=_load_task, daemon=True).start()

# --- END utils/video_thread.py ---


# --- BEGIN ui/main_window_v2.py ---

import os
import sys
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QFileDialog, 
    QMessageBox, QFrame, QSizePolicy, QApplication, 
    QProgressBar, QGraphicsDropShadowEffect, QSlider, QSpacerItem,
    QComboBox, QCheckBox
)
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, pyqtSlot, QSize, QTimer, QDateTime
from PyQt5.QtMultimedia import QCameraInfo


def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class VideoPanel(QFrame):
    def __init__(self, title, icon, border_id, parent=None):
        super().__init__(parent)
        self.setObjectName(border_id)
        self.setProperty("class", "videoPanelCard")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 1. Title Bar
        title_bar = QFrame()
        title_bar.setProperty("class", "panelTitleBar")
        title_layout = QHBoxLayout(title_bar)
        
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 16px;")
        
        title_label = QLabel(title)
        title_label.setProperty("class", "panelTitleLabel")
        
        title_layout.addStretch()
        title_layout.addWidget(icon_label)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        # 2. Video Area
        self.canvas = QLabel()
        self.canvas.setProperty("class", "videoArea")
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(300, 400)
        self.canvas.setText("INITIALIZING FEED...")
        self.canvas.setStyleSheet("color: #334155; font-weight: bold;")
        
        layout.addWidget(title_bar)
        layout.addWidget(self.canvas)

class SnapshotCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "snapshotCard")
        self.setFixedSize(400, 370)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        self.icon = QLabel("📷")
        self.icon.setAlignment(Qt.AlignCenter)
        self.icon.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.icon.setStyleSheet("font-size: 48px; color: #cbd5e1;")
        
        self.label = QLabel("No Detections Yet")
        self.label.setProperty("class", "snapshotLabel")
        self.label.setStyleSheet("font-size: 14px; font-weight: 600;")
        self.label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.icon)
        layout.addWidget(self.label)
        
    def set_image(self, pixmap):
        self.icon.setPixmap(pixmap.scaled(self.width()-20, self.height()-40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.label.hide()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-Powered Clinical Diagnostic System")
        self.setMinimumSize(1280, 720)
        self.showMaximized()
        
        self.data_manager = DataManager()
        self.video_thread = VideoThread()
        
        self.init_ui()
        self.setup_signals()
        self.load_styles()
        
        # Auto-detect cameras
        self.available_cameras = [c.description() for c in QCameraInfo.availableCameras()]
        self.update_camera_dropdown()
        self.camera_check_timer = QTimer()
        self.camera_check_timer.timeout.connect(self.check_for_new_cameras)
        self.camera_check_timer.start(2000)
        
        # Timer for footer clock removed

    def load_styles(self):
        style_path = get_resource_path(os.path.join("assets", "stylesheet.qss"))
        if os.path.exists(style_path):
            with open(style_path, "r") as f:
                self.setStyleSheet(f.read())

    def check_for_new_cameras(self):
        try:
            current_cameras = [c.description() for c in QCameraInfo.availableCameras()]
            if not hasattr(self, 'available_cameras'):
                self.available_cameras = current_cameras
                self.update_camera_dropdown()
                return
            if self.available_cameras != current_cameras:
                added = set(current_cameras) - set(self.available_cameras)
                self.available_cameras = current_cameras
                self.update_camera_dropdown()
                if added or not self.video_thread.isRunning():
                    # Only auto-start if we are in camera mode or nothing is running
                    if getattr(self.video_thread, 'is_camera', True) or not self.video_thread.isRunning():
                        self.handle_start()
        except Exception as e:
            print(f"Error checking cameras: {e}")

    def setup_signals(self):
        self.video_thread.change_pixmap_signal.connect(self.update_feeds)
        self.video_thread.snapshot_signal.connect(self.on_snapshot)
        self.video_thread.status_signal.connect(lambda s: self.status_dot_lbl.setText("● " + s))
        self.video_thread.progress_signal.connect(self.update_progress)
        self.video_thread.fps_signal.connect(self.update_fps)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. TOP HEADER
        header = QFrame()
        header.setObjectName("headerFrame")
        header.setFixedHeight(85)
        header_layout = QHBoxLayout(header)
        
        ecg_left = QLabel("")
        ecg_left.setStyleSheet("color: #3b82f6; font-size: 24px; font-weight: bold;")
        
        title = QLabel("AI-POWERED CLINICAL DIAGNOSTIC SYSTEM")
        title.setObjectName("headerTitle")
        title.setAlignment(Qt.AlignCenter)
        
        ecg_right = QLabel("")
        ecg_right.setStyleSheet("color: #3b82f6; font-size: 24px; font-weight: bold;")
        
        header_layout.addStretch()
        header_layout.addWidget(ecg_left)
        header_layout.addSpacing(20)
        header_layout.addWidget(title)
        header_layout.addSpacing(20)
        header_layout.addWidget(ecg_right)
        header_layout.addStretch()
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0,0,0,30))
        header.setGraphicsEffect(shadow)
        
        main_layout.addWidget(header)

        # 2. VIDEO PANELS
        video_container = QWidget()
        video_layout = QHBoxLayout(video_container)
        video_layout.setContentsMargins(20, 0, 20, 0)
        video_layout.setSpacing(15)
        
        self.panel_wli = VideoPanel("White Light Video", "", "panelWLI")
        self.panel_ai = VideoPanel("Cancer Detection AI", "", "panelAI")
        
        video_layout.addWidget(self.panel_wli)
        video_layout.addWidget(self.panel_ai)
        
        main_layout.addWidget(video_container, stretch=4)

        # 3. SNAPSHOT SECTION
        snapshot_section = QFrame()
        snapshot_section.setObjectName("snapshotArea")
        snap_vbox = QVBoxLayout(snapshot_section)
        snap_vbox.setContentsMargins(0, 0, 0, 0)
        snap_vbox.setSpacing(5)
        
        snap_header = QHBoxLayout()
        snap_icon = QLabel("")
        snap_icon.setStyleSheet("font-size: 18px;")
        snap_title = QLabel("")
        snap_title.setStyleSheet("font-weight: 800; font-size: 15px; color: #1e293b;")
        snap_header.addStretch()
        snap_header.addWidget(snap_icon)
        snap_header.addWidget(snap_title)
        snap_header.addStretch()
        snap_vbox.addLayout(snap_header)
        
        self.snap_layout = QHBoxLayout()
        self.snap_layout.setContentsMargins(20, 0, 20, 0)
        self.snap_layout.setSpacing(10)
        self.snap_cards = [SnapshotCard() for _ in range(5)]
        for card in self.snap_cards:
            self.snap_layout.addWidget(card)
        snap_vbox.addLayout(self.snap_layout)
        
        main_layout.addWidget(snapshot_section, stretch=4)

        # 4. BOTTOM CONTROL PANEL
        bottom_panel = QFrame()
        bottom_panel.setObjectName("bottomPanel")
        bottom_panel.setFixedHeight(250)
        
        bottom_layout = QHBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(20, 10, 40, 10)
        bottom_layout.setSpacing(20)
        bottom_layout.setAlignment(Qt.AlignVCenter)
        
        app_logo_lbl = QLabel()
        app_logo_lbl.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        logo_path = get_resource_path(os.path.join("assets", "App logo.png"))
        app_logo_pixmap = QPixmap(logo_path)
        
        if not app_logo_pixmap.isNull():
            app_logo_lbl.setPixmap(app_logo_pixmap.scaled(1000, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            app_logo_lbl.setText("🏥")
            app_logo_lbl.setStyleSheet("font-size: 50px;")
            
        bottom_layout.addWidget(app_logo_lbl)
        bottom_layout.addSpacing(40)
        
        # Patient ID Block
        pid_vbox = QVBoxLayout()
        pid_vbox.setAlignment(Qt.AlignVCenter)
        pid_vbox.setSpacing(5)
        
        pid_lbl = QLabel("Patient ID")
        pid_lbl.setStyleSheet("font-weight: 800; font-size: 14px; color: #64748b;")
        self.pid_input = QLineEdit()
        self.pid_input.setObjectName("patientInput")
        self.pid_input.setPlaceholderText("👤 Enter Patient ID")
        self.pid_input.setFixedSize(220, 45)
        self.pid_input.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        
        pid_vbox.addWidget(pid_lbl)
        pid_vbox.addWidget(self.pid_input)
        
        bottom_layout.addLayout(pid_vbox)
        bottom_layout.addSpacing(20)

        # Model Selection Block moved down to progress area

        
        # Buttons
        self.btn_save = self.create_action_btn("SAVE", "btnSave", "💾")
        self.btn_load = self.create_action_btn("Load Video", "btnLoad", "📂")
        self.btn_start = self.create_action_btn("Start Camera", "btnStart", "📷")
        self.btn_pause = self.create_action_btn("Pause", "btnPause", "⏸")
        
        bottom_layout.addWidget(self.btn_save, alignment=Qt.AlignVCenter)
        bottom_layout.addSpacing(5)
        bottom_layout.addWidget(self.btn_load, alignment=Qt.AlignVCenter)
        bottom_layout.addSpacing(5)
        bottom_layout.addWidget(self.btn_start, alignment=Qt.AlignVCenter)
        bottom_layout.addSpacing(5)
        bottom_layout.addWidget(self.btn_pause, alignment=Qt.AlignVCenter)
        
        bottom_layout.addSpacing(40)
        
        # Progress Area
        prog_vbox = QVBoxLayout()
        prog_vbox.setAlignment(Qt.AlignVCenter)
        prog_vbox.setSpacing(5)
        
        prog_header = QHBoxLayout()
        prog_title = QLabel("Processing Progress")
        prog_title.setObjectName("progressTitle")
        self.prog_perc = QLabel("0%")
        self.prog_perc.setStyleSheet("font-weight: 900; font-size: 12px; color: #0f172a;")
        prog_header.addWidget(prog_title)
        prog_header.addStretch()
        prog_header.addWidget(self.prog_perc)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedSize(320, 12)
        self.progress_bar.setTextVisible(False)
        
        status_layout = QHBoxLayout()
        self.status_dot_lbl = QLabel("● System Ready")
        self.status_dot_lbl.setStyleSheet("color: #10b981; font-size: 11px; font-weight: 700;")
        
        self.fps_lbl = QLabel("FPS: --")
        self.fps_lbl.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 700;")
        
        status_layout.addWidget(self.status_dot_lbl)
        status_layout.addStretch()
        status_layout.addWidget(self.fps_lbl)
        
        prog_vbox.addLayout(prog_header)
        prog_vbox.addWidget(self.progress_bar)
        prog_vbox.addLayout(status_layout)
        
        # Model Selection Block
        model_lbl = QLabel("AI Model")
        model_lbl.setStyleSheet("font-weight: 800; font-size: 12px; color: #64748b;")
        self.model_dropdown = QComboBox()
        self.model_dropdown.setObjectName("modelDropdown")
        self.model_dropdown.setFixedSize(320, 35)
        self.model_dropdown.setStyleSheet("font-size: 12px; font-weight: bold; padding: 5px; background: white;")
        
        # Add models map
        self.models_map = {
            "YOLOv8 Default": "best.pt",
            "YOLOv11": "Yolov11 best.pt/best.pt",
            "Faster R-CNN": "faster rnn best.pt file/best (1).pt",
            "RT-DETR": "RT-DETR best.pt file/best.pt",
            "SSD": "SSD best.pt file/best.pt"
        }
        self.model_dropdown.addItems(list(self.models_map.keys()))

        # Checkbox to enable/disable AI detection
        self.cb_enable_detection = QCheckBox("Enable AI Cancer Detection")
        self.cb_enable_detection.setObjectName("enableDetectionCheckBox")
        self.cb_enable_detection.setChecked(True)
        
        # Camera Selection Block
        camera_lbl = QLabel("Camera Source")
        camera_lbl.setStyleSheet("font-weight: 800; font-size: 12px; color: #64748b;")
        self.camera_dropdown = QComboBox()
        self.camera_dropdown.setObjectName("cameraDropdown")
        self.camera_dropdown.setFixedSize(320, 35)
        self.camera_dropdown.setStyleSheet("font-size: 12px; font-weight: bold; padding: 5px; background: white;")
        
        prog_vbox.addSpacing(10)
        prog_vbox.addWidget(model_lbl)
        prog_vbox.addWidget(self.model_dropdown)
        prog_vbox.addSpacing(8)
        prog_vbox.addWidget(self.cb_enable_detection)
        prog_vbox.addSpacing(10)
        prog_vbox.addWidget(camera_lbl)
        prog_vbox.addWidget(self.camera_dropdown)
        
        bottom_layout.addLayout(prog_vbox)
        
        main_layout.addWidget(bottom_panel)



        # Button Actions
        self.btn_load.clicked.connect(self.handle_load)
        self.btn_start.clicked.connect(self.handle_start)
        self.btn_pause.clicked.connect(self.handle_pause)
        self.btn_save.clicked.connect(self.handle_save)
        self.model_dropdown.currentIndexChanged.connect(self.handle_model_change)
        self.camera_dropdown.currentIndexChanged.connect(self.handle_camera_change)
        self.cb_enable_detection.toggled.connect(self.handle_detection_toggle)

    def handle_model_change(self):
        selected_model = self.model_dropdown.currentText()
        model_path = self.models_map.get(selected_model)
        if model_path:
            self.video_thread.change_model(model_path)

    def handle_camera_change(self):
        if self.video_thread.isRunning() and getattr(self.video_thread, 'is_camera', False):
            self.handle_start()

    def handle_detection_toggle(self, checked):
        self.video_thread.set_detection_enabled(checked)
        if not checked:
            self.status_dot_lbl.setText("● AI Detection Bypassed")
            self.status_dot_lbl.setStyleSheet("color: #64748b; font-weight: bold;")
        else:
            self.status_dot_lbl.setText("● SYSTEM ACTIVE")
            self.status_dot_lbl.setStyleSheet("color: #10b981; font-weight: bold;")

    def update_camera_dropdown(self):
        self.camera_dropdown.blockSignals(True)
        current_selection = self.camera_dropdown.currentText()
        self.camera_dropdown.clear()
        
        if not self.available_cameras:
            self.camera_dropdown.addItem("Default Camera (Index 0)")
        else:
            for idx, cam_name in enumerate(self.available_cameras):
                self.camera_dropdown.addItem(f"{cam_name} (Camera {idx})")
                
        index = self.camera_dropdown.findText(current_selection)
        if index >= 0:
            self.camera_dropdown.setCurrentIndex(index)
        else:
            self.camera_dropdown.setCurrentIndex(0)
        self.camera_dropdown.blockSignals(False)

    def create_action_btn(self, main_text, obj_id, icon):
        btn = QPushButton()
        btn.setObjectName(obj_id)
        btn.setProperty("class", "actionButton")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedSize(145, 45)
        
        layout = QHBoxLayout(btn)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignCenter)
        
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet("font-size: 18px; color: white;")
        icon_lbl.setAlignment(Qt.AlignVCenter)
        
        text_vbox = QVBoxLayout()
        text_vbox.setAlignment(Qt.AlignVCenter)
        text_vbox.setSpacing(0)
        
        main_lbl = QLabel(main_text)
        main_lbl.setStyleSheet("font-weight: 850; font-size: 12px; color: white;")
        
        text_vbox.addWidget(main_lbl)
        
        layout.addWidget(icon_lbl)
        layout.addLayout(text_vbox)
        return btn



    @pyqtSlot(object, object, object)
    def update_feeds(self, wli, enhanced, detected):
        self.panel_wli.canvas.setPixmap(self.cv_to_pixmap(wli, self.panel_wli.canvas))
        self.panel_ai.canvas.setPixmap(self.cv_to_pixmap(detected, self.panel_ai.canvas))

    def cv_to_pixmap(self, img, label):
        if img is None or img.size == 0: 
            return QPixmap()
        try:
            h, w, ch = img.shape
            if h <= 0 or w <= 0: return QPixmap()
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bytes_per_line = rgb.strides[0]
            # Create QImage and copy data to be safe (prevents crashes from buffer expiration)
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            
            # Scale to fit label while maintaining aspect ratio
            return QPixmap.fromImage(qimg).scaled(
                label.width(), label.height(), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        except Exception as e:
            print(f"Conversion error: {e}")
            return QPixmap()

    @pyqtSlot(int)
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.prog_perc.setText(f"{value}%")

    @pyqtSlot(float)
    def update_fps(self, fps):
        self.fps_lbl.setText(f"FPS: {fps:.1f}")

    @pyqtSlot(object, str, str)
    def on_snapshot(self, frame, timestamp, label):
        # Shift
        for i in range(4, 0, -1):
            if self.snap_cards[i-1].label.isHidden(): # Already has image
                self.snap_cards[i].set_image(self.snap_cards[i-1].icon.pixmap())
            
        pixmap = self.cv_to_pixmap(frame, self.snap_cards[0])
        self.snap_cards[0].set_image(pixmap)
    
        if self.pid_input.text():
            self.data_manager.save_snapshot(self.pid_input.text(), frame, timestamp)

    def handle_load(self):
        """Handles loading a video file for analysis."""
        try:
            pid = self.pid_input.text().strip()
            if not pid:
                # Auto-generate a generic session ID
                pid = f"Session_{int(time.time())}"
                self.pid_input.setText(pid)
            
            # Reverted to standard Windows dialog for better user experience
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Video Source", "", 
                "Videos (*.mp4 *.avi *.mov *.mkv)"
            )
            
            if path:
                # Update UI to show we are working
                self.status_dot_lbl.setText("● PREPARING VIDEO...")
                self.status_dot_lbl.setStyleSheet("color: #3b82f6; font-weight: bold;")
                QApplication.processEvents() # Ensure UI updates
                
                # Stop any existing video safely
                if self.video_thread.isRunning():
                    self.video_thread.stop()
                
                self.video_thread.video_source = path
                self.video_thread.is_camera = False
                
                # Setup recording directories
                dirs = self.data_manager.create_patient_directory(pid)
                if not dirs or not dirs.get("detected_video"):
                    raise Exception("Failed to initialize storage directories. Check folder permissions.")
                
                self.video_thread.save_path_root = dirs["detected_video"]
                self.video_thread.record_flag = True
                
                # Start processing
                self.video_thread.start()
                
        except Exception as e:
            QMessageBox.critical(self, "System Error", f"An error occurred while loading the video:\n{str(e)}")
            print(f"DEBUG: handle_load error: {e}")

    def handle_start(self):
        pid = self.pid_input.text().strip()
        if not pid:
            # Auto-generate a generic session ID so the camera can start immediately
            pid = f"Session_{int(time.time())}"
            self.pid_input.setText(pid)
            
        self.video_thread.stop()
        
        # Use selected camera index from dropdown
        selected_idx = self.camera_dropdown.currentIndex()
        self.video_thread.video_source = max(0, selected_idx)
        self.video_thread.is_camera = True
        
        # Setup recording
        dirs = self.data_manager.create_patient_directory(pid)
        self.video_thread.save_path_root = dirs["detected_video"]
        self.video_thread.record_flag = True
        
        self.video_thread.start()

    def handle_pause(self):
        paused = self.video_thread.toggle_pause()
        self.status_dot_lbl.setText("● PAUSED" if paused else "● SYSTEM ACTIVE")

    def handle_save(self):
        pid = self.pid_input.text().strip()
        if not pid:
            QMessageBox.warning(self, "Patient ID Required", "Please enter a Patient ID.")
            return

        if self.video_thread.isRunning():
            self.video_thread.stop()
            
        dirs = self.data_manager.create_patient_directory(pid)
        
        # If it was a file, copy the exact source file to original_video
        if not self.video_thread.is_camera and isinstance(self.video_thread.video_source, str):
            self.data_manager.copy_original_video(pid, self.video_thread.video_source)
            
        import shutil
        import os
        orig_rec = os.path.join(dirs["detected_video"], "original_feed.mp4")
        
        if os.path.exists(orig_rec):
            if self.video_thread.is_camera:
                shutil.move(orig_rec, os.path.join(dirs["original_video"], "camera_feed.mp4"))
            else:
                os.remove(orig_rec) # Remove duplicate low-res recording
                
        self.data_manager.save_report(pid, ["Diagnostic session saved successfully."])
        QMessageBox.information(self, "Data Saved", f"Diagnostic data for patient {pid} has been saved to the app's 'data' folder!")

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

# --- END ui/main_window_v2.py ---


# --- BEGIN main.py ---

import sys
import os
import io
import traceback
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QIcon

# Fix PyInstaller console=False encoding issue
if sys.stdout is None:
    sys.stdout = io.TextIOWrapper(
        open(os.devnull, 'wb'),
        encoding='utf-8'
    )

if sys.stderr is None:
    sys.stderr = io.TextIOWrapper(
        open(os.devnull, 'wb'),
        encoding='utf-8'
    )

def exception_hook(exctype, value, tb):
    """Global exception handler to show errors in a message box."""
    err_msg = "".join(traceback.format_exception(exctype, value, tb))
    # Log to a file in the data directory if possible
    try:
        from datetime import datetime
        preferred_dir = r"C:\Users\rugve\OneDrive\Documents\cancer detection-app\data"
        
        is_writable = False
        try:
            os.makedirs(preferred_dir, exist_ok=True)
            test_file = os.path.join(preferred_dir, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            is_writable = True
        except Exception:
            is_writable = False
            
        if is_writable:
            log_path = os.path.join(preferred_dir, "error_log.txt")
            log_location_str = "the local 'data' folder"
        else:
            if hasattr(sys, 'frozen'):
                app_dir = os.path.dirname(sys.executable)
            else:
                app_dir = os.path.abspath(os.path.dirname(__file__))
                
            local_data_dir = os.path.join(app_dir, "data")
            
            try:
                os.makedirs(local_data_dir, exist_ok=True)
                log_path = os.path.join(local_data_dir, "error_log.txt")
                log_location_str = "the local 'data' folder"
            except Exception:
                log_path = os.path.join(os.path.expanduser("~"), "Documents", "ClinicalCancerApp_Data", "error_log.txt")
                log_location_str = "your Documents folder"
            
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"\n--- CRASH LOG: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write(err_msg)
            f.write("-" * 50 + "\n")
    except Exception as log_e:
        print(f"Failed to write to log file: {log_e}")
        log_location_str = "your Documents folder"
        
    if QApplication.instance():
        QMessageBox.critical(None, "System Error", 
                            f"An unexpected error occurred. A log file has been saved to {log_location_str}.\n\nError: {value}")
    else:
        print(err_msg)
    sys.exit(1)

sys.excepthook = exception_hook


def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def main():
    # Setup application
    app = QApplication(sys.argv)

    # Set Window Icon
    icon_path = get_resource_path(os.path.join("assets", "icon.ico"))
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # Initialize and show main window
    window = MainWindow()
    window.show()

    # Execute application loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
# --- END main.py ---
