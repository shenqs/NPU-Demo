#!/usr/bin/env python3
"""Orange Pi AI Pro - YOLOv8 Detection App (ACL NPU, Stability Optimized).

Uses raw Ascend Computing Language (ACL) API with pre-compiled .om model
for deterministic memory access patterns and explicit resource lifecycle.
"""
import os
import sys
import signal
import threading
import time
import subprocess
from collections import deque

import cv2
import numpy as np
import acl

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# ==============================================================================
# Configuration
# ==============================================================================
DEMO_DIR = "/home/HwHiAiUser/npu_demo"
VIDEO_DIR = f"{DEMO_DIR}/videos"
OM_MODEL = os.path.join(DEMO_DIR, "yolov8n_npu.om")

MAX_FPS = 30                       # Cap inference rate
INTER_FRAME_DELAY_MS = 0          # No artificial delay between inferences
DISPLAY_SKIP_THRESHOLD_FPS = 20   # Skip alternate display updates above this FPS
THERMAL_PAUSE_THRESHOLD = 75      # Pause inference above this temp (C)
THERMAL_RESUME_THRESHOLD = 70     # Resume below this temp (hysteresis)
MAX_CONSECUTIVE_FAILURES = 3      # Pause after N consecutive inference errors
DETECTION_TIMES_MAXLEN = 100      # Bounded deque for timing history

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

COLORS = np.random.RandomState(42).randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)

# ==============================================================================
# Graceful shutdown
# ==============================================================================
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    print(f"\n[Watchdog] Signal {signum} received, shutting down...")
    _shutdown_requested = True


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ==============================================================================
# ACL Inference Engine
# ==============================================================================
class AclInferenceEngine:
    """YOLOv8 inference using raw ACL API with pre-compiled .om model.

    Provides explicit device memory lifecycle and buffer reuse to minimize
    DDR pressure on the Ascend 310B4.
    """

    def __init__(self, om_path):
        self.om_path = om_path
        self.model_id = None
        self.context = None
        self.stream = None
        self.input_buffer = None
        self.output_buffer = None
        self.input_dataset = None
        self.output_dataset = None
        self.input_size = 0
        self.output_size = 0
        self._cleaned_up = False

        # Pre-allocated host-side buffers (reused every frame)
        self._padded_buf = np.full((640, 640, 3), 114, dtype=np.uint8)
        self._input_flat = None  # Allocated after model load (need exact size)
        self._output_buf = None  # Allocated after model load

        try:
            self._init_acl()
            self._load_model()
        except Exception:
            # Clean up partial ACL state so torch_npu fallback can init cleanly
            self.cleanup()
            raise

    def _init_acl(self):
        ret = acl.init()
        if ret != 0:
            raise RuntimeError(f"acl.init failed: {ret}")

        ret = acl.rt.set_device(0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.set_device failed: {ret}")

        self.context, ret = acl.rt.create_context(0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.create_context failed: {ret}")

        self.stream, ret = acl.rt.create_stream()
        if ret != 0:
            print(f"[ACL] Warning: create_stream failed ({ret}), using synchronous mode")
            self.stream = None

        print(f"[ACL] Initialized: device=0, context={self.context}")

    def _load_model(self):
        if not os.path.exists(self.om_path):
            raise FileNotFoundError(f"OM model not found: {self.om_path}")

        print(f"[ACL] Loading model from: {self.om_path}", flush=True)
        self.model_id, ret = acl.mdl.load_from_file(self.om_path)
        print(f"[ACL] load_from_file returned: model_id={self.model_id}, ret={ret}", flush=True)
        if ret != 0:
            raise RuntimeError(f"Failed to load OM model: {ret}")

        # Create model description to query input/output sizes
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        if ret != 0:
            raise RuntimeError(f"Failed to get model description: {ret}")

        self.input_size = acl.mdl.get_input_size_by_index(self.model_desc, 0)
        self.output_size = acl.mdl.get_output_size_by_index(self.model_desc, 0)

        print(f"[ACL] Model loaded: {os.path.basename(self.om_path)}", flush=True)
        print(f"[ACL] Input: {self.input_size} bytes, Output: {self.output_size} bytes", flush=True)

        # Allocate device memory (0 = ACL_MEM_MALLOC_HUGE_FIRST)
        self.input_buffer, ret = acl.rt.malloc(self.input_size, 0)
        print(f"[ACL] input malloc: ret={ret}, buf={self.input_buffer}", flush=True)
        if ret != 0:
            raise RuntimeError(f"Failed to allocate input buffer: {ret}")

        self.output_buffer, ret = acl.rt.malloc(self.output_size, 0)
        print(f"[ACL] output malloc: ret={ret}, buf={self.output_buffer}", flush=True)
        if ret != 0:
            raise RuntimeError(f"Failed to allocate output buffer: {ret}")

        # Create input dataset
        self.input_dataset = acl.mdl.create_dataset()
        print(f"[ACL] input dataset: {self.input_dataset}", flush=True)
        input_db = acl.create_data_buffer(self.input_buffer, self.input_size)
        print(f"[ACL] input data_buffer: {input_db}", flush=True)
        if input_db is None:
            raise RuntimeError("Failed to create input data buffer")
        acl.mdl.add_dataset_buffer(self.input_dataset, input_db)

        # Create output dataset
        self.output_dataset = acl.mdl.create_dataset()
        output_db = acl.create_data_buffer(self.output_buffer, self.output_size)
        if output_db is None:
            raise RuntimeError("Failed to create output data buffer")
        acl.mdl.add_dataset_buffer(self.output_dataset, output_db)

        # Pre-allocate host-side reusable buffers
        self._input_flat = np.zeros(self.input_size // 4, dtype=np.float32)
        self._output_buf = np.zeros(self.output_size // 4, dtype=np.float32)

        print("[ACL] Device memory allocated, ready for inference")

    def preprocess(self, img):
        """Letterbox-resize image to 640x640, reusing pre-allocated buffer."""
        h, w = img.shape[:2]
        scale = 640 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (new_w, new_h))

        # Reuse padded buffer
        self._padded_buf[:] = 114
        self._padded_buf[:new_h, :new_w] = img_resized

        # Normalize uint8->float32 and transpose HWC->CHW directly into pre-allocated buffer
        # transpose on contiguous array creates a view (no copy), np.multiply broadcasts in-place
        img_chw = self._padded_buf.transpose(2, 0, 1)  # uint8 view, shape (3,640,640)
        np.multiply(img_chw, np.float32(1.0 / 255.0), out=self._input_flat.reshape(3, 640, 640))

        return scale, new_w, new_h, w, h

    def infer(self, img, conf_threshold=0.25, iou_threshold=0.45):
        """Run full inference pipeline: preprocess -> NPU execute -> postprocess."""
        # Ensure ACL context is set on this thread
        acl.rt.set_context(self.context)

        scale, new_w, new_h, orig_w, orig_h = self.preprocess(img)

        # Host -> Device (1 = ACL_MEMCPY_HOST_TO_DEVICE)
        ret = acl.rt.memcpy(
            self.input_buffer, self.input_size,
            self._input_flat.ctypes.data, self.input_size,
            1
        )
        if ret != 0:
            raise RuntimeError(f"H2D memcpy failed: {ret}")

        # Execute
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        if ret != 0:
            raise RuntimeError(f"Model execute failed: {ret}")

        # Synchronize
        acl.rt.synchronize_device()

        # Device -> Host (2 = ACL_MEMCPY_DEVICE_TO_HOST)
        ret = acl.rt.memcpy(
            self._output_buf.ctypes.data, self.output_size,
            self.output_buffer, self.output_size,
            2
        )
        if ret != 0:
            raise RuntimeError(f"D2H memcpy failed: {ret}")

        # Postprocess
        output = self._output_buf.reshape(84, 8400).T
        boxes, scores, labels = self._postprocess(output, scale, new_w, new_h, orig_w, orig_h,
                                                   conf_threshold, iou_threshold)
        return boxes, scores, labels

    def _postprocess(self, output, scale, new_w, new_h, orig_w, orig_h,
                     conf_threshold, iou_threshold):
        """Vectorized YOLOv8 postprocessing with NMS."""
        # Filter by confidence (vectorized)
        class_scores = output[:, 4:]
        max_scores = class_scores.max(axis=1)
        mask = max_scores >= conf_threshold

        if not mask.any():
            return np.array([]), np.array([]), np.array([])

        filtered = output[mask]
        filtered_scores = max_scores[mask]
        filtered_labels = class_scores[mask].argmax(axis=1)

        # Convert center-wh to x1y1x2y2
        cx, cy, w_box, h_box = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
        x1 = cx - w_box / 2
        y1 = cy - h_box / 2
        x2 = cx + w_box / 2
        y2 = cy + h_box / 2

        # Scale back to original image coordinates
        x1 = (x1 / scale).clip(0, orig_w)
        y1 = (y1 / scale).clip(0, orig_h)
        x2 = (x2 / scale).clip(0, orig_w)
        y2 = (y2 / scale).clip(0, orig_h)

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), filtered_scores.tolist(),
                                   conf_threshold, iou_threshold)
        if len(indices) > 0:
            indices = indices.flatten()
            return boxes[indices], filtered_scores[indices], filtered_labels[indices]

        return np.array([]), np.array([]), np.array([])

    def cleanup(self):
        """Release all ACL resources. Safe to call multiple times."""
        if self._cleaned_up:
            return
        self._cleaned_up = True
        print("[ACL] Cleaning up resources...")
        try:
            if getattr(self, 'model_desc', None) is not None:
                acl.mdl.destroy_desc(self.model_desc)
                self.model_desc = None
            if self.model_id is not None:
                acl.mdl.unload(self.model_id)
                self.model_id = None
            if self.input_buffer is not None:
                acl.rt.free(self.input_buffer)
                self.input_buffer = None
            if self.output_buffer is not None:
                acl.rt.free(self.output_buffer)
                self.output_buffer = None
            if self.stream is not None:
                acl.rt.destroy_stream(self.stream)
                self.stream = None
            if self.context is not None:
                acl.rt.destroy_context(self.context)
                self.context = None
            acl.rt.reset_device(0)
            acl.finalize()
            print("[ACL] Cleanup complete")
        except Exception as e:
            print(f"[ACL] Cleanup error (non-fatal): {e}")

    def __del__(self):
        self.cleanup()


# ==============================================================================
# Torch NPU Engine (middle fallback)
# ==============================================================================
class TorchNPUEngine:
    """Inference using ultralytics + torch_npu on NPU.

    Used when ACL fails but NPU device is still accessible.
    Requires torch_npu 2.1.0.post10+ (compatible with CANN 8.0.0).
    """

    def __init__(self, model_path):
        import torch
        import torch_npu  # noqa: F401
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.device = 'npu:0' if torch.npu.is_available() else 'cpu'
        self.model.to(self.device)
        self._use_half = (self.device != 'cpu')
        if self._use_half:
            self.model.model.half()
        print(f"[TorchNPU] Model loaded on {self.device} (half={self._use_half})")

        # Warmup: first inference compiles the graph
        print("[TorchNPU] Running warmup inference...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.infer(dummy)
        print("[TorchNPU] Warmup complete")

    def infer(self, img, conf_threshold=0.25, iou_threshold=0.45):
        """Run inference, returns (boxes, scores, labels)."""
        results = self.model.predict(
            img, device=self.device, verbose=False,
            conf=conf_threshold, iou=iou_threshold,
            half=self._use_half, workers=0
        )
        result = results[0]
        boxes_obj = result.boxes

        if boxes_obj is not None and len(boxes_obj) > 0:
            boxes = boxes_obj.xyxy.cpu().numpy()
            scores = boxes_obj.conf.cpu().numpy()
            labels = boxes_obj.cls.cpu().numpy().astype(int)
            return boxes, scores, labels
        return np.array([]), np.array([]), np.array([])

    def cleanup(self):
        """Release resources."""
        import torch
        del self.model
        torch.npu.empty_cache()
        print("[TorchNPU] Cleanup complete")


# ==============================================================================
# CPU Fallback Engine
# ==============================================================================
class CPUFallbackEngine:
    """Fallback inference using ultralytics on CPU.

    Used when ACL/NPU is unavailable (e.g., stale device state after crash).
    Slower but guaranteed to work.
    """

    def __init__(self, model_path):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.device = 'cpu'
        print(f"[CPU] Model loaded on CPU: {model_path}")

    def infer(self, img, conf_threshold=0.25, iou_threshold=0.45):
        """Run inference on CPU, returns (boxes, scores, labels)."""
        results = self.model.predict(
            img, device='cpu', verbose=False,
            conf=conf_threshold, iou=iou_threshold, workers=0
        )
        result = results[0]
        boxes_obj = result.boxes

        if boxes_obj is not None and len(boxes_obj) > 0:
            boxes = boxes_obj.xyxy.cpu().numpy()
            scores = boxes_obj.conf.cpu().numpy()
            labels = boxes_obj.cls.cpu().numpy().astype(int)
            return boxes, scores, labels
        return np.array([]), np.array([]), np.array([])

    def cleanup(self):
        """Release resources."""
        del self.model
        print("[CPU] Cleanup complete")


# ==============================================================================
# NPU Monitor
# ==============================================================================
class NPUMonitor:
    """Polls npu-smi for temperature, power, and health status."""

    def __init__(self):
        self.npu_temp = 0
        self.npu_power = 0.0
        self.aicore_usage = 0
        self.memory_usage = "0/0"
        self.health = "N/A"
        self._lock = threading.Lock()

    @property
    def is_healthy(self):
        """Check if NPU is safe for inference (thermal-based)."""
        with self._lock:
            return self.npu_temp < THERMAL_PAUSE_THRESHOLD

    def update(self):
        try:
            result = subprocess.run(['npu-smi', 'info'], capture_output=True,
                                    text=True, timeout=5)
            temp = 0
            power = 0.0
            health = "N/A"
            for line in result.stdout.split('\n'):
                if '310B4' in line:
                    parts = line.split()
                    # Parse: | 0  310B4  | Health | Power  Temp  Hugepages |
                    for i, p in enumerate(parts):
                        try:
                            if p in ('OK', 'Alarm', 'Normal'):
                                health = p
                        except:
                            pass
                    # Find numeric values after 310B4
                    nums = [p for p in parts if p.replace('.', '').isdigit()]
                    if len(nums) >= 2:
                        power = float(nums[0])
                        temp = int(nums[1])

            # Try dedicated temp command for accuracy
            try:
                temp_result = subprocess.run(['npu-smi', 'info', '-t', 'temp', '-i', '0'],
                                            capture_output=True, text=True, timeout=5)
                for line in temp_result.stdout.split('\n'):
                    if 'Temperature' in line:
                        temp = int(line.split(':')[1].strip().replace('C', '').strip())
            except:
                pass

            # Try usages
            aicore = 0
            mem_usage = "0/0"
            try:
                usages = subprocess.run(['npu-smi', 'info', '-t', 'usages', '-i', '0'],
                                        capture_output=True, text=True, timeout=5)
                for line in usages.stdout.split('\n'):
                    if 'Aicore' in line:
                        aicore = int(line.split(':')[1].strip().replace('%', ''))
                    elif 'Memory' in line and 'Capacity' not in line:
                        mem_usage = line.split(':')[1].strip()
            except:
                pass

            with self._lock:
                self.npu_temp = temp
                self.npu_power = power
                self.health = health
                self.aicore_usage = aicore
                self.memory_usage = mem_usage

        except Exception:
            pass


# ==============================================================================
# GUI Application
# ==============================================================================
class YOLODetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Orange Pi AI Pro - YOLO Detection (NPU ACL)")
        self.root.geometry("1400x980")
        self.root.configure(bg='#1a1a2e')

        # State
        self.video_path = None
        self.cap = None
        self._cap_lock = threading.Lock()
        self.is_playing = False
        self.is_paused = False
        self.detection_thread = None
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.fps_video = 30
        self.total_frames = 0

        # Stats (bounded)
        self.frame_count = 0
        self.total_detections = 0
        self.detection_times = deque(maxlen=DETECTION_TIMES_MAXLEN)
        self.processing_start_time = 0

        # NPU
        self.npu_monitor = NPUMonitor()
        self.engine = None
        self._engines_cache = {}
        self._engine_name = "none"
        self._available_engines = []
        self._engine_switch_lock = threading.Lock()

        self.npu_status_var = tk.StringVar(value="Loading model...")
        self._engine_ready = False
        self._setup_ui()
        # Load engine in background (torch_npu first-time compile can take minutes)
        threading.Thread(target=self._load_engine, daemon=True).start()
        self._start_npu_monitor()
        # Auto-load first sample video so user can click Play immediately
        self.root.after(500, self._load_sample)
        # Auto-play when both engine and video are ready
        self._schedule_autoplay()

    def _schedule_autoplay(self):
        """Poll until engine and video are both ready, then auto-play."""
        if self._engine_ready and self.cap is not None and not self.is_playing:
            self._play()
        elif not self.is_playing:
            self.root.after(2000, self._schedule_autoplay)

    def _load_engine(self):
        """Load inference engines: ACL (fastest) -> torch_npu (if ACL fails) -> CPU.

        Populates the engine cache and available engines list for the toggle UI.
        """
        pt_model = os.path.join(DEMO_DIR, "yolov8n.pt")
        available = []

        # 1. Try ACL direct (pre-compiled .om model, fastest)
        try:
            self.root.after(0, lambda: self.engine_status_label.config(
                text="Trying ACL...", fg='#ffaa00'))
            engine = AclInferenceEngine(OM_MODEL)
            self._engines_cache['ACL (NPU .om)'] = engine
            available.append('ACL (NPU .om)')
            print("[Engine] ACL engine loaded successfully")
        except Exception as e:
            print(f"[Engine] ACL unavailable: {e}")

            # 2. Try torch_npu only if ACL failed (they conflict)
            try:
                self.root.after(0, lambda: self.engine_status_label.config(
                    text="Trying torch_npu...", fg='#ffaa00'))
                engine = TorchNPUEngine(pt_model)
                self._engines_cache['torch_npu (NPU)'] = engine
                available.append('torch_npu (NPU)')
                print(f"[Engine] torch_npu loaded on {engine.device}")
            except Exception as e2:
                print(f"[Engine] torch_npu unavailable: {e2}")

        # 3. CPU is always available (lazy-loaded on first use)
        available.append('CPU (fallback)')
        self._available_engines = available

        # Activate best available engine
        if available:
            best = available[0]
            self._activate_engine(best)

        # Update UI on main thread
        self.root.after(0, self._update_engine_ui)
        self._engine_ready = True

    def _setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')

        # Header
        header = tk.Frame(self.root, bg='#1a1a2e')
        header.pack(fill=tk.X, padx=15, pady=(10, 5))
        tk.Label(header, text=" Orange Pi AI Pro - YOLO Detection (NPU ACL)",
                 font=('Arial', 18, 'bold'), fg='#00ff88', bg='#1a1a2e',
                 anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(header, textvariable=self.npu_status_var, font=('Arial', 10),
                 fg='#00d4ff', bg='#1a1a2e').pack(side=tk.RIGHT, padx=10)

        # Main layout
        main = tk.Frame(self.root, bg='#1a1a2e')
        main.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        # Left panel
        left_panel = tk.Frame(main, bg='#1a1a2e', width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        self._create_npu_card(left_panel)
        self._create_engine_card(left_panel)
        self._create_video_card(left_panel)
        self._create_settings_card(left_panel)
        self._create_playback_card(left_panel)
        self._create_stats_card(left_panel)

        # Right panel (video canvas)
        right_panel = tk.Frame(main, bg='#000000')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(right_panel, bg='#000000', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Status bar
        status_bar = tk.Frame(self.root, bg='#0a0a1a', height=30)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_text = tk.StringVar(value="Loading...")
        tk.Label(status_bar, textvariable=self.status_text, fg='#888', bg='#0a0a1a',
                 font=('Arial', 9), anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.fps_display = tk.StringVar(value="")
        tk.Label(status_bar, textvariable=self.fps_display, fg='#00ff88', bg='#0a0a1a',
                 font=('Arial', 9, 'bold'), anchor=tk.E).pack(side=tk.RIGHT, padx=10)

    def _create_card(self, parent, title, y, height):
        card = tk.Frame(parent, bg='#2d2d44', relief=tk.RAISED, bd=1)
        card.place(x=0, y=y, width=340, height=height)
        tk.Label(card, text=title, font=('Arial', 11, 'bold'), fg='#00d4ff',
                 bg='#2d2d44', anchor=tk.W).pack(fill=tk.X, padx=10, pady=(8, 5))
        return card

    def _create_npu_card(self, parent):
        card = self._create_card(parent, " NPU Status", 0, 180)
        content = tk.Frame(card, bg='#2d2d44')
        content.pack(fill=tk.BOTH, padx=10, pady=5)

        metrics = [
            ("NPU Name:", "Ascend 310B4"),
            ("Health:", "Checking..."),
            ("Temperature:", "-- C"),
            ("Power:", "-- W"),
            ("AI Core:", "-- %"),
            ("Memory:", "-- / --"),
        ]
        self.npu_labels = {}
        for row, (label, default) in enumerate(metrics):
            tk.Label(content, text=label, fg='#aaa', bg='#2d2d44',
                     font=('Arial', 9)).grid(row=row, column=0, sticky=tk.W, pady=2)
            val = tk.Label(content, text=default, fg='#00ff88', bg='#2d2d44',
                           font=('Arial', 9, 'bold'))
            val.grid(row=row, column=1, sticky=tk.E, pady=2, padx=(10, 0))
            self.npu_labels[label] = val

    def _create_engine_card(self, parent):
        card = self._create_card(parent, " Engine", 185, 80)
        content = tk.Frame(card, bg='#2d2d44')
        content.pack(fill=tk.BOTH, padx=10, pady=5)

        selector_frame = tk.Frame(content, bg='#2d2d44')
        selector_frame.pack(fill=tk.X, pady=3)

        tk.Label(selector_frame, text="Backend:", fg='#aaa', bg='#2d2d44',
                 font=('Arial', 9)).pack(side=tk.LEFT)

        self.engine_var = tk.StringVar(value="(loading...)")
        self.engine_combo = ttk.Combobox(
            selector_frame, textvariable=self.engine_var,
            values=[], state='disabled', font=('Arial', 9), width=18
        )
        self.engine_combo.pack(side=tk.LEFT, padx=(8, 0), fill=tk.X, expand=True)
        self.engine_combo.bind("<<ComboboxSelected>>", self._on_engine_selected)

        self.engine_status_label = tk.Label(
            content, text="Loading...", fg='#ffaa00', bg='#2d2d44',
            font=('Arial', 8, 'italic'))
        self.engine_status_label.pack(anchor=tk.W)

    def _create_video_card(self, parent):
        card = self._create_card(parent, " Video Source", 270, 130)
        content = tk.Frame(card, bg='#2d2d44')
        content.pack(fill=tk.BOTH, padx=10, pady=5)

        tk.Label(content, text="Sample Videos:", fg='#aaa', bg='#2d2d44',
                 font=('Arial', 9)).pack(anchor=tk.W)

        self.sample_var = tk.StringVar()
        samples = self._get_sample_videos()
        if samples:
            self.sample_combo = ttk.Combobox(content, textvariable=self.sample_var,
                                             values=samples, state='readonly', font=('Arial', 8))
            self.sample_combo.set(samples[0] if samples else "")
            self.sample_combo.pack(fill=tk.X, pady=5)

        btn_frame = tk.Frame(content, bg='#2d2d44')
        btn_frame.pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="Load Sample", command=self._load_sample,
                  bg='#0a3d62', fg='white', font=('Arial', 9), relief=tk.FLAT
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(btn_frame, text="Open File...", command=self._open_video,
                  bg='#0a3d62', fg='white', font=('Arial', 9), relief=tk.FLAT
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.video_info = tk.StringVar(value="No video loaded")
        tk.Label(content, textvariable=self.video_info, fg='#aaa', bg='#2d2d44',
                 font=('Arial', 8), wraplength=300, justify=tk.LEFT).pack(anchor=tk.W)

    def _create_settings_card(self, parent):
        card = self._create_card(parent, " Detection Settings", 405, 100)
        content = tk.Frame(card, bg='#2d2d44')
        content.pack(fill=tk.BOTH, padx=10, pady=5)

        # Confidence
        conf_frame = tk.Frame(content, bg='#2d2d44')
        conf_frame.pack(fill=tk.X, pady=3)
        tk.Label(conf_frame, text="Confidence:", fg='#aaa', bg='#2d2d44',
                 width=10, anchor=tk.W).pack(side=tk.LEFT)
        self.conf_var = tk.DoubleVar(value=0.25)
        ttk.Scale(conf_frame, from_=0.1, to=0.9, variable=self.conf_var,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.conf_label = tk.Label(conf_frame, text="0.25", fg='#00ff88',
                                   bg='#2d2d44', width=4, font=('Arial', 9, 'bold'))
        self.conf_label.pack(side=tk.RIGHT)
        self.conf_var.trace_add('write', lambda *a: self.conf_label.config(
            text=f"{self.conf_var.get():.2f}"))

        # IOU
        iou_frame = tk.Frame(content, bg='#2d2d44')
        iou_frame.pack(fill=tk.X, pady=3)
        tk.Label(iou_frame, text="IOU:", fg='#aaa', bg='#2d2d44',
                 width=10, anchor=tk.W).pack(side=tk.LEFT)
        self.iou_var = tk.DoubleVar(value=0.45)
        ttk.Scale(iou_frame, from_=0.1, to=0.9, variable=self.iou_var,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.iou_label = tk.Label(iou_frame, text="0.45", fg='#00ff88',
                                  bg='#2d2d44', width=4, font=('Arial', 9, 'bold'))
        self.iou_label.pack(side=tk.RIGHT)
        self.iou_var.trace_add('write', lambda *a: self.iou_label.config(
            text=f"{self.iou_var.get():.2f}"))

    def _create_playback_card(self, parent):
        card = self._create_card(parent, " Playback", 510, 80)
        content = tk.Frame(card, bg='#2d2d44')
        content.pack(fill=tk.BOTH, padx=10, pady=5)

        btn_frame = tk.Frame(content, bg='#2d2d44')
        btn_frame.pack(fill=tk.X, pady=5)

        self.play_btn = tk.Button(btn_frame, text=" Play", command=self._play,
                                  bg='#00aa55', fg='white', font=('Arial', 10, 'bold'),
                                  relief=tk.FLAT, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.pause_btn = tk.Button(btn_frame, text=" Pause", command=self._pause,
                                   bg='#cc8800', fg='white', font=('Arial', 10, 'bold'),
                                   relief=tk.FLAT, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.stop_btn = tk.Button(btn_frame, text=" Stop", command=self._stop,
                                  bg='#cc3333', fg='white', font=('Arial', 10, 'bold'),
                                  relief=tk.FLAT, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

    def _create_stats_card(self, parent):
        card = self._create_card(parent, " Performance", 595, 180)
        content = tk.Frame(card, bg='#2d2d44')
        content.pack(fill=tk.BOTH, padx=10, pady=5)

        stats = [
            ("Frames:", "0"),
            ("Detection FPS:", "--"),
            ("Avg Latency:", "-- ms"),
            ("Total Objects:", "0"),
            ("Elapsed:", "0.0s"),
            ("Progress:", "0%"),
            ("FPS Cap:", f"{MAX_FPS}"),
        ]
        self.stats_labels = {}
        for label, default in stats:
            row_frame = tk.Frame(content, bg='#2d2d44')
            row_frame.pack(fill=tk.X, pady=2)
            tk.Label(row_frame, text=label, fg='#aaa', bg='#2d2d44',
                     font=('Arial', 9)).pack(side=tk.LEFT)
            val = tk.Label(row_frame, text=default, fg='#00d4ff', bg='#2d2d44',
                           font=('Arial', 9, 'bold'))
            val.pack(side=tk.RIGHT)
            self.stats_labels[label] = val

    # --- NPU Monitor ---
    def _start_npu_monitor(self):
        def loop():
            while not _shutdown_requested:
                self.npu_monitor.update()
                self.root.after(0, self._update_npu_display)
                time.sleep(3)
        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def _update_npu_display(self):
        m = self.npu_monitor
        with m._lock:
            health = m.health
            temp = m.npu_temp
            power = m.npu_power
            aicore = m.aicore_usage
            mem = m.memory_usage

        self.npu_labels["Health:"].config(
            text=health, fg='#ff4444' if health == 'Alarm' else '#00ff88')
        self.npu_labels["Temperature:"].config(
            text=f"{temp} C", fg='#ff4444' if temp >= THERMAL_PAUSE_THRESHOLD else '#00ff88')
        self.npu_labels["Power:"].config(text=f"{power:.1f} W")
        self.npu_labels["AI Core:"].config(text=f"{aicore} %")
        self.npu_labels["Memory:"].config(text=mem)

    # --- Engine Management ---
    def _activate_engine(self, engine_name):
        """Activate a specific engine by name. Returns True on success."""
        if engine_name == self._engine_name:
            return True

        pt_model = os.path.join(DEMO_DIR, "yolov8n.pt")

        if engine_name in self._engines_cache:
            self.engine = self._engines_cache[engine_name]
        elif engine_name == 'CPU (fallback)':
            try:
                engine = CPUFallbackEngine(pt_model)
                self._engines_cache[engine_name] = engine
                self.engine = engine
            except Exception as e:
                print(f"[Engine] CPU creation failed: {e}")
                return False
        else:
            return False

        self._engine_name = engine_name
        print(f"[Engine] Activated: {engine_name}")
        return True

    def _on_engine_selected(self, event=None):
        """Handle user selecting a different engine from the dropdown."""
        selected = self.engine_var.get()
        if selected == self._engine_name:
            return

        def do_switch():
            with self._engine_switch_lock:
                was_playing = self.is_playing
                if was_playing:
                    self.root.after(0, self._stop)
                    time.sleep(0.5)

                self.root.after(0, lambda: self.engine_status_label.config(
                    text=f"Switching to {selected}...", fg='#ffaa00'))

                success = self._activate_engine(selected)

                if success:
                    self.root.after(0, lambda: self.engine_status_label.config(
                        text=f"Active: {selected}", fg='#00ff88'))
                    self.root.after(0, lambda: self.npu_status_var.set(f"Engine: {selected}"))
                    self.root.after(0, lambda: self.status_text.set(f"Ready - {selected}"))
                    if was_playing:
                        self.root.after(500, self._play)
                else:
                    self.root.after(0, lambda: self.engine_var.set(self._engine_name))
                    self.root.after(0, lambda: self.engine_status_label.config(
                        text=f"Failed! Staying on {self._engine_name}", fg='#ff4444'))

        threading.Thread(target=do_switch, daemon=True).start()

    def _update_engine_ui(self):
        """Update engine combo box with available engines (main thread)."""
        self.engine_combo.config(values=self._available_engines, state='readonly')
        self.engine_var.set(self._engine_name)
        self.engine_status_label.config(text=f"Active: {self._engine_name}", fg='#00ff88')
        self.npu_status_var.set(f"Engine: {self._engine_name}")
        self.status_text.set(f"Ready - {self._engine_name}")

    # --- Video ---
    def _get_sample_videos(self):
        if not os.path.exists(VIDEO_DIR):
            return []
        return sorted([os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR)
                       if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))])

    def _open_video(self):
        path = filedialog.askopenfilename(title="Select Video",
                                          filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self._load_video(path)

    def _load_sample(self):
        selected = self.sample_var.get()
        if selected and os.path.exists(selected):
            self._load_video(selected)

    def _load_video(self, path):
        self._stop()
        with self._cap_lock:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Cannot open: {path}")
                self.cap = None
                return
            self.video_path = path
            self.fps_video = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vid_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = self.total_frames / self.fps_video if self.fps_video > 0 else 0
        self.video_info.set(f"{os.path.basename(path)}\n"
                            f"{vid_w}x{vid_h} @ {self.fps_video:.1f}fps\n"
                            f"Duration: {duration:.1f}s ({self.total_frames} frames)")
        self.play_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self._reset_stats()
        self._show_first_frame()

    def _show_first_frame(self):
        with self._cap_lock:
            if not self.cap:
                return
            ret, frame = self.cap.read()
            if ret:
                self._display_frame(frame)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _display_frame(self, frame):
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return

        h, w = frame.shape[:2]
        scale = min(canvas_w / w, canvas_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        self.canvas.delete("all")
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk  # prevent GC

    # --- Playback ---
    def _play(self):
        if not self.engine:
            self.status_text.set("Model still loading, please wait...")
            return
        with self._cap_lock:
            if not self.cap:
                self.status_text.set("Load a video first (click 'Load Sample')")
                return

        if self.is_paused:
            self.is_paused = False
            return

        # Reset if at end
        if self.frame_count >= self.total_frames - 1:
            with self._cap_lock:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._reset_stats()

        self.is_playing = True
        self.is_paused = False
        self.play_btn.config(state=tk.DISABLED)
        self.status_text.set("NPU Detection running...")
        self.processing_start_time = time.time()
        # Cache settings for thread safety (Tk vars can't be read from threads reliably)
        self._conf_threshold = self.conf_var.get()
        self._iou_threshold = self.iou_var.get()

        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.start()

    def _pause(self):
        self.is_paused = not self.is_paused
        self.status_text.set("Paused" if self.is_paused else "NPU Detection running...")

    def _stop(self):
        self.is_playing = False
        self.is_paused = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=3)
        self.detection_thread = None
        self.play_btn.config(state=tk.NORMAL)
        self.status_text.set("Stopped")

    def _reset_stats(self):
        self.frame_count = 0
        self.total_detections = 0
        self.detection_times.clear()

    # --- Detection Loop ---
    def _detection_loop(self):
        global _shutdown_requested
        consecutive_failures = 0
        min_interval = 1.0 / MAX_FPS if MAX_FPS > 0 else 0
        display_counter = 0

        while self.is_playing and not _shutdown_requested:
            if self.is_paused:
                time.sleep(0.1)
                continue

            # Thermal watchdog
            if not self.npu_monitor.is_healthy:
                self.root.after(0, lambda: self.status_text.set(
                    f"[THERMAL PAUSE] NPU {self.npu_monitor.npu_temp}C - cooling..."))
                while (not self.npu_monitor.is_healthy and
                       self.is_playing and not _shutdown_requested):
                    time.sleep(2)
                if not self.is_playing:
                    break
                while (self.npu_monitor.npu_temp > THERMAL_RESUME_THRESHOLD and
                       self.is_playing and not _shutdown_requested):
                    time.sleep(1)
                self.root.after(0, lambda: self.status_text.set("Detection running..."))
                continue

            frame_start = time.time()

            # Read frame (thread-safe)
            with self._cap_lock:
                if self.cap is None:
                    break
                ret, frame = self.cap.read()

            if not ret:
                self.root.after(0, self._stop)
                break

            # Inference with failure tracking
            try:
                boxes, scores, labels = self.engine.infer(
                    frame, self._conf_threshold, self._iou_threshold)
                consecutive_failures = 0

                # Draw detections
                if len(boxes) > 0:
                    self.total_detections += len(boxes)
                    for box, score, label_idx in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = box.astype(int)
                        color = tuple(int(c) for c in COLORS[int(label_idx)])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        text = f"{COCO_CLASSES[int(label_idx)]} {score:.2f}"
                        cv2.putText(frame, text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            except Exception as e:
                consecutive_failures += 1
                print(f"[Watchdog] Inference error ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    self.root.after(0, lambda: messagebox.showwarning(
                        "NPU Error",
                        f"NPU inference failed {MAX_CONSECUTIVE_FAILURES}x consecutively.\n"
                        f"Last error: {e}\n\nDetection paused."))
                    self.root.after(0, self._stop)
                    break
                time.sleep(0.5)
                continue

            det_time = time.time() - frame_start
            self.detection_times.append(det_time)
            self.frame_count += 1
            display_counter += 1

            # Frame skipping: if inference is slower than video FPS, seek ahead
            if self.fps_video > 0:
                frames_behind = int(det_time * self.fps_video) - 1
                if frames_behind > 0:
                    with self._cap_lock:
                        if self.cap is not None:
                            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                            new_pos = min(current_pos + frames_behind, self.total_frames - 1)
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                            self.frame_count += frames_behind

            # Display throttling: skip alternate canvas updates at high FPS
            should_display = True
            if len(self.detection_times) >= 5:
                recent_avg = sum(list(self.detection_times)[-5:]) / 5
                recent_fps = 1.0 / recent_avg if recent_avg > 0 else 0
                if recent_fps > DISPLAY_SKIP_THRESHOLD_FPS:
                    should_display = (display_counter % 2 == 0)

            if should_display:
                self.root.after(0, self._display_frame, frame.copy())

            self.root.after(0, self._update_stats_display)

            # Frame throttle to MAX_FPS
            elapsed = time.time() - frame_start
            sleep_time = min_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.is_playing = False
        self.root.after(0, lambda: self.play_btn.config(state=tk.NORMAL))

    def _update_stats_display(self):
        elapsed = time.time() - self.processing_start_time if self.processing_start_time else 0
        if self.detection_times:
            avg_det = sum(self.detection_times) / len(self.detection_times)
            det_fps = 1.0 / avg_det if avg_det > 0 else 0
            avg_latency = avg_det * 1000
        else:
            det_fps = 0
            avg_latency = 0
        progress = (self.frame_count / self.total_frames * 100) if self.total_frames > 0 else 0

        self.stats_labels["Frames:"].config(text=str(self.frame_count))
        self.stats_labels["Detection FPS:"].config(text=f"{det_fps:.1f}")
        self.stats_labels["Avg Latency:"].config(text=f"{avg_latency:.1f} ms")
        self.stats_labels["Total Objects:"].config(text=str(self.total_detections))
        self.stats_labels["Elapsed:"].config(text=f"{elapsed:.1f}s")
        self.stats_labels["Progress:"].config(text=f"{progress:.1f}%")

        self.fps_display.set(f"Detection: {det_fps:.1f} FPS | Frame {self.frame_count}/{self.total_frames}")

    # --- Shutdown ---
    def shutdown(self):
        """Clean shutdown: stop threads, release all engines, release video."""
        global _shutdown_requested
        _shutdown_requested = True
        self.is_playing = False

        # Join detection thread
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=3)

        # Release video
        with self._cap_lock:
            if self.cap:
                self.cap.release()
                self.cap = None

        # Release all cached engines
        for name, eng in self._engines_cache.items():
            try:
                eng.cleanup()
                print(f"[App] Cleaned up engine: {name}")
            except Exception as e:
                print(f"[App] Cleanup error for {name}: {e}")
        self._engines_cache.clear()
        self.engine = None

        print("[App] Shutdown complete")


# ==============================================================================
# Main
# ==============================================================================
def main():
    root = tk.Tk()
    app = YOLODetectionApp(root)

    def on_closing():
        app.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Also handle shutdown flag from signal handler
    def check_shutdown():
        if _shutdown_requested:
            on_closing()
        else:
            root.after(500, check_shutdown)

    root.after(500, check_shutdown)
    root.mainloop()


if __name__ == "__main__":
    main()
