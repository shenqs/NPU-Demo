#!/usr/bin/env python3
"""Orange Pi AI Pro - Desktop YOLO Detection (NPU .om model via ACL)."""
import os
import sys
import time
import threading
import subprocess
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

import acl

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
DEMO_DIR = "/home/HwHiAiUser/npu_demo"
VIDEO_DIR = f"{DEMO_DIR}/videos"
OUTPUT_DIR = f"{DEMO_DIR}/output_video"
OM_MODEL = os.path.join(DEMO_DIR, "yolov8n_npu.om")

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

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255), (255, 255, 128), (255, 128, 255),
]


# ──────────────────────────────────────────────────────────────
# YOLOv8 ACL NPU Engine
# ──────────────────────────────────────────────────────────────
class YOLOv8ACLEngine:
    def __init__(self, om_path):
        self.om_path = om_path
        self._conf = 0.25
        self._iou = 0.45
        self._lock = threading.Lock()
        print("  Initializing ACL runtime...")
        acl.init()
        acl.rt.set_device(0)
        self._load_model()
        print("  NPU .om model loaded via ACL!")

    def _load_model(self):
        self.model_id, ret = acl.mdl.load_from_file(self.om_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load .om model: {ret}")

        self.input_size = acl.mdl.get_input_size_by_index(self.model_id, 0)
        self.output_size = acl.mdl.get_output_size_by_index(self.model_id, 0)
        self.input_dims = acl.mdl.get_input_dims(self.model_id, 0)
        self.output_dims = acl.mdl.get_output_dims(self.model_id, 0)

        # Allocate device memory
        ret, self.input_buffer = acl.rt.malloc(self.input_size, acl.rt.mem.MEMORY_NORMAL)
        if ret != 0:
            raise RuntimeError(f"Failed to allocate input buffer: {ret}")

        ret, self.output_buffer = acl.rt.malloc(self.output_size, acl.rt.mem.MEMORY_NORMAL)
        if ret != 0:
            raise RuntimeError(f"Failed to allocate output buffer: {ret}")

        # Create input dataset
        self.input_dataset = acl.mdl.create_dataset()
        ret, input_tensor = acl.create_data_buffer(self.input_buffer, self.input_size)
        if ret != 0:
            raise RuntimeError(f"Failed to create input data buffer: {ret}")
        acl.mdl.add_dataset_buffer(self.input_dataset, input_tensor)

        # Create output dataset
        self.output_dataset = acl.mdl.create_dataset()
        ret, output_tensor = acl.create_data_buffer(self.output_buffer, self.output_size)
        if ret != 0:
            raise RuntimeError(f"Failed to create output data buffer: {ret}")
        acl.mdl.add_dataset_buffer(self.output_dataset, output_tensor)

        print(f"  Model: inputs={acl.mdl.get_num_inputs(self.model_id)}, outputs={acl.mdl.get_num_outputs(self.model_id)}")
        print(f"  Input dims: {self.input_dims}, Output dims: {self.output_dims}")

    def update_params(self, conf, iou, max_det=100):
        with self._lock:
            self._conf = conf
            self._iou = iou

    def _preprocess(self, img):
        h, w = img.shape[:2]
        scale = 640.0 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        padded = np.full((640, 640, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = img_resized
        img_norm = padded.astype(np.float32) / 255.0
        input_data = img_norm.transpose(2, 0, 1).flatten().astype(np.float32)
        return input_data, scale, (new_w, new_h)

    def _postprocess(self, output_data, conf_threshold, iou_threshold):
        output_data = output_data.reshape(84, 8400).T
        boxes, scores, labels = [], [], []
        for pred in output_data:
            class_scores = pred[4:]
            max_score = np.max(class_scores)
            if max_score < conf_threshold:
                continue
            max_class = np.argmax(class_scores)
            cx, cy, pw, ph = pred[0], pred[1], pred[2], pred[3]
            x1 = cx - pw / 2
            y1 = cy - ph / 2
            x2 = cx + pw / 2
            y2 = cy + ph / 2
            boxes.append([x1, y1, x2, y2])
            scores.append(max_score)
            labels.append(max_class)

        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        boxes = np.array(boxes)
        scores = np.array(scores)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)
        if len(indices) > 0:
            indices = indices.flatten()
            boxes = boxes[indices]
            scores = scores[indices]
            labels = np.array(labels)[indices]
        return boxes, scores, labels

    def _draw_boxes(self, img, boxes, scores, labels, scale):
        out = img.copy()
        h, w = out.shape[:2]
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [int(b / scale) for b in box]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            color = COLORS[int(label) % len(COLORS)]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            text = f"{COCO_CLASSES[int(label)]} {score:.2f}"
            t_w, t_h = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(out, (x1, y1 - t_h - 4), (x1 + t_w, y1), color, -1)
            cv2.putText(out, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1)
        return out

    def process_frame(self, frame):
        with self._lock:
            conf, iou = self._conf, self._iou

        input_data, scale, _ = self._preprocess(frame)

        # Copy host -> device
        acl.rt.memcpy(self.input_buffer, self.input_size,
                      input_data.ctypes.data, self.input_size,
                      acl.rt.memcpy.MEMCPY_HOST_TO_DEVICE)

        # Execute
        acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        acl.rt.synchronize_device()

        # Copy device -> host
        num_elements = self.output_size // 4
        output_data = np.zeros(num_elements, dtype=np.float32)
        acl.rt.memcpy(output_data.ctypes.data, self.output_size,
                      self.output_buffer, self.output_size,
                      acl.rt.memcpy.MEMCPY_DEVICE_TO_HOST)

        boxes, scores, labels = self._postprocess(output_data, conf, iou)
        annotated = self._draw_boxes(frame, boxes, scores, labels, scale)
        return annotated, len(boxes)

    def cleanup(self):
        try:
            acl.mdl.unload(self.model_id)
            acl.rt.free(self.input_buffer)
            acl.rt.free(self.output_buffer)
            acl.rt.reset_device(0)
            acl.finalize()
        except:
            pass


# ──────────────────────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────────────────────
class DetectionApp:
    def __init__(self, root, engine):
        self.root = root
        self.engine = engine
        self.cap = None
        self.playing = False
        self.fps_display = 0.0
        self.total_frames = 0
        self.current_frame = 0
        self.processed_count = 0

        self._setup_style()
        self._build_ui()
        self._populate_videos()
        self._bind_keys()

    def _setup_style(self):
        self.root.configure(bg="#1a1a2e")
        self.root.title("Orange Pi AI Pro - NPU Detection (ACL .om)")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1a1a2e")
        style.configure("TLabel", background="#1a1a2e", foreground="#e0e0e0", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("Primary.TButton", font=("Segoe UI", 11, "bold"), background="#00d4ff", foreground="#000")
        style.configure("Danger.TButton", font=("Segoe UI", 10), background="#ff4757", foreground="#fff")
        style.configure("TScale", background="#1a1a2e", troughcolor="#333", borderwidth=0)
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#00d4ff")
        style.configure("Info.TLabel", font=("Consolas", 9), foreground="#aaa")
        style.configure("Stat.TLabel", font=("Consolas", 11), foreground="#0f0")
        style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"), foreground="#fff")

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(fill=tk.X)
        ttk.Label(top, text="Orange Pi AI Pro  •  NPU Detection (ACL .om)",
                  style="Header.TLabel").pack(side=tk.LEFT)
        self.npu_label = ttk.Label(top, text="NPU: Ascend 310B4 | ACL Runtime",
                                   style="Info.TLabel", foreground="#00ff88")
        self.npu_label.pack(side=tk.RIGHT, padx=10)

        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        left = ttk.Frame(main, width=280)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        ttk.Label(left, text="Video Source", style="Title.TLabel").pack(anchor=tk.W, pady=(4, 2))
        self.video_var = tk.StringVar(value="Select a video to start")
        ttk.Label(left, textvariable=self.video_var, style="Info.TLabel",
                  wraplength=260).pack(anchor=tk.W)

        self.sample_var = tk.StringVar()
        self.sample_combo = ttk.Combobox(left, textvariable=self.sample_var, state="readonly",
                                          width=34, font=("Segoe UI", 9))
        self.sample_combo.pack(fill=tk.X, pady=(2, 4))
        self.sample_combo.bind("<<ComboboxSelected>>", self._on_sample_select)

        btn_row = ttk.Frame(left)
        btn_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(btn_row, text="Open File", command=self._open_file).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(btn_row, text="Refresh Samples", command=self._populate_videos).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._add_slider(left, "Confidence", 0.1, 0.9, 0.25, self._conf_changed)
        self._add_slider(left, "IOU Threshold", 0.1, 0.9, 0.45, self._iou_changed)
        self._add_slider(left, "Max Detections", 10, 300, 100, self._maxdet_changed)

        ctrl_row = ttk.Frame(left)
        ctrl_row.pack(fill=tk.X, pady=(12, 8))
        self.play_btn = ttk.Button(ctrl_row, text=" Play", style="Primary.TButton",
                                   command=self._toggle_play)
        self.play_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.stop_btn = ttk.Button(ctrl_row, text=" Stop", style="Danger.TButton",
                                   command=self._stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(left, text="Progress", style="Title.TLabel").pack(anchor=tk.W, pady=(8, 2))
        self.progress = ttk.Progressbar(left, orient=tk.HORIZONTAL, mode="determinate")
        self.progress.pack(fill=tk.X)
        self.progress_label = ttk.Label(left, text="0 / 0  frames", style="Info.TLabel")
        self.progress_label.pack(anchor=tk.W, pady=(2, 0))

        ttk.Label(left, text="Statistics", style="Title.TLabel").pack(anchor=tk.W, pady=(14, 2))
        self.stats_frame = ttk.Frame(left)
        self.stats_frame.pack(fill=tk.X)
        stats = [
            ("FPS:", "fps_val"),
            ("Objects:", "obj_val"),
            ("Elapsed:", "elapsed_val"),
            ("NPU Temp:", "npu_temp_val"),
            ("AI Core:", "npu_aicore_val"),
        ]
        for i, (label, attr) in enumerate(stats):
            ttk.Label(self.stats_frame, text=label, style="Title.TLabel").grid(row=i, column=0, sticky=tk.W, pady=2)
            setattr(self, attr, ttk.Label(self.stats_frame, text="0.0", style="Stat.TLabel"))
            getattr(self, attr).grid(row=i, column=1, sticky=tk.E, padx=(10, 0), pady=2)

        ttk.Label(left, text="Log", style="Title.TLabel").pack(anchor=tk.W, pady=(14, 2))
        self.log_text = tk.Text(left, height=8, bg="#111", fg="#0f0",
                                font=("Consolas", 8), relief=tk.FLAT,
                                highlightthickness=1, highlightbackground="#333")
        self.log_text.pack(fill=tk.X)
        self.log_text.config(state=tk.DISABLED)

        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(right, bg="#000", relief=tk.SUNKEN, bd=2)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.status_bar = ttk.Label(self.root, text="Ready", style="Info.TLabel",
                                    relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 5))

    def _add_slider(self, parent, label, from_, to, value, cmd):
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(f, text=label, style="TLabel").pack(anchor=tk.W)
        var = tk.DoubleVar(value=value)
        slider = ttk.Scale(f, from_=from_, to=to, variable=var, command=cmd)
        slider.pack(fill=tk.X)
        val_label = ttk.Label(f, text=f"{value}", style="Info.TLabel")
        val_label.pack(anchor=tk.E)
        slider.var = var
        slider.val_label = val_label
        return slider

    def _log(self, msg):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _status(self, msg):
        self.status_bar.config(text=msg)

    def _conf_changed(self, val):
        s = self.conf_slider
        s.val_label.config(text=f"{float(val):.2f}")
        self.engine.update_params(float(val), float(self.iou_slider.var.get()),
                                  int(self.maxdet_slider.var.get()))

    def _iou_changed(self, val):
        s = self.iou_slider
        s.val_label.config(text=f"{float(val):.2f}")
        self.engine.update_params(float(self.conf_slider.var.get()), float(val),
                                  int(self.maxdet_slider.var.get()))

    def _maxdet_changed(self, val):
        s = self.maxdet_slider
        s.val_label.config(text=f"{int(float(val))}")
        self.engine.update_params(float(self.conf_slider.var.get()),
                                  float(self.iou_slider.var.get()),
                                  int(float(val)))

    def _populate_videos(self):
        self._log("Refreshing sample videos...")
        samples = []
        if os.path.exists(VIDEO_DIR):
            for f in sorted(os.listdir(VIDEO_DIR)):
                if f.endswith((".mp4", ".avi", ".mov")):
                    samples.append(os.path.join(VIDEO_DIR, f))
        self.sample_combo["values"] = samples
        if samples:
            self.sample_combo.current(0)

    def _on_sample_select(self, event=None):
        path = self.sample_var.get()
        if path and os.path.exists(path):
            self._load_video(path)

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")],
            initialdir=VIDEO_DIR,
        )
        if path:
            self.sample_var.set(path)
            self._load_video(path)

    def _load_video(self, path):
        if self.playing:
            self._stop()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open: {path}")
            self.cap = None
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps_raw = self.cap.get(cv2.CAP_PROP_FPS)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_var.set(f"{os.path.basename(path)}  ({w}x{h}, {self.fps_raw:.1f} fps)")
        self.progress["maximum"] = self.total_frames
        self.progress["value"] = 0
        self._log(f"Loaded: {os.path.basename(path)}  |  {w}x{h}  |  {self.total_frames} frames")
        self._status("Video loaded - press Play")
        self.play_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def _toggle_play(self):
        if not self.playing:
            self._start()
        else:
            self._pause()

    def _start(self):
        if not self.cap or not self.cap.isOpened():
            return
        self.playing = True
        self.start_time = time.time()
        self.processed_count = 0
        self.play_btn.config(text=" Pause", state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._status("Processing on NPU...")
        self._log(" Processing started (ACL .om model)")
        threading.Thread(target=self._process_loop, daemon=True).start()
        self._schedule_ui_update()

    def _pause(self):
        self.playing = False
        self.play_btn.config(text=" Play", state=tk.NORMAL)
        self._status("Paused")
        self._log(" Paused")

    def _stop(self):
        self.playing = False
        self.play_btn.config(text=" Play", state=tk.NORMAL if self.cap else tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self._status("Stopped")
        self._log(" Stopped")

    def _process_loop(self):
        while self.playing and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.root.after(0, self._on_video_end)
                return
            annotated, num_det = self.engine.process_frame(frame)
            with threading.Lock():
                self._latest_frame = annotated
                self._latest_detections = num_det
            self.processed_count += 1
            self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def _schedule_ui_update(self):
        if self.playing:
            self._update_ui()
            self.root.after(10, self._schedule_ui_update)

    def _update_ui(self):
        if not hasattr(self, "_latest_frame") or self._latest_frame is None:
            return

        frame = self._latest_frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        area_w = self.video_label.winfo_width()
        area_h = self.video_label.winfo_height()
        if area_w < 10 or area_h < 10:
            area_w, area_h = 800, 600
        ratio = min(area_w / img.width, area_h / img.height)
        new_w, new_h = int(img.width * ratio), int(img.height * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.video_label.config(image=self._photo)

        elapsed = time.time() - self.start_time if hasattr(self, "start_time") else 0
        self.fps_display = self.processed_count / elapsed if elapsed > 0 else 0
        self.fps_val.config(text=f"{self.fps_display:.1f}")
        self.obj_val.config(text=f"{self._latest_detections}")
        self.elapsed_val.config(text=f"{elapsed:.1f}s")

        self.progress["value"] = self.current_frame
        self.progress_label.config(text=f"{self.current_frame} / {self.total_frames}  frames")

    def _on_video_end(self):
        self.playing = False
        elapsed = time.time() - self.start_time
        self._log(f" Done - {self.processed_count} frames in {elapsed:.1f}s  ({self.fps_display:.1f} fps avg)")
        self._log(f"  Output saved to: {OUTPUT_DIR}")
        self._save_output()
        self.play_btn.config(text=" Play", state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self._status("Finished")

    def _save_output(self):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "processed_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            annotated, _ = self.engine.process_frame(frame)
            out.write(annotated)
            count += 1
        out.release()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._log(f"  Saved: {out_path}  ({os.path.getsize(out_path)/1024/1024:.1f} MB)")

    def _bind_keys(self):
        self.root.bind("<space>", lambda e: self._toggle_play())
        self.root.bind("<Escape>", lambda e: self._stop())
        self.root.bind("o", lambda e: self._open_file())
        self.npu_temp_val.config(text="--")
        self.npu_aicore_val.config(text="--")
        threading.Thread(target=self._npu_monitor_loop, daemon=True).start()

    def _npu_monitor_loop(self):
        while True:
            temp, aicore = self._get_npu_info()
            self.root.after(0, lambda t=temp, a=aicore: self._update_npu_display(t, a))
            time.sleep(2)

    def _get_npu_info(self):
        temp = "--"
        aicore = "--"
        try:
            r = subprocess.run(
                ['npu-smi', 'info', '-t', 'usages', '-i', '0'],
                capture_output=True, text=True, timeout=3
            )
            for line in r.stdout.split('\n'):
                if 'Aicore' in line:
                    try:
                        aicore = line.split(':')[1].strip().replace('%', '')
                    except:
                        pass
        except:
            pass
        try:
            r = subprocess.run(
                ['npu-smi', 'info', '-t', 'temp', '-i', '0'],
                capture_output=True, text=True, timeout=3
            )
            for line in r.stdout.split('\n'):
                if 'Temperature' in line:
                    try:
                        temp = line.split(':')[1].strip().replace('C', '').strip()
                    except:
                        pass
        except:
            pass
        return temp, aicore

    def _update_npu_display(self, temp, aicore):
        self.npu_temp_val.config(text=f"{temp} C")
        self.npu_aicore_val.config(text=f"{aicore} %")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Orange Pi AI Pro - NPU YOLO Detection (ACL .om)")
    print("=" * 55)
    engine = YOLOv8ACLEngine(OM_MODEL)
    print("  Model ready!")

    root = tk.Tk()
    app = DetectionApp(root, engine)
    root.geometry("1200x750")
    root.minsize(900, 600)
    app._log("NPU ACL engine ready. Select a video and press Play.")
    try:
        root.mainloop()
    finally:
        engine.cleanup()


if __name__ == "__main__":
    main()
