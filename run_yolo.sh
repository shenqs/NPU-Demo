#!/bin/bash
# YOLO Detection Launcher with tmux protection
# If X11 crashes, reconnect with: tmux attach -t yolo
#
# Usage: bash /home/HwHiAiUser/npu_demo/run_yolo.sh [npu|om|app]

DEMO_DIR="/home/HwHiAiUser/npu_demo"
SESSION_NAME="yolo"
PYTHON="/usr/local/miniconda3/bin/python3"

# Default to the NPU version
SCRIPT="${1:-npu}"
case "$SCRIPT" in
    npu)  TARGET="$DEMO_DIR/yolo_detection_app_npu.py" ;;
    om)   TARGET="$DEMO_DIR/yolo_detection_app_om.py" ;;
    app)  TARGET="$DEMO_DIR/yolo_detection_app.py" ;;
    *)    TARGET="$SCRIPT" ;;
esac

if [ ! -f "$TARGET" ]; then
    echo "Error: Script not found: $TARGET"
    exit 1
fi

# Check if tmux is available
if ! command -v tmux &>/dev/null; then
    echo "tmux not installed. Install with: sudo apt install tmux"
    echo "Running without tmux protection..."
    exec $PYTHON "$TARGET"
fi

# Check NPU health before starting
echo "Checking NPU health..."
NPU_HEALTH=$(npu-smi info 2>/dev/null | grep "310B4" | awk '{print $6}')
NPU_TEMP=$(npu-smi info -t temp -i 0 2>/dev/null | grep "Temperature" | awk -F: '{print $2}' | tr -d ' ')
if [ "$NPU_HEALTH" = "Alarm" ]; then
    echo "NOTE: NPU shows Alarm (known current-sensor issue on 310B4, non-fatal)"
    echo "  Temperature: ${NPU_TEMP:-unknown}C"
    if [ "${NPU_TEMP:-0}" -ge 80 ]; then
        echo "WARNING: NPU temperature is HIGH! Let it cool down first."
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Kill existing session if any
tmux has-session -t "$SESSION_NAME" 2>/dev/null && tmux kill-session -t "$SESSION_NAME"

# Launch in tmux
echo "Starting YOLO detection in tmux session '$SESSION_NAME'..."
echo "  Script: $TARGET"
echo ""
echo "  If your terminal or X11 crashes, reconnect with:"
echo "    tmux attach -t $SESSION_NAME"
echo ""

# Set DISPLAY for GUI apps inside tmux
# Ascend NPU stability env vars:
#   ASCEND_GLOBAL_LOG_LEVEL=3  - ERROR only (reduce log I/O)
#   ASCEND_SLOG_PRINT_TO_STDOUT=0 - no stdout spam
#   ASCEND_DEVICE_ID=0 - explicit device binding
#   ASCEND_LAUNCH_BLOCKING=1 - serialize H2D/D2H transfers, reduce DDR contention
NPU_ENV="export DISPLAY=:0; \
export ASCEND_GLOBAL_LOG_LEVEL=3; \
export ASCEND_SLOG_PRINT_TO_STDOUT=0; \
export ASCEND_DEVICE_ID=0; \
export ASCEND_LAUNCH_BLOCKING=1; \
export TE_PARALLEL_COMPILER=0; \
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null"

tmux new-session -d -s "$SESSION_NAME" "$NPU_ENV; $PYTHON '$TARGET'; echo ''; echo 'Process exited. Press Enter to close.'; read"

# Attach to the session
tmux attach -t "$SESSION_NAME"
