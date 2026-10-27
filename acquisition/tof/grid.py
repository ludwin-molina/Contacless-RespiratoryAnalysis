"""Diagnostic view: standard deviation of depth in a coarse grid. Not a respiratory-rate method."""

import time
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, render_template_string

from acquisition.settings import tof_grid_port

from .lidar_driver import LidarDriver

# Cell size in pixels. Smaller cells resolve the chest, larger cells average noise.
GRID_SIZE = 20
# Frames kept for the per-cell standard deviation.
HISTORY_LEN = 40
# Standard deviation, in millimetres, that saturates the colour map.
# Quiet breathing is usually 5 to 15 mm.
MAX_STD_VIS = 20.0

driver = LidarDriver()
app = Flask(__name__)

grid_w = 320 // GRID_SIZE
grid_h = 240 // GRID_SIZE
history_buffer = deque(maxlen=HISTORY_LEN)


def calculate_grid_std_dev(depth_frame):
    """Standard deviation of each grid cell across the recent frames, in millimetres."""
    small_view = cv2.resize(depth_frame, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    small_view = small_view.astype(np.float32)
    history_buffer.append(small_view)
    if len(history_buffer) < HISTORY_LEN:
        return np.zeros((grid_h, grid_w), dtype=np.float32)
    stack = np.array(history_buffer)
    return np.std(stack, axis=0)


def generate_frames():
    """Yield JPEG frames with the depth image under a standard-deviation colour map."""
    driver.setup_camera()
    while True:
        depth = driver.get_frame()
        if depth is None:
            time.sleep(0.01)
            continue

        std_map = calculate_grid_std_dev(depth)
        vis_grid = np.clip(std_map / MAX_STD_VIS * 255, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(vis_grid, cv2.COLORMAP_JET)
        heatmap_big = cv2.resize(heatmap, (320, 240), interpolation=cv2.INTER_NEAREST)

        vis_preview = np.clip(depth / 16, 0, 255).astype(np.uint8)
        depth_vis = cv2.cvtColor(vis_preview, cv2.COLOR_GRAY2BGR)
        final_vis = cv2.addWeighted(depth_vis, 0.6, heatmap_big, 0.4, 0)

        for y in range(grid_h):
            for x in range(grid_w):
                px = x * GRID_SIZE
                py = y * GRID_SIZE
                cv2.rectangle(final_vis, (px, py), (px + GRID_SIZE, py + GRID_SIZE), (50, 50, 50), 1)
                value = std_map[y, x]
                # Skip labels on cells that are only background noise.
                if value > 2.0:
                    color = (255, 255, 255) if value < (MAX_STD_VIS / 1.5) else (0, 0, 0)
                    cv2.putText(final_vis, f"{value:.1f}", (px + 2, py + 12), cv2.FONT_HERSHEY_PLAIN, 0.7, color, 1)

        ok, buf = cv2.imencode(".jpg", final_vis)
        if ok:
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


@app.route("/")
def home():
    """Browser page for the variance grid."""
    return render_template_string(
        """
        <body style='background:#111; color:#fff; text-align:center; font-family:sans-serif;'>
            <h1>ToF deviation grid (standard deviation)</h1>
            <p>Each cell is the <b>standard deviation (mm)</b> over the last 40 frames.</p>
            <p>Blue means stable (noise under 5 mm). Red means motion (deviation over 20 mm).</p>
            <img src='/stream' style='border: 2px solid #555;'>
        </body>
    """
    )


@app.route("/stream")
def stream():
    """Multipart JPEG stream of the variance overlay."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=tof_grid_port(), threaded=True)
