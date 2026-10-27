import cv2
import numpy as np
import time
from flask import Flask, Response, render_template_string, jsonify

from acquisition.settings import tof_http_port, tof_recordings_dir
from . import data_logger as dl
from . import data_plotter as plotter
from .lidar_driver import LidarDriver
from .signal_processing import AdaptiveTrackingStrategy, SignalProcessor

# Seconds to sit in front of the camera, then seconds used to build the variance model.
SETUP_DURATION = 5
CALIB_DURATION = 30
driver = LidarDriver()
strategy = AdaptiveTrackingStrategy()
proc = SignalProcessor()
logger = dl.BreathLogger(output_dir=tof_recordings_dir())

force_reset = False
app = Flask(__name__)

def generate_frames():
    """Yield JPEG frames: setup, calibration, then the live depth trace."""
    global force_reset
    
    driver.setup_camera()
    start_time = time.time()
    calibration_done = False
    
    while True:
        if force_reset:
            start_time = time.time()
            calibration_done = False
            strategy.reset()
            proc.reset()
            force_reset = False
            print("--- system reset ---")

        # --- AKWIZYCJA DANYCH ---
        depth = driver.get_frame()
        if depth is None:
            time.sleep(0.01)
            continue
            
        # Preview uses 8 bits. Full scale is about 4 m.
        vis_preview = np.clip(depth / 16, 0, 255).astype(np.uint8)
        
        elapsed = time.time() - start_time
        phase = "SETUP"
        
        if elapsed > SETUP_DURATION:
            if elapsed < (SETUP_DURATION + CALIB_DURATION):
                phase = "CALIB"
                strategy.add_calibration_frame(depth)
            else:
                phase = "RUN"
                if not calibration_done:
                    print("Building the variance model...")
                    success = strategy.build_variance_model()
                    if not success:
                        # Automatyczny restart w przypadku nieudanej kalibracji
                        force_reset = True 
                    calibration_done = True

        if phase == "RUN":
            vis_frame, signal_val = strategy.process(depth) 
            proc.update(signal_val, is_calibrating=False)
            
            if logger.is_recording:
                filtered_val = proc.filtered[-1] if len(proc.filtered) > 0 else 0.0
                is_apnea = True if (proc.bpm == 0.0) else False
                logger.log_sample("ToF", signal_val, filtered_val, proc.bpm, is_apnea)

        else:
            vis_frame = cv2.cvtColor(vis_preview, cv2.COLOR_GRAY2BGR)
            
            if phase == "CALIB":
                progress = (elapsed - SETUP_DURATION) / CALIB_DURATION
                bar_w = int(320 * progress)
                cv2.rectangle(vis_frame, (0, 230), (bar_w, 240), (0, 255, 255), -1)
            
            signal_val = 0

        h_cam, w_cam = vis_frame.shape[:2]
        header_h = 60
        plot_h = 100
        total_h = header_h + h_cam + plot_h
        
        canvas = np.zeros((total_h, w_cam, 3), dtype=np.uint8)
        
        if vis_frame.dtype == np.uint16:
            vis_frame = np.clip(vis_frame / 16, 0, 255).astype(np.uint8)
        
        if len(vis_frame.shape) == 2:
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_GRAY2BGR)
                
        canvas[header_h:header_h+h_cam, 0:w_cam] = vis_frame

        if phase == "SETUP":
            cv2.putText(canvas, "SIT COMFORTABLY", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
            cv2.putText(canvas, "High Precision Mode (16-bit)", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        elif phase == "CALIB":
            cv2.putText(canvas, "BREATHE DEEPLY", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            rem = int((SETUP_DURATION + CALIB_DURATION) - elapsed)
            cv2.putText(canvas, f"Collecting data: {rem}s", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        elif phase == "RUN":
            col = (0,255,0) if 8 <= proc.bpm <= 30 else (0,0,255)
            
            if len(proc.buf) < int(proc.fps * 15.0):
                bpm_txt = "BPM: LOADING..."
                col = (0, 165, 255) 
            else:
                bpm_txt = f"BPM: {proc.bpm:.1f}" if proc.bpm > 0 else "APNEA!"
                
            cv2.putText(canvas, bpm_txt, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, col, 3)
            
            if logger.is_recording:
                if int(time.time()*2)%2==0:
                    cv2.circle(canvas, (w_cam-30, 30), 10, (0,0,255), -1)
                    cv2.putText(canvas, "REC", (w_cam-80, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        plot_area_y = header_h + h_cam
        cv2.line(canvas, (0, plot_area_y), (w_cam, plot_area_y), (50,50,50), 1)

        if len(proc.filtered) > 1 and phase == "RUN":
            sig = np.array(proc.filtered)
            
            try:
                amp = (proc.scale_max - proc.scale_min)
                scale_factor = max(amp, 1.0) 
                norm = sig / (scale_factor * 0.5) 
            except Exception:
                norm = np.zeros_like(sig)
            
            pts = []
            cy = plot_area_y + (plot_h // 2)
            
            visible_len = 150
            
            for i, v in enumerate(norm):
                if len(norm) > visible_len and i < len(norm) - visible_len: 
                    continue 
                
                actual_visible = min(len(norm), visible_len)
                idx_in_window = i - (len(norm) - actual_visible)
                
                x = int(idx_in_window * (w_cam / actual_visible))
                y = int(cy - (v * (plot_h / 2)))
                y = max(plot_area_y, min(total_h - 1, y))
                pts.append((x, y))
            
            if len(pts) > 1:
                cv2.polylines(canvas, [np.array(pts)], False, (0,255,255), 2)

        ok, buf = cv2.imencode(".jpg", canvas)
        if ok: yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

@app.route("/")
def home():
    """Operator page for the ToF recording."""
    html = """
    <html>
    <head>
        <style>
            body { font-family: sans-serif; background: #222; color: #fff; text-align: center; }
            img { border: 2px solid #555; margin-top: 10px; max-width: 100%; }
            button {
                background-color: #d9534f; border: none; color: white;
                padding: 15px 32px; text-align: center; text-decoration: none;
                display: inline-block; font-size: 16px; margin: 4px 2px;
                cursor: pointer; border-radius: 8px; font-weight: bold;
            }
            button:hover { background-color: #c9302c; }
            .btn-start { background-color: #28a745; }
            .btn-start:hover { background-color: #218838; }
            .btn-reset { background-color: #f0ad4e; color: black; }
            .btn-reset:hover { background-color: #ec971f; }
        </style>
        <script>
            function req(ep) { fetch(ep).then(r => r.json()).then(d => alert(d.msg)); }
            function resetCalibration() { fetch('/reset_trigger'); }
        </script>
    </head>
    <body>
        <h1>ToF depth (16-bit)</h1>
        <p>Millimetre mode, sub-sample peak interpolation, hybrid apnea guard</p>
        
        <div>
            <button class="btn-start" onclick="req('/start_rec')">START REC</button>
            <button onclick="req('/stop_rec')">STOP REC</button>
            <button class="btn-reset" onclick="resetCalibration()">RECALIBRATE</button>
        </div>
        
        <img src='/stream'>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route("/stream")
def stream():
    """JPEG stream from the frame generator."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/reset_trigger")
def reset_trigger():
    """Restart setup and calibration."""
    global force_reset
    force_reset = True
    return "OK"

@app.route("/start_rec")
def start_rec():
    """Start a ToF recording."""
    logger.start(["ToF"])
    return jsonify({"msg": "Recording started"})

@app.route("/stop_rec")
def stop_rec():
    """Stop logging and save the session figure."""
    logger.stop()
    res = plotter.generate_session_plot()
    msg = "Files saved. Figure ready." if res else "No samples to plot."
    return jsonify({"msg": msg})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=tof_http_port(), threaded=True)