import cv2
import numpy as np
import time
from collections import deque
from flask import Flask, Response, render_template_string, redirect, url_for

from . import data_logger
from . import flow_utils as fu
from . import maix_driver as maix
from . import pose_estimation as pose
from . import signal_processing as sig
from acquisition.settings import rgb_http_port, rgb_recordings_dir

# Target frame rate of the optical-flow loop.
TARGET_FPS = 30.0

# Seconds spent sitting still, then breathing, before monitoring starts.
SETUP_SEC = 8.0
CALIB_SEC = 15.0

# Analysis window, in seconds. Matches BreathAnalysisEngine.
WINDOW_SEC = 30.0

# Leak of the position integrator. 1.0 would let the position drift without bound.
INTEGRATION_DECAY = 0.98

recorder = data_logger.BreathLogger(output_dir=rgb_recordings_dir())

class BreathStream:
    """One optical-flow method: smooth the velocity, integrate it, and draw the result."""
    def __init__(self, name, plot_color, smoothing_factor=0.3):
        self.name = name
        self.color = plot_color
        
        self.engine = sig.BreathAnalysisEngine(buffer_size_sec=WINDOW_SEC, fps=TARGET_FPS)
        
        self.bpm = None
        self.std = 0.0
        self.is_apnea = False
        self.calib_std = 1.0 
        self.breath_thresh = 0.1
        self.is_calibrated = False
        self.last_smooth_vy = 0.0
        self.alpha_smooth = smoothing_factor
        self.current_pos = 0.0
        
        self.plotter = fu.RealTimePlotter(width=320, height=80, line_color=plot_color)
        self.vis_buffer = []

    def reset_buffers(self):
        """Clear the buffers before calibration starts."""
        self.engine = sig.BreathAnalysisEngine(buffer_size_sec=WINDOW_SEC, fps=TARGET_FPS)
        self.last_smooth_vy = 0.0
        self.current_pos = 0.0
        self.vis_buffer = []
        self.plotter.update([]) 

    def update(self, t_now, vy_raw, dt, is_calibrating=False):
        """Smooth the velocity, leaky-integrate it into position, and hand the sample to the engine."""
        vy_smooth = (1.0 - self.alpha_smooth) * vy_raw + self.alpha_smooth * self.last_smooth_vy
        self.last_smooth_vy = vy_smooth
        
        # Leaky integrator: the decay stops a slow drift from walking the position off scale.
        self.current_pos = self.current_pos * INTEGRATION_DECAY + vy_smooth * max(dt, 1e-3)
        
        if is_calibrating: 
            self.engine.process_sample(t_now, self.current_pos, 0.0)
            if isinstance(self.vis_buffer, list):
                self.vis_buffer.append(self.current_pos)
                if len(self.vis_buffer) > int(WINDOW_SEC*TARGET_FPS): 
                    self.vis_buffer.pop(0)
            return

        bpm_res, apnea_flag, filtered_sig, std_res = self.engine.process_sample(t_now, self.current_pos, self.breath_thresh)
        
        self.std = std_res
        self.is_apnea = apnea_flag
        
        if bpm_res is not None: 
            self.bpm = bpm_res
        elif apnea_flag: 
            self.bpm = 0.0
            
        if len(filtered_sig) > 0:
            self.vis_buffer = filtered_sig

    def calibrate(self):
        """Set the apnea threshold from the standard deviation of the calibration window."""
        _, _, filtered_sig, _ = self.engine.process_sample(time.time(), self.current_pos, 0.0)
        if len(filtered_sig) < 30: return

        std_val = np.std(filtered_sig)
        self.calib_std = max(std_val, 0.0001)
        self.breath_thresh = self.calib_std * 0.25
        self.plotter.set_scale_reference(self.calib_std)
        self.is_calibrated = True
        print(f"[{self.name}] Calibration done. Std={self.calib_std:.4f}")

    def draw_panel(self, full_frame_bgr, roi_coords, keypoints, full_h):
        """Draw the camera, the rate, and the scrolling trace for this method."""
        panel_w = 320
        panel = np.zeros((full_h, panel_w, 3), dtype=np.uint8)
        
        display_img = full_frame_bgr.copy()
        
        if roi_coords is not None:
            rx, ry, rw, rh = [int(v) for v in roi_coords]
            cv2.rectangle(display_img, (rx, ry), (rx+rw, ry+rh), self.color, 2)
            cv2.putText(display_img, "Chest (ROI)", (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 1)

        if keypoints is not None:
            for i, pt in enumerate(keypoints):
                if pt is not None:
                    if i in [1, 2, 5]: 
                        cv2.circle(display_img, pt, 4, (255, 0, 0), -1) 
                    elif i in [18, 19]: 
                        cv2.circle(display_img, pt, 4, (255, 0, 255), -1) 
                    elif i in [20, 21]: 
                        cv2.circle(display_img, pt, 4, (0, 0, 255), -1) 

        ih, iw = display_img.shape[:2]
        if iw != panel_w:
            scale = panel_w / iw
            display_img = cv2.resize(display_img, (0,0), fx=scale, fy=scale)
        
        vid_h = min(display_img.shape[0], full_h - 140)
        y_vid = 40
        panel[y_vid:y_vid+vid_h, 0:panel_w] = display_img[0:vid_h, :]

        status_txt, status_col, head_col = "OK", (0,255,0), (50,50,50)
        if not self.is_calibrated: 
            status_txt, status_col = "CALIB", (150,150,150)
        elif self.is_apnea: 
            status_txt, status_col, head_col = "APNEA", (0,0,255), (0,0,100)
        
        cv2.rectangle(panel, (0,0), (panel_w, 40), head_col, -1)
        cv2.putText(panel, self.name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(panel, status_txt, (panel_w - 80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_col, 2)

        stats_y = y_vid + vid_h + 30
        if self.is_calibrated:
            bpm_val = f"{self.bpm:.1f}" if (self.bpm is not None and self.bpm > 0) else "--"
            if self.is_apnea: bpm_val = "0.0"
            cv2.putText(panel, f"RPM: {bpm_val}", (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_col, 2)
            
            bar_x, bar_w, bar_h = panel_w-30, 20, 80
            bar_y = stats_y - 60
            scale = self.calib_std * 3.0
            h_curr = int((self.std / scale)*bar_h)
            h_thresh = int((self.breath_thresh / scale)*bar_h)
            
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (30,30,30), -1)
            cv2.rectangle(panel, (bar_x, bar_y+bar_h-min(h_curr, bar_h)), (bar_x+bar_w, bar_y+bar_h), self.color if not self.is_apnea else (0,0,255), -1)
            cv2.line(panel, (bar_x-5, bar_y+bar_h-h_thresh), (bar_x+bar_w+5, bar_y+bar_h-h_thresh), (0,255,255), 2)
        else:
            cv2.putText(panel, "CALIBRATING...", (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150,150,150), 1)

        plot = self.plotter.update(self.vis_buffer)
        h_plot = plot.shape[0]
        panel[full_h-h_plot:full_h, 0:panel_w] = plot
        cv2.rectangle(panel, (0,0), (panel_w-1, full_h-1), (80,80,80), 1)
        
        return panel

app = Flask(__name__)
active_roi = None
active_keypoints = None

# Pose is expensive, and posture changes slowly, so it runs every fifth frame.
POSE_EVERY = 5

# How quickly the region of interest follows a new pose estimate.
ROI_SMOOTH_ALPHA = 0.2

# Per-method gain so the three flow fields enter the same analysis scale.
GAIN_FB = 10.0
GAIN_HS = 100.0
GAIN_LK = 25.0

def generate_frames():
    """Acquire frames, run the three flow methods, and yield JPEG dashboard images."""
    global active_roi, active_keypoints 
    maix.setup_camera()
    
    stream_fb = BreathStream("Farneback", (0, 255, 0), smoothing_factor=0.2)
    stream_hs = BreathStream("Horn-Schunck", (0, 255, 255), smoothing_factor=0.4)
    stream_lk = BreathStream("Lucas-Kanade", (255, 50, 255), smoothing_factor=0.3)
    
    streams = [stream_fb, stream_hs, stream_lk]
    prev_gray_roi = None
    frame_counter = 0
    t_start = time.time()
    t_last = time.time()
    
    calib_done = False
    bufs_cleared = False

    while True:
        frame = maix.get_maix_frame()
        if frame is None: 
            time.sleep(0.01)
            continue
            
        t_now = time.time()
        dt = t_now - t_last
        t_last = t_now
        frame_counter += 1
        elapsed = t_now - t_start
        
        phase = "MONITOR"
        if elapsed < SETUP_SEC: 
            phase = "SETUP"
        elif elapsed < (SETUP_SEC + CALIB_SEC):
            phase = "CALIB"
            if not bufs_cleared:
                for s in streams: s.reset_buffers()
                bufs_cleared = True
        else:
            if not calib_done:
                for s in streams: s.calibrate()
                calib_done = True

        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Update the pose every few frames and ease the region of interest toward it.
        if frame_counter % POSE_EVERY == 0 or active_roi is None:
            tgt, new_keypoints = pose.get_pose_roi(frame)
            
            if new_keypoints is not None:
                active_keypoints = new_keypoints
                
            if tgt:
                if active_roi is None: 
                    active_roi = [float(v) for v in tgt]
                else:
                    for i in range(4): 
                        active_roi[i] = (1-ROI_SMOOTH_ALPHA)*active_roi[i] + ROI_SMOOTH_ALPHA*tgt[i]

        vy_fb, vy_hs, vy_lk = 0.0, 0.0, 0.0
        
        if active_roi:
            x, y, w, h = [int(v) for v in active_roi]
            h_img, w_img = gray_full.shape
            x = max(0, min(w_img-1, x)); y = max(0, min(h_img-1, y))
            w = max(1, min(w_img-x, w)); h = max(1, min(h_img-y, h))
            
            gray_roi = gray_full[y:y+h, x:x+w]
            gray_boost = sig.contrast_boost(gray_roi)
            
            if prev_gray_roi is not None and prev_gray_roi.shape == gray_boost.shape:
                
                # Dense Farneback flow. The gain puts it on the same scale as the other two methods.
                flow = cv2.calcOpticalFlowFarneback(prev_gray_roi, gray_boost, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                vy_fb = float(np.mean(flow[..., 1])) * GAIN_FB
                
                # Dense Horn-Schunck. Large regions are halved so the iteration stays cheap.
                hs_p, hs_c = prev_gray_roi, gray_boost
                if w > 100: 
                    hs_p = cv2.resize(hs_p, (0,0), fx=0.5, fy=0.5)
                    hs_c = cv2.resize(hs_c, (0,0), fx=0.5, fy=0.5)
                vy_hs = fu.horn_schunck_numpy_dense(hs_p, hs_c, Niter=5) * GAIN_HS
                
                # Sparse Lucas-Kanade on Shi-Tomasi points.
                vy_lk = fu.sparse_lucas_kanade_features(prev_gray_roi, gray_boost) * GAIN_LK

            prev_gray_roi = gray_boost.copy()
        else:
            prev_gray_roi = None

        is_cal = (phase == "CALIB")
        streams[0].update(t_now, vy_fb, dt, is_calibrating=is_cal)
        streams[1].update(t_now, vy_hs, dt, is_calibrating=is_cal)
        streams[2].update(t_now, vy_lk, dt, is_calibrating=is_cal)

        if recorder.is_recording and not is_cal:
            for s in streams:
                recorder.log_sample(s.name, s.current_pos, s.std, s.bpm, s.is_apnea)

        panel_h = 440
        panels = [s.draw_panel(frame, active_roi, active_keypoints, panel_h) for s in streams]
        combined = np.hstack(panels)
        
        fh, fw = combined.shape[:2]
        if recorder.is_recording:
            cv2.circle(combined, (fw-30, 30), 10, (0, 0, 255), -1) 
            cv2.putText(combined, "REC", (fw-70, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        if phase != "MONITOR":
            ov = combined.copy()
            cv2.rectangle(ov, (0,0), (fw, 120), (0,0,0), -1)
            combined = cv2.addWeighted(ov, 0.7, combined, 0.3, 0)
            msg1, msg2 = ("STEP 1: SETUP", f"Starts in {int(SETUP_SEC-elapsed)} s") if phase=="SETUP" else ("STEP 2: CALIBRATION", f"Time left: {int(SETUP_SEC+CALIB_SEC-elapsed)} s")
            col = (0,255,255) if phase=="SETUP" else (0,255,0)
            cv2.putText(combined, msg1, (fw//2-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 2)
            cv2.putText(combined, msg2, (fw//2-80, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,200), 2)

        ret, buffer = cv2.imencode('.jpg', combined, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    """Operator page: the live dashboard and the record button."""
    return render_template_string("""
        <body style='background:#111; color:white; font-family:sans-serif; text-align:center; margin:0;'>
            <div style="padding:10px; background:#222; border-bottom:1px solid #444;">
                <h3 style="margin:0; display:inline-block;">RGB breath monitor</h3>
                <button id="recBtn" onclick="toggleRec()" style="margin-left:20px; padding:10px 20px; font-size:16px; background:#555; color:white; border:none; cursor:pointer;">
                    START RECORDING
                </button>
            </div>
            
            <div style="display:inline-block; border: 2px solid #333; margin-top:5px;">
                <img src="/video_feed" style='width:100%; max-width:1200px; height:auto;'/>
            </div>
            <p style='color:#666'>FB (Green) | HS (Yellow) | LK (Magenta)</p>

            <script>
                function toggleRec() {
                    fetch('/toggle_rec').then(r => r.text()).then(status => {
                        let btn = document.getElementById('recBtn');
                        if (status === 'RECORDING') {
                            btn.innerText = 'STOP RECORDING';
                            btn.style.background = '#d00';
                        } else {
                            btn.innerText = 'START RECORDING';
                            btn.style.background = '#555';
                        }
                    });
                }
            </script>
        </body>
    """)

@app.route('/toggle_rec')
def toggle_rec():
    """Start or stop the CSV recording."""
    if recorder.is_recording:
        recorder.stop()
        return "STOPPED"
    else:
        recorder.start(["Farneback", "Horn-Schunck", "Lucas-Kanade"])
        return "RECORDING"

@app.route('/video_feed')
def video_feed():
    """JPEG stream of the three-method dashboard."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=rgb_http_port(), debug=False, threaded=True)