"""On-screen trace and the two optical-flow methods that are not OpenCV Farnebäck."""

import cv2
import numpy as np


class RealTimePlotter:
    """Draw a scrolling trace. The vertical scale follows the 98th percentile, not the peak."""

    def __init__(self, width=320, height=100, bg_color=(30, 30, 30), line_color=(0, 255, 0)):
        self.w = width
        self.h = height
        self.bg_color = bg_color
        self.line_color = line_color
        self.canvas = np.full((self.h, self.w, 3), self.bg_color, dtype=np.uint8)
        # Floor on the Y range so a breath-hold does not zoom into sensor noise.
        self.min_y_range = 0.5

    def set_scale_reference(self, amplitude_std):
        """Set the display range from the standard deviation measured at calibration."""
        if amplitude_std > 0:
            self.min_y_range = amplitude_std * 6.0

    def update(self, data_buffer):
        """Return the canvas with the latest buffer drawn on it."""
        self.canvas[:] = self.bg_color
        if len(data_buffer) < 2:
            return self.canvas

        data = np.array(list(data_buffer))
        data = data - np.mean(data)
        limit = np.percentile(np.abs(data), 98)
        display_limit = max(limit, self.min_y_range / 2.0)
        draw_h = self.h * 0.9
        scale = (draw_h / 2.0) / display_limit if display_limit > 1e-5 else 1.0
        center_y = self.h / 2.0
        xs = np.linspace(0, self.w - 1, len(data)).astype(int)
        ys = center_y - (np.clip(data, -display_limit, display_limit) * scale)
        ys = np.clip(ys, 0, self.h - 1).astype(int)
        pts = np.column_stack((xs, ys))
        cv2.polylines(self.canvas, [pts], isClosed=False, color=self.line_color, thickness=1, lineType=cv2.LINE_AA)
        return self.canvas


def sparse_lucas_kanade_features(prev_gray, curr_gray, max_points=50):
    """Median vertical motion of Shi-Tomasi points tracked with pyramidal Lucas-Kanade."""
    feature_params = dict(maxCorners=max_points, qualityLevel=0.01, minDistance=5, blockSize=3)
    points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    if points is None:
        return 0.0

    points = points.reshape(-1, 2)
    if len(points) < 1:
        return 0.0

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    tracked, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, points, None, **lk_params)
    if tracked is None:
        return 0.0

    keep = status.flatten() == 1
    good_new = tracked[keep]
    good_old = points[keep]
    if len(good_new) < 1:
        return 0.0

    # The median ignores points that jumped to the wrong texture.
    flow_y = good_new[:, 1] - good_old[:, 1]
    return float(np.median(flow_y))


def horn_schunck_numpy_dense(img1, img2, alpha=1.0, Niter=3):
    """Mean vertical Horn-Schunck flow. Smoothness is controlled by alpha and by Niter."""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Spatial and temporal derivatives on the overlapping pixels.
    ex = (img1[:-1, 1:] - img1[:-1, :-1] + img2[:-1, 1:] - img2[:-1, :-1]) / 2
    ey = (img1[1:, :-1] - img1[:-1, :-1] + img2[1:, :-1] - img2[:-1, :-1]) / 2
    et = img2[:-1, :-1] - img1[:-1, :-1]

    horizontal = np.zeros_like(ex)
    vertical = np.zeros_like(ex)
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32) / 4

    for _ in range(Niter):
        horizontal_avg = cv2.filter2D(horizontal, -1, kernel)
        vertical_avg = cv2.filter2D(vertical, -1, kernel)
        numerator = ex * horizontal_avg + ey * vertical_avg + et
        denominator = alpha**2 + ex**2 + ey**2
        denominator[denominator == 0] = 1e-6
        horizontal = horizontal_avg - ex * (numerator / denominator)
        vertical = vertical_avg - ey * (numerator / denominator)

    return float(np.mean(vertical))
