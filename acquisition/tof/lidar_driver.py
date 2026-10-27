"""HTTP client for the MaixSense depth stream."""

import struct

import numpy as np
import requests


class LidarDriver:
    """Keep one HTTP session open and decode each depth frame as uint16 millimetres."""

    def __init__(self, host=None, port=None):
        from acquisition.settings import sensor_host, sensor_port

        self.host = sensor_host() if host is None else host
        self.port = sensor_port() if port is None else port
        # A persistent connection avoids a new TCP handshake on every frame.
        self.http_session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=0)
        self.http_session.mount("http://", adapter)

    def _frame_config_decode(self, frame_config):
        """Unpack the C struct that describes the frame layout."""
        return struct.unpack("<BBBBBBBBi", frame_config)

    def _frame_config_encode(
        self,
        trigger_mode=1,
        deep_mode=1,
        deep_shift=255,
        ir_mode=1,
        status_mode=2,
        status_mask=7,
        rgb_mode=1,
        rgb_res=0,
        expose_time=0,
    ):
        """Pack camera settings into the binary header the sensor expects."""
        return struct.pack(
            "<BBBBBBBBi",
            trigger_mode,
            deep_mode,
            deep_shift,
            ir_mode,
            status_mode,
            status_mask,
            rgb_mode,
            rgb_res,
            expose_time,
        )

    def _frame_payload_decode(self, frame_data: bytes, with_config: tuple):
        """Decode the payload into a 240 by 320 depth image. One unit is one millimetre."""
        _deep_data_size, _rgb_data_size = struct.unpack("<ii", frame_data[:8])
        frame_payload = frame_data[8:]
        # Raw mode is 320x240 pixels, two bytes each.
        deep_len = 320 * 240 * 2
        depth_bytes = frame_payload[:deep_len]
        return np.frombuffer(depth_bytes, dtype=np.uint16).reshape((240, 320))

    def setup_camera(self):
        """Ask the sensor for raw 16-bit depth. Exposure 80 trades frame rate for less noise."""
        config = self._frame_config_encode(
            trigger_mode=1,
            deep_mode=0,  # Raw distance, no onboard colour map.
            deep_shift=0,  # No bit shift, so the full 16-bit range is kept.
            ir_mode=0,
            status_mode=2,
            status_mask=7,
            rgb_mode=1,
            rgb_res=0,
            expose_time=80,
        )
        try:
            self.http_session.post(f"http://{self.host}:{self.port}/set_cfg", config, timeout=2)
        except requests.exceptions.RequestException:
            pass

    def get_frame(self):
        """Return the latest depth image, or None when the sensor does not answer in time."""
        try:
            response = self.http_session.get(f"http://{self.host}:{self.port}/getdeep", timeout=0.5)
            if response.status_code == 200:
                payload = response.content
                config = self._frame_config_decode(payload[16:28])
                return self._frame_payload_decode(payload[28:], config)
        except requests.exceptions.RequestException:
            pass
        return None
