"""HTTP client for the MaixSense RGB stream."""

import struct

import cv2
import numpy as np
import requests


def _url(path):
    from acquisition.settings import sensor_host, sensor_port

    return f"http://{sensor_host()}:{sensor_port()}{path}"


# One session for every frame, so the camera does not renegotiate TCP each time.
http_session = requests.Session()


def frame_config_decode(frame_config):
    """Unpack the C struct that describes the frame layout."""
    return struct.unpack("<BBBBBBBBi", frame_config)


def frame_config_encode(
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


def frame_payload_decode(frame_data: bytes, with_config: tuple):
    """Skip depth, infrared, and status blocks, then decode the JPEG colour image."""
    _deep_data_size, _rgb_data_size = struct.unpack("<ii", frame_data[:8])
    frame_payload = frame_data[8:]

    depth_size = (320 * 240 * 2) >> with_config[1]
    frame_payload = frame_payload[depth_size:]

    ir_size = (320 * 240 * 2) >> with_config[3]
    frame_payload = frame_payload[ir_size:]

    status_size = (320 * 240 // 8) * (
        16 if 0 == with_config[4] else 2 if 1 == with_config[4] else 8 if 2 == with_config[4] else 1
    )
    frame_payload = frame_payload[status_size:]

    rgb_size = len(frame_payload)
    rgb_img_bytes = struct.unpack("<%us" % rgb_size, frame_payload[:rgb_size])[0] if 0 != rgb_size else None

    if (rgb_img_bytes is not None) and (1 == with_config[6]):
        jpeg = cv2.imdecode(np.frombuffer(rgb_img_bytes, "uint8"), cv2.IMREAD_COLOR)
        if jpeg is not None:
            return jpeg
    return None


def setup_camera():
    """Enable the RGB stream at the sensor's default resolution."""
    print(f"Configuring camera {_url('')}")
    config = frame_config_encode(
        trigger_mode=1,
        deep_mode=1,
        deep_shift=255,
        ir_mode=0,
        status_mode=2,
        status_mask=7,
        rgb_mode=1,
        rgb_res=0,
        expose_time=0,
    )
    try:
        response = http_session.post(_url("/set_cfg"), config, timeout=2)
        if response.status_code == requests.codes.ok:
            return True
    except requests.exceptions.RequestException:
        pass
    return False


def get_maix_frame():
    """Return the latest BGR frame, or None when the packet is missing."""
    try:
        response = http_session.get(_url("/getdeep"), timeout=1)
        if response.status_code == requests.codes.ok:
            payload = response.content
            config = frame_config_decode(payload[16:28])
            return frame_payload_decode(payload[28:], config)
    except requests.exceptions.RequestException:
        pass
    return None
