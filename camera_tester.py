"""
Standalone FOD Detection Viewer
--------------------------------
Receives live video from the Raspberry Pi's webcam over TCP,
runs the YOLO model locally, and shows the detection results.

NO rover control. NO mission logic. NO arm. Just video + detection.

Use this to:
  - Verify your Pi video stream is working
  - Check your best.pt model is detecting correctly
  - Tune confidence threshold before running the full mission

USAGE
-----
1. Make sure rover_server.py is running on the Pi
2. Set PI_IP below to your Pi's IP address
3. Set MODEL_PATH below to your best.pt location
4. Run: python detect_viewer.py

Press 'q' to quit.
Press '+' to increase confidence threshold
Press '-' to decrease confidence threshold
"""

import socket
import struct
import threading
import time
import numpy as np
import cv2
from ultralytics import YOLO

# =============================================================================
# SETTINGS — edit these
# =============================================================================
PI_IP             = "aizen.local"   # <- your Pi's IP (hostname -I on Pi)
VIDEO_PORT        = 6000             # must match rover_server.py LAPTOP_VIDEO_PORT
MODEL_PATH        = "models/best.pt" # path to your trained YOLO weights
CONFIDENCE        = 0.5              # starting confidence threshold (0.0 - 1.0)
IOU_THRESHOLD     = 0.45
IMAGE_SIZE        = 640
WINDOW_NAME       = "FOD Detection Viewer  |  q=quit  +=conf up  -=conf down"
# =============================================================================

# Color palette — each class gets a consistent color
_PALETTE = [
    (66, 135, 245), (245, 66, 90),  (66, 245, 158), (245, 200, 66),
    (188, 66, 245), (66, 245, 236), (245, 66, 182), (154, 245, 66),
    (245, 130, 66), (66, 200, 245), (200, 245, 66), (245, 66, 140),
]


def color_for_class(class_id: int):
    return _PALETTE[class_id % len(_PALETTE)]


# -----------------------------------------------------------------------------
# Video receiver — same protocol as network_video_receiver.py
# Runs in a background thread, always holds the latest frame
# -----------------------------------------------------------------------------
class VideoReceiver:
    def __init__(self, port: int):
        self.port       = port
        self._frame     = None
        self._lock      = threading.Lock()
        self._running   = False
        self._connected = False

    def start(self):
        self._running = True
        threading.Thread(target=self._server_loop, daemon=True).start()
        return self

    def _server_loop(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", self.port))
        server.listen(1)
        print(f"[video] Listening on port {self.port}. Waiting for Pi to connect...")

        while self._running:
            try:
                server.settimeout(1.0)
                conn, addr = server.accept()
                print(f"[video] Pi connected from {addr}")
                self._connected = True
                self._recv_loop(conn)
                self._connected = False
                print("[video] Pi disconnected. Waiting for reconnect...")
            except socket.timeout:
                continue
            except OSError:
                break

    def _recv_loop(self, conn: socket.socket):
        conn.settimeout(2.0)
        try:
            while self._running:
                # Read 4-byte length header
                header = self._recv_exact(conn, 4)
                if header is None:
                    break
                (length,) = struct.unpack(">I", header)

                # Read JPEG payload
                payload = self._recv_exact(conn, length)
                if payload is None:
                    break

                # Decode JPEG to OpenCV frame
                arr   = np.frombuffer(payload, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    with self._lock:
                        self._frame = frame

        except (socket.timeout, ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    @staticmethod
    def _recv_exact(conn, n):
        buf = b""
        while len(buf) < n:
            try:
                chunk = conn.recv(n - len(buf))
            except socket.timeout:
                return None
            if not chunk:
                return None
            buf += chunk
        return buf

    def read(self):
        """Returns the latest frame, or None if none received yet."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def connected(self):
        return self._connected

    def stop(self):
        self._running = False


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 55)
    print("  FOD Detection Viewer")
    print("=" * 55)
    print(f"  Pi IP       : {PI_IP}")
    print(f"  Video port  : {VIDEO_PORT}")
    print(f"  Model       : {MODEL_PATH}")
    print(f"  Confidence  : {CONFIDENCE}")
    print("=" * 55)

    # Load YOLO model
    print("\nLoading YOLO model...")
    try:
        model = YOLO(MODEL_PATH)
        print(f"Model loaded. {len(model.names)} classes: {list(model.names.values())}")
    except Exception as e:
        print(f"ERROR: Could not load model: {e}")
        print(f"Make sure '{MODEL_PATH}' exists.")
        return

    # Start video receiver
    receiver = VideoReceiver(port=VIDEO_PORT)
    receiver.start()
    print(f"\nWaiting for Pi to connect on port {VIDEO_PORT}...")
    print("(Make sure rover_server.py is running on the Pi)\n")

    conf         = CONFIDENCE
    prev_time    = time.time()
    fps          = 0.0
    frame_count  = 0
    total_detections = 0

    try:
        while True:
            frame = receiver.read()

            # Show waiting screen if no frame yet
            if frame is None:
                waiting = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(waiting, "Waiting for Pi video stream...",
                            (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                cv2.putText(waiting, f"Pi IP: {PI_IP}  Port: {VIDEO_PORT}",
                            (120, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                cv2.putText(waiting, "Make sure rover_server.py is running on Pi",
                            (60, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)
                cv2.imshow(WINDOW_NAME, waiting)
                key = cv2.waitKey(200) & 0xFF
                if key == ord("q"):
                    break
                continue

            # Run YOLO detection
            results     = model.predict(
                source  = frame,
                conf    = conf,
                iou     = IOU_THRESHOLD,
                imgsz   = IMAGE_SIZE,
                device  = "cpu",
                verbose = False,
            )

            detections = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id     = int(box.cls[0])
                    class_name = model.names[cls_id]
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append((class_name, confidence, cls_id, x1, y1, x2, y2))

            total_detections += len(detections)
            frame_count      += 1

            # Draw bounding boxes and labels
            for class_name, confidence_val, cls_id, x1, y1, x2, y2 in detections:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                color           = color_for_class(cls_id)

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label background + text
                label            = f"{class_name}  {confidence_val:.2f}"
                (tw, th), base   = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - base - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Center dot
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)

            # FPS calculation (smoothed)
            now         = time.time()
            fps         = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time   = now

            # HUD overlay (top-left)
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, f"Detections: {len(detections)}",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, f"Conf threshold: {conf:.2f}  (+/-)",
                        (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

            # Connection status (top-right)
            status       = "Pi: CONNECTED" if receiver.connected else "Pi: RECONNECTING..."
            status_color = (0, 255, 0) if receiver.connected else (0, 100, 255)
            h, w         = frame.shape[:2]
            cv2.putText(frame, status, (w - 260, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

            cv2.imshow(WINDOW_NAME, frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("+") or key == ord("="):
                conf = min(0.95, round(conf + 0.05, 2))
                print(f"Confidence threshold: {conf:.2f}")
            elif key == ord("-"):
                conf = max(0.05, round(conf - 0.05, 2))
                print(f"Confidence threshold: {conf:.2f}")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        receiver.stop()
        cv2.destroyAllWindows()
        print(f"\nSession summary:")
        print(f"  Frames processed : {frame_count}")
        print(f"  Total detections : {total_detections}")
        print(f"  Final conf thresh: {conf:.2f}")
        print("Viewer closed.")


if __name__ == "__main__":
    main()