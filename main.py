"""
Autonomous Runway Intelligence & FOD Removal System — LAPTOP SIDE
-----------------------------------------------------------------
Receives live video from the Pi's webcam, runs YOLO FOD detection,
and executes the full mission: drive 5m, pick all FOD found, U-turn,
drive 5m back home.

ARCHITECTURE:
  Pi webcam ──TCP video──► NetworkVideoStream (laptop)
                                    │
                                    ▼
                              FODDetector (YOLO, best.pt)
                                    │
                                    ▼
                           MissionController
                           (outbound 5m → U-turn → return 5m)
                                    │ drive + arm_pick commands
                                    ▼
                           RoverClient ──TCP──► Pi (rover_server.py)

RUN:
    python main.py
    python main.py --config config/config.yaml

Press 'q' in the video window to quit.
The rover receives an immediate stop command before the program exits.
"""

import argparse
import time
import sys
import cv2

from src.utils.config_loader import load_config
from src.utils.logger import setup_logging, get_logger
from src.camera.network_video_receiver import NetworkVideoStream
from src.detection.detector import FODDetector
from src.navigation.navigator import Navigator
from src.navigation.rover_client import RoverClient
from src.mission.mission_controller import MissionController
from src.visualization.renderer import draw_detections, draw_hud


def parse_args():
    parser = argparse.ArgumentParser(description="FOD Detection + Mission System")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)

    setup_logging(
        level       = config["logging"]["level"],
        log_to_file = config["logging"]["log_to_file"],
        log_file    = config["logging"]["log_file"],
    )
    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info("Autonomous Runway Intelligence & FOD Removal System")
    logger.info("=" * 60)

    # -- Video stream (Pi sends frames here) ----------------------------------
    net_cfg      = config["network"]
    video_stream = NetworkVideoStream(
        listen_ip   = "0.0.0.0",
        listen_port = net_cfg["video_listen_port"],
    ).start()
    logger.info(f"Waiting for Pi video on port {net_cfg['video_listen_port']}...")

    # -- Detection model -------------------------------------------------------
    model_cfg = config["model"]
    try:
        detector = FODDetector(
            weights_path         = model_cfg["weights_path"],
            confidence_threshold = model_cfg["confidence_threshold"],
            iou_threshold        = model_cfg["iou_threshold"],
            device               = model_cfg["device"],
            image_size           = model_cfg["image_size"],
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(f"Put your trained weights at '{model_cfg['weights_path']}'")
        video_stream.stop()
        sys.exit(1)

    # Show a real frame before connecting the drive client. This is the
    # readiness gate that prevents the mission from driving blind.
    display_cfg = config["display"]
    logger.info("Waiting for first video frame from Pi...")
    while True:
        frame_id, frame = video_stream.read()
        if frame is not None:
            cv2.imshow(display_cfg["window_name"], frame)
            cv2.waitKey(1)
            break
        time.sleep(0.1)
    logger.info("Video stream is visible on the laptop. Rover control enabled.")

    # -- Rover connection (blocks until Pi accepts) ----------------------------
    try:
        rover_client = RoverClient(
            rover_ip     = net_cfg["rover_ip"],
            rover_port   = net_cfg["rover_command_port"],
            send_rate_hz = net_cfg["send_rate_hz"],
        )
    except OSError as e:
        logger.error(f"Could not connect to rover: {e}")
        logger.error("Make sure rover_server.py is running on the Pi.")
        video_stream.stop()
        sys.exit(1)

    # -- Navigation + mission --------------------------------------------------
    navigator = Navigator(config["navigation"])
    mission   = MissionController(config["mission"], navigator, rover_client)
    prev_time             = time.time()
    fps                   = 0.0
    last_processed_frame  = -1

    try:
        while True:
            frame_id, frame = video_stream.read()
            if frame is None:
                time.sleep(0.01)
                continue
            if frame_id == last_processed_frame:
                time.sleep(0.005)
                continue
            last_processed_frame = frame_id

            # Detect FOD objects in the current frame
            infer_start = time.time()
            detections = detector.detect(frame, frame_id)
            infer_time = time.time() - infer_start
            if infer_time > 1.0:  # Log slow inferences
                logger.debug(f"YOLO inference took {infer_time:.2f}s (watchdog timeout is {mission.rover_client.min_send_interval:.1f}s)")

            # Mission controller handles everything:
            # cruise straight, stop for FOD, approach, arm pick, U-turn, return
            mission_status = mission.update(detections, frame_width=frame.shape[1])

            # Draw detections and HUD
            frame = draw_detections(frame, detections, display_cfg)

            now         = time.time()
            instant_fps = 1.0 / max(now - prev_time, 1e-6)
            fps         = 0.9 * fps + 0.1 * instant_fps
            prev_time   = now
            frame       = draw_hud(frame, fps, len(detections), display_cfg)

            # Mission state overlay
            state = mission_status.get("state", "")
            info  = f"Mission: {state}"
            if "leg_remaining_s"  in mission_status:
                info += f"  |  Leg: ~{mission_status['leg_remaining_s']}s left"
            elif "turn_remaining_s" in mission_status:
                info += f"  |  Turning: ~{mission_status['turn_remaining_s']}s left"
            cv2.putText(frame, info, (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

            cv2.imshow(display_cfg["window_name"], frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit key pressed.")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted (Ctrl+C).")

    finally:
        logger.info("Stopping rover and cleaning up...")
        rover_client.stop()
        video_stream.stop()
        cv2.destroyAllWindows()
        logger.info("System stopped cleanly.")


if __name__ == "__main__":
    main()
