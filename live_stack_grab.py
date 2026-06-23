"""
Live-preview Jenga stacking runner.

Shows the RealSense camera stream with the current best detection overlay.
Press g in the preview window to grab the currently detected block and place it
on the stack. Press q or Esc to quit.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from simple_jenga_grab import (
    DEFAULT_CONFIG,
    VisionPipeline,
    draw_detection,
    enter_position_mode,
    get_arm_position_mm_deg,
    get_realsense_intrinsics,
    initialize_arm,
    is_position_safe,
    load_json,
    require_arm_ok,
    resolve_path,
    tcp_from_camera_transform,
    wait_for_realsense_frames,
)
from stack_grab import (
    compute_pick,
    compute_stack_place,
    execute_stack_pick_and_place,
    get_stack_travel_z,
    is_eye_in_hand_setup,
    move_to_scan_pose,
    print_stack_path_debug,
    write_debug_images,
)
from vision_pipeline import EXPECTED_LONG_PX, EXPECTED_SHORT_PX


WINDOW_NAME = "Live Stack Grab"
DEFAULT_LIVE_WIDTH = 1280
DEFAULT_LIVE_HEIGHT = 720
DEFAULT_FAST_MOVE_SPEED = 180.0
DEFAULT_FAST_SPEED_MIN_CLEARANCE_MM = 35.0


def prepare_live_config(base_config, width, height, fps):
    config = dict(base_config)
    config["camera"] = dict(base_config["camera"])
    source_width = float(config["camera"].get("width", width))
    source_height = float(config["camera"].get("height", height))
    config["camera"]["width"] = int(width)
    config["camera"]["height"] = int(height)
    if fps is not None:
        config["camera"]["fps"] = int(fps)

    scale_x = float(width) / source_width if source_width > 0 else 1.0
    scale_y = float(height) / source_height if source_height > 0 else 1.0
    linear_scale = (scale_x + scale_y) / 2.0
    area_scale = max(scale_x * scale_y, 1.0)

    config["vision"] = dict(base_config.get("vision", {}))
    if linear_scale > 1.0:
        config["vision"]["edge_margin_px"] = int(
            round(float(config["vision"].get("edge_margin_px", 40)) * linear_scale)
        )
        config["vision"]["min_area_px"] = int(
            round(float(config["vision"].get("min_area_px", 500)) * area_scale)
        )
        config["vision"]["max_long_px"] = float(
            config["vision"].get("max_long_px", EXPECTED_LONG_PX * 1.75)
        ) * linear_scale
        config["vision"]["max_short_px"] = float(
            config["vision"].get("max_short_px", EXPECTED_SHORT_PX * 1.8)
        ) * linear_scale

    return config


class RealSenseLiveStream:
    """Keep one aligned RealSense color/depth stream open for preview and grabs."""

    def __init__(self, config):
        self.config = config
        self.camera_cfg = config["camera"]
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            rs.stream.depth,
            int(self.camera_cfg["width"]),
            int(self.camera_cfg["height"]),
            rs.format.z16,
            int(self.camera_cfg["fps"]),
        )
        self.rs_config.enable_stream(
            rs.stream.color,
            int(self.camera_cfg["width"]),
            int(self.camera_cfg["height"]),
            rs.format.bgr8,
            int(self.camera_cfg["fps"]),
        )
        self.profile = None
        self.align = rs.align(rs.stream.color)
        self.started = False
        self.intrinsics = None
        self.depth_scale_m = None

    def start(self):
        self.profile = self.pipeline.start(self.rs_config)
        self.started = True

        warmup_frames = int(self.camera_cfg.get("warmup_frames", 30))
        frame_timeout_ms = int(self.camera_cfg.get("frame_timeout_ms", 5000))
        frame_retries = int(self.camera_cfg.get("frame_retries", 3))
        for _ in range(warmup_frames):
            wait_for_realsense_frames(
                self.pipeline,
                frame_timeout_ms,
                frame_retries,
                "warmup",
            )

        color, _, color_frame = self.read(include_frame=True)
        self.intrinsics = get_realsense_intrinsics(color_frame)
        self.depth_scale_m = float(
            self.profile.get_device().first_depth_sensor().get_depth_scale()
        )
        print(f"  Live color frame: {color.shape}")
        print(
            "  RealSense intrinsics: "
            f"fx={self.intrinsics['fx']:.1f}, fy={self.intrinsics['fy']:.1f}, "
            f"depth_scale={self.depth_scale_m:.6f} m/unit"
        )
        return self

    def read(self, include_frame=False):
        frame_timeout_ms = int(self.camera_cfg.get("frame_timeout_ms", 5000))
        frame_retries = int(self.camera_cfg.get("frame_retries", 3))
        frames = wait_for_realsense_frames(
            self.pipeline,
            frame_timeout_ms,
            frame_retries,
            "preview",
        )
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense did not return both color and depth frames.")

        color = np.asanyarray(color_frame.get_data()).copy()
        depth = np.asanyarray(depth_frame.get_data()).copy()
        if include_frame:
            return color, depth, color_frame
        return color, depth

    def stop(self):
        if self.started:
            self.pipeline.stop()
            self.started = False


def draw_live_overlay(preview, best, stack_count, busy=False):
    if best is None:
        status = "No valid Jenga block detected"
        color = (0, 0, 255)
    else:
        u, v = best["center_uv"]
        status = (
            f"Ready: score={best['score']:.2f} "
            f"z={best['depth_m']:.3f}m yaw={best['gripper_yaw_deg']:.1f} "
            f"uv=({u},{v})"
        )
        color = (0, 180, 0)

    if busy:
        status = "Robot is moving..."
        color = (0, 180, 255)

    cv2.rectangle(preview, (0, 0), (preview.shape[1], 76), (20, 20, 20), -1)
    cv2.putText(
        preview,
        status,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        f"Stack count: {stack_count}   g: grab current detection   q/Esc: quit",
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return preview


def copy_config_with_move_speed(config, move_speed):
    fast_config = dict(config)
    fast_config["robot"] = dict(config["robot"])
    fast_config["robot"]["move_speed"] = float(move_speed)
    return fast_config


def choose_live_move_speed(config, pick_xyz, place_xyz, travel_z, fast_move_speed, min_clearance):
    configured_speed = float(config["robot"]["move_speed"])
    fast_move_speed = float(fast_move_speed)
    min_clearance = float(min_clearance)
    clearance = float(travel_z) - max(float(pick_xyz[2]), float(place_xyz[2]))
    if fast_move_speed <= configured_speed:
        return configured_speed, clearance, "configured"
    if clearance < min_clearance:
        return configured_speed, clearance, "clearance-limited"
    return float(fast_move_speed), clearance, "fast-clearance"


def stack_current_detection(
    arm,
    config,
    config_dir,
    stack_count,
    debug_index,
    color,
    depth,
    pipeline,
    best,
    fast_move_speed,
    fast_speed_min_clearance_mm,
):
    if best is None:
        print("  No valid Jenga block detected. Wait for the overlay, then press g again.")
        return False

    capture_pose = get_arm_position_mm_deg(arm)
    write_debug_images(
        resolve_path(config_dir, config["grasp"].get("debug_dir", "grab_debug")),
        debug_index,
        color,
        depth,
        pipeline,
        best,
    )

    cam_x, cam_y, cam_z = best["xyz_cam"]
    print(
        "  Detection: "
        f"score={best['score']:.3f}, "
        f"camera_xyz=({cam_x:.4f}, {cam_y:.4f}, {cam_z:.4f}) m, "
        f"vision_yaw={best['gripper_yaw_deg']:.1f}"
    )

    robot_xyz, pick_yaw = compute_pick(config, config_dir, best, capture_pose)
    place_xyz, place_yaw, layer_number, slot_number = compute_stack_place(
        config,
        stack_count,
    )

    safe_range = config["safe_range"]
    if not is_position_safe(robot_xyz[0], robot_xyz[1], robot_xyz[2], safe_range):
        print("  Pick target is outside the configured safe range. Skip.")
        return False
    if not is_position_safe(place_xyz[0], place_xyz[1], place_xyz[2], safe_range):
        print("  Place target is outside the configured safe range. Skip.")
        return False

    travel_z = get_stack_travel_z(config, robot_xyz, place_xyz)
    live_move_speed, clearance, speed_reason = choose_live_move_speed(
        config,
        robot_xyz,
        place_xyz,
        travel_z,
        fast_move_speed,
        fast_speed_min_clearance_mm,
    )
    print(
        "  Pick: "
        f"x={robot_xyz[0]:.1f}, y={robot_xyz[1]:.1f}, z={robot_xyz[2]:.1f}, "
        f"yaw={pick_yaw:.1f}"
    )
    print(
        "  Place: "
        f"x={place_xyz[0]:.1f}, y={place_xyz[1]:.1f}, z={place_xyz[2]:.1f}, "
        f"yaw={place_yaw:.1f}, layer={layer_number}, slot={slot_number}/2"
    )
    print_stack_path_debug(config, stack_count, robot_xyz, place_xyz, travel_z)
    print(
        "  Live speed: "
        f"travel={live_move_speed:.1f} mm/s ({speed_reason}), "
        f"approach={float(config['robot']['approach_speed']):.1f} mm/s, "
        f"clearance={clearance:.1f} mm"
    )

    motion_config = copy_config_with_move_speed(config, live_move_speed)
    execute_stack_pick_and_place(
        arm,
        motion_config,
        robot_xyz,
        place_xyz,
        pick_yaw,
        place_yaw,
    )
    move_to_scan_pose(arm, config)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Continuously stack Jenga blocks with live RealSense preview.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.json")
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_LIVE_WIDTH,
        help="Live RealSense stream width. Use 640 to match config.json.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_LIVE_HEIGHT,
        help="Live RealSense stream height. Use 480 to match config.json.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Live RealSense stream FPS. Defaults to camera.fps from the config.",
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=1.0,
        help="Scale the preview window without changing camera resolution.",
    )
    parser.add_argument(
        "--fast-move-speed",
        type=float,
        default=DEFAULT_FAST_MOVE_SPEED,
        help="Travel speed for safe live stack moves. Approach speed stays from config.",
    )
    parser.add_argument(
        "--fast-speed-min-clearance-mm",
        type=float,
        default=DEFAULT_FAST_SPEED_MIN_CLEARANCE_MM,
        help="Minimum travel clearance required before using the faster travel speed.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    config = prepare_live_config(load_json(config_path), args.width, args.height, args.fps)

    print("=" * 60)
    print("  Live Jenga Stack Grab")
    print("=" * 60)
    print(f"  Config: {config_path}")
    print(
        "  Live stream: "
        f"{config['camera']['width']}x{config['camera']['height']} "
        f"@ {config['camera']['fps']} FPS"
    )
    print("  Preview controls: press g to grab/place current block, q or Esc to quit.")

    if is_eye_in_hand_setup(config):
        tcp_from_camera_transform(config)

    arm = None
    stream = None
    stack_count = 0
    debug_index = 1

    try:
        arm = initialize_arm(config)
        stream = RealSenseLiveStream(config)
        enter_position_mode(arm)
        move_to_scan_pose(arm, config)
        stream.start()
        pipeline = VisionPipeline(
            stream.intrinsics,
            depth_scale_m=stream.depth_scale_m,
            config=config["vision"],
        )

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        preview_scale = max(float(args.preview_scale), 0.1)
        cv2.resizeWindow(
            WINDOW_NAME,
            int(stream.intrinsics["width"] * preview_scale),
            int(stream.intrinsics["height"] * preview_scale),
        )
        last_color = None
        last_depth = None
        last_best = None

        while True:
            color, depth = stream.read()
            best = pipeline.get_best(color, depth)
            preview = draw_detection(color, best)
            preview = draw_live_overlay(preview, best, stack_count)
            cv2.imshow(WINDOW_NAME, preview)

            last_color = color
            last_depth = depth
            last_best = best

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("  Quit requested.")
                break
            if key != ord("g"):
                continue

            if last_color is None or last_depth is None:
                print("  No frame is ready yet.")
                continue

            busy_preview = draw_live_overlay(
                draw_detection(last_color, last_best),
                last_best,
                stack_count,
                busy=True,
            )
            cv2.imshow(WINDOW_NAME, busy_preview)
            cv2.waitKey(1)

            print(f"\n[Grab {stack_count + 1}] Use current live frame")
            success = stack_current_detection(
                arm,
                config,
                config_dir,
                stack_count,
                debug_index,
                last_color,
                last_depth,
                pipeline,
                last_best,
                args.fast_move_speed,
                args.fast_speed_min_clearance_mm,
            )
            debug_index += 1
            if success:
                stack_count += 1
                print(f"  Stack count: {stack_count}")

            time.sleep(0.2)
            require_arm_ok(arm, "post-grab live preview")
    finally:
        if stream is not None:
            stream.stop()
        cv2.destroyAllWindows()
        if arm is not None:
            arm.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
