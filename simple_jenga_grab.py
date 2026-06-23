"""
Real-hardware simple Jenga grab.

This script performs one complete cycle:
    1. Capture one aligned RealSense color/depth frame.
    2. Detect the best Jenga block with OpenCV.
    3. Transform the camera-frame point into the xArm base frame.
    4. Run a simple pick-and-place motion on the real robot.

There is no simulator and no mock robot in this folder.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI

from vision_pipeline import VisionPipeline, draw_detection, normalize_angle


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def resolve_path(base_dir, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def get_realsense_intrinsics(color_frame):
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
        "model": str(intr.model),
        "coeffs": [float(value) for value in intr.coeffs],
    }


def wait_for_realsense_frames(rs_pipeline, timeout_ms, retries, label):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return rs_pipeline.wait_for_frames(timeout_ms)
        except RuntimeError as exc:
            last_error = exc
            print(f"  RealSense {label} timeout {attempt}/{retries}: {exc}")
            time.sleep(0.25)

    raise RuntimeError(
        "RealSense frame timeout. The camera pipeline started, but no frame "
        "arrived. Check that the camera is plugged into a USB 3 port, is not "
        "open in another app, and supports the configured resolution/FPS."
    ) from last_error


def capture_realsense_frame(config):
    camera_cfg = config["camera"]
    rs_pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(
        rs.stream.depth,
        int(camera_cfg["width"]),
        int(camera_cfg["height"]),
        rs.format.z16,
        int(camera_cfg["fps"]),
    )
    rs_config.enable_stream(
        rs.stream.color,
        int(camera_cfg["width"]),
        int(camera_cfg["height"]),
        rs.format.bgr8,
        int(camera_cfg["fps"]),
    )

    profile = None
    started = False
    try:
        profile = rs_pipeline.start(rs_config)
        started = True
        align = rs.align(rs.stream.color)

        warmup_frames = int(camera_cfg.get("warmup_frames", 30))
        frame_timeout_ms = int(camera_cfg.get("frame_timeout_ms", 5000))
        frame_retries = int(camera_cfg.get("frame_retries", 3))
        for _ in range(warmup_frames):
            wait_for_realsense_frames(
                rs_pipeline,
                frame_timeout_ms,
                frame_retries,
                "warmup",
            )

        frames = wait_for_realsense_frames(
            rs_pipeline,
            frame_timeout_ms,
            frame_retries,
            "capture",
        )
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense did not return both color and depth frames.")

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        intrinsics = get_realsense_intrinsics(color_frame)
        depth_scale_m = profile.get_device().first_depth_sensor().get_depth_scale()
        return color, depth, intrinsics, float(depth_scale_m)
    finally:
        if started:
            rs_pipeline.stop()


def load_calibration(config, config_dir):
    calibration_file = resolve_path(config_dir, config["calibration"]["file"])
    if not calibration_file.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {calibration_file}. "
            "Run calibrate.py first."
        )
    calibration = load_json(calibration_file)
    return calibration, calibration_file


def camera_to_robot_mm(xyz_cam_m, calibration):
    rotation = np.array(calibration["rotation_matrix"], dtype=float)
    translation = np.array(calibration["camera_position_mm"], dtype=float)
    cam_mm = np.array(xyz_cam_m, dtype=float) * 1000.0
    return rotation @ cam_mm + translation


def rot_x(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rpy_to_matrix(roll_deg, pitch_deg, yaw_deg):
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def make_transform(rotation, translation):
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return transform


def pose_mm_deg_to_transform(pose):
    if len(pose) < 6:
        raise ValueError(f"xArm pose must contain 6 values, got {pose}")
    x, y, z, roll, pitch, yaw = [float(value) for value in pose[:6]]
    return make_transform(rpy_to_matrix(roll, pitch, yaw), [x, y, z])


def get_arm_position_mm_deg(arm):
    try:
        code, pose = arm.get_position(is_radian=False)
    except TypeError:
        code, pose = arm.get_position()
    if code != 0:
        raise RuntimeError(f"Failed to read xArm position, code={code}")
    return [float(value) for value in pose]


def require_xarm_ok(arm, code, action):
    if code not in (0, None):
        raise RuntimeError(f"xArm command failed during {action}, code={code}")
    error_code = getattr(arm, "error_code", 0)
    if error_code != 0:
        raise RuntimeError(f"xArm reported an error during {action}, error_code={error_code}")


def enter_position_mode(arm):
    require_xarm_ok(arm, arm.set_mode(0), "set position mode")
    require_xarm_ok(arm, arm.set_state(0), "set ready state")


def enter_teach_mode(arm, sensitivity=None):
    if sensitivity is not None:
        require_xarm_ok(
            arm,
            arm.set_teach_sensitivity(int(sensitivity)),
            "set teach sensitivity",
        )
    require_xarm_ok(arm, arm.set_mode(2), "set joint teaching mode")
    require_xarm_ok(arm, arm.set_state(0), "set ready state")


def is_eye_in_hand_setup(config):
    setup = str(config["calibration"].get("setup", "fixed_camera")).lower()
    return setup in {"eye_in_hand_manual", "eye-in-hand-manual", "eye_in_hand"}


def tcp_from_camera_transform(config):
    manual = config["calibration"].get("eye_in_hand_manual", {})
    if not manual.get("verified", False):
        raise RuntimeError(
            "calibration.eye_in_hand_manual.verified is false. "
            "Measure camera_origin_in_tcp_mm and camera_axes_to_tcp_rotation_matrix, "
            "then set verified to true."
        )

    translation = manual.get("camera_origin_in_tcp_mm")
    if translation is None or len(translation) != 3:
        raise ValueError("camera_origin_in_tcp_mm must contain 3 numbers.")

    rotation_matrix = manual.get("camera_axes_to_tcp_rotation_matrix")
    rpy_deg = manual.get("camera_axes_to_tcp_rpy_deg")
    if rotation_matrix is not None:
        rotation = np.array(rotation_matrix, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError("camera_axes_to_tcp_rotation_matrix must be 3x3.")
    elif rpy_deg is not None:
        if len(rpy_deg) != 3:
            raise ValueError("camera_axes_to_tcp_rpy_deg must contain 3 numbers.")
        rotation = rpy_to_matrix(*rpy_deg)
    else:
        raise ValueError(
            "Provide camera_axes_to_tcp_rotation_matrix or camera_axes_to_tcp_rpy_deg."
        )

    return make_transform(rotation, translation)


def camera_to_robot_eye_in_hand_mm(xyz_cam_m, capture_pose, config):
    t_base_from_tcp = pose_mm_deg_to_transform(capture_pose)
    t_tcp_from_camera = tcp_from_camera_transform(config)
    point_camera = np.array([
        float(xyz_cam_m[0]) * 1000.0,
        float(xyz_cam_m[1]) * 1000.0,
        float(xyz_cam_m[2]) * 1000.0,
        1.0,
    ])
    point_base = t_base_from_tcp @ t_tcp_from_camera @ point_camera
    return point_base[:3]


def drag_camera_to_view(arm, config):
    grasp_cfg = config["grasp"]
    sensitivity = grasp_cfg.get("view_teach_sensitivity", 3)
    settle_seconds = float(grasp_cfg.get("view_settle_seconds", 0.5))

    print("""
[View Setup]
  The camera is configured as eye-in-hand.
  The script will put xArm into joint teaching mode.
  Drag the arm until the Jenga block is visible in the RealSense image.
  Release the arm, then press Enter to lock the pose and capture a frame.
""")
    enter_teach_mode(arm, sensitivity=sensitivity)
    input("  Press Enter after positioning the camera above the block...")
    enter_position_mode(arm)
    time.sleep(settle_seconds)
    pose = get_arm_position_mm_deg(arm)
    print(
        "  Capture TCP pose: "
        f"x={pose[0]:.1f}, y={pose[1]:.1f}, z={pose[2]:.1f}, "
        f"roll={pose[3]:.1f}, pitch={pose[4]:.1f}, yaw={pose[5]:.1f}"
    )
    return pose


def is_position_safe(x, y, z, safe_range):
    return (
        float(safe_range["x_min"]) <= x <= float(safe_range["x_max"])
        and float(safe_range["y_min"]) <= y <= float(safe_range["y_max"])
        and float(safe_range["z_min"]) <= z <= float(safe_range["z_max"])
    )


def require_arm_ok(arm, step_name, code=0):
    if code not in (0, None):
        raise RuntimeError(f"xArm command failed during {step_name}, code={code}")
    error_code = getattr(arm, "error_code", 0)
    if error_code != 0:
        raise RuntimeError(f"xArm error after {step_name}: error_code={error_code}")


def get_wrist_grasp_offset_deg(config):
    grasp_cfg = config.get("grasp", {})
    side = str(grasp_cfg.get("grab_from_side", "long")).strip().lower()
    side_offsets = {
        "long": 90.0,
        "long_side": 90.0,
        "short": 0.0,
        "width": 0.0,
    }
    if side not in side_offsets:
        raise ValueError(
            "grasp.grab_from_side must be 'long' or 'short'. "
            f"Got: {grasp_cfg.get('grab_from_side')!r}"
        )
    return side_offsets[side] + float(grasp_cfg.get("wrist_yaw_offset_deg", 0.0))


def get_place_yaw_deg(config):
    return float(config.get("grasp", {}).get("place_yaw_deg", 0.0))


def nearest_equivalent_yaw(yaw_deg, reference_yaw_deg):
    yaw = normalize_angle(float(yaw_deg))
    reference = float(reference_yaw_deg)
    while yaw - reference > 90.0:
        yaw -= 180.0
    while yaw - reference < -90.0:
        yaw += 180.0
    return yaw


def initialize_arm(config):
    robot_cfg = config["robot"]
    gripper_cfg = config["gripper"]
    arm = XArmAPI(robot_cfg["ip"], is_radian=False)

    arm.motion_enable(enable=True)
    arm.clean_error()
    enter_position_mode(arm)
    arm.set_collision_sensitivity(int(robot_cfg.get("collision_sensitivity", 3)))
    arm.set_tcp_load(
        float(robot_cfg.get("tcp_load_kg", 0.82)),
        robot_cfg.get("tcp_load_center_mm", [0.0, 0.0, 48.0]),
    )

    arm.set_gripper_mode(0)
    arm.set_gripper_enable(True)
    arm.set_gripper_speed(int(gripper_cfg["speed"]))
    require_arm_ok(arm, "initialization")
    return arm


def execute_pick_and_place(arm, config, pick_xyz_mm, place_xyz_mm, yaw_deg, place_yaw_deg=None):
    robot_cfg = config["robot"]
    gripper_cfg = config["gripper"]
    pick_yaw_deg = float(yaw_deg)
    if place_yaw_deg is None:
        place_yaw_deg = pick_yaw_deg
    else:
        place_yaw_deg = float(place_yaw_deg)

    safe_z = float(robot_cfg["safe_z_mm"])
    move_speed = float(robot_cfg["move_speed"])
    approach_speed = float(robot_cfg["approach_speed"])
    gripper_open = int(gripper_cfg["open_position"])
    gripper_close = int(gripper_cfg["close_position"])

    pick_x, pick_y, pick_z = [float(value) for value in pick_xyz_mm]
    place_x, place_y, place_z = [float(value) for value in place_xyz_mm]

    print("\n[Step 5] Execute real robot motion")
    print("  Motion 1/9: open gripper")
    code = arm.set_gripper_position(gripper_open, wait=True)
    require_arm_ok(arm, "open gripper", code)

    print("  Motion 2/9: move to safe height")
    code = arm.set_position(z=safe_z, speed=move_speed, wait=True)
    require_arm_ok(arm, "safe height", code)

    print("  Motion 3/9: move above the block")
    code = arm.set_position(
        x=pick_x,
        y=pick_y,
        z=safe_z,
        roll=180,
        pitch=0,
        yaw=pick_yaw_deg,
        speed=move_speed,
        wait=True,
    )
    require_arm_ok(arm, "move above block", code)

    print("  Motion 4/9: descend to pick height")
    code = arm.set_position(z=pick_z, speed=approach_speed, wait=True)
    require_arm_ok(arm, "descend to pick", code)

    print("  Motion 5/9: close gripper")
    code = arm.set_gripper_position(gripper_close, wait=True)
    time.sleep(0.5)
    require_arm_ok(arm, "close gripper", code)

    print("  Motion 6/9: lift")
    code = arm.set_position(z=safe_z, speed=move_speed, wait=True)
    require_arm_ok(arm, "lift", code)

    print("  Motion 7/9: move above place position")
    code = arm.set_position(
        x=place_x,
        y=place_y,
        z=safe_z,
        roll=180,
        pitch=0,
        yaw=place_yaw_deg,
        speed=move_speed,
        wait=True,
    )
    require_arm_ok(arm, "move above place", code)

    print("  Motion 8/9: descend to place height")
    code = arm.set_position(z=place_z, speed=approach_speed, wait=True)
    require_arm_ok(arm, "descend to place", code)

    print("  Motion 9/9: open gripper")
    code = arm.set_gripper_position(gripper_open, wait=True)
    time.sleep(0.3)
    require_arm_ok(arm, "release", code)

    print("  Finish: return to safe height")
    code = arm.set_position(z=safe_z, speed=move_speed, wait=True)
    require_arm_ok(arm, "finish", code)


def main():
    parser = argparse.ArgumentParser(description="Pick one Jenga block with RealSense and xArm.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.json")
    parser.add_argument("--yes", action="store_true", help="Skip the final motion confirmation")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    config = load_json(config_path)

    print("=" * 60)
    print("  Real Simple Jenga Grab")
    print("=" * 60)
    print(f"  Config: {config_path}")
    print(f"  Robot IP: {config['robot']['ip']}")
    print(f"  Calibration setup: {config['calibration'].get('setup', 'fixed_camera')}")

    eye_in_hand = is_eye_in_hand_setup(config)
    arm = None
    capture_pose = None

    try:
        if eye_in_hand:
            tcp_from_camera_transform(config)
            arm = initialize_arm(config)
            if config["grasp"].get("drag_to_view_before_capture", True):
                capture_pose = drag_camera_to_view(arm, config)
            else:
                capture_pose = get_arm_position_mm_deg(arm)

        print("\n[Step 1] Capture one aligned RealSense frame")
        color, depth, intrinsics, depth_scale_m = capture_realsense_frame(config)
        if eye_in_hand and capture_pose is None:
            capture_pose = get_arm_position_mm_deg(arm)
        print(f"  Color frame: {color.shape}")
        print(f"  Depth frame: {depth.shape}, dtype={depth.dtype}, scale={depth_scale_m:.6f} m/unit")
        print(f"  Intrinsics: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")

        debug_dir = resolve_path(config_dir, config["grasp"].get("debug_dir", "grab_debug"))
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "color.png"), color)
        cv2.imwrite(str(debug_dir / "depth_raw.png"), depth)

        print("\n[Step 2] Detect the best Jenga block")
        pipeline = VisionPipeline(intrinsics, depth_scale_m=depth_scale_m, config=config["vision"])
        best = pipeline.get_best(color, depth)
        if best is None:
            if pipeline.last_mask is not None:
                cv2.imwrite(str(debug_dir / "mask.png"), pipeline.last_mask)
            failure_debug = draw_detection(color, None)
            cv2.putText(
                failure_debug,
                "No valid Jenga block detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(debug_dir / "detection.png"), failure_debug)
            print("  No valid Jenga block detected. Stop.")
            return 1

        cv2.imwrite(str(debug_dir / "mask.png"), pipeline.last_mask)
        cv2.imwrite(str(debug_dir / "detection.png"), draw_detection(color, best))

        u, v = best["center_uv"]
        cam_x, cam_y, cam_z = best["xyz_cam"]
        print(f"  Pixel center: ({u}, {v})")
        print(f"  Camera XYZ: x={cam_x:.4f} m, y={cam_y:.4f} m, z={cam_z:.4f} m")
        print(f"  Block long-axis angle: {best['angle_deg']:.1f} deg")
        print(f"  Vision yaw: {best['gripper_yaw_deg']:.1f} deg")
        print(f"  Score: {best['score']:.3f}")

        print("\n[Step 3] Transform camera coordinates to robot coordinates")
        pick_z_offset = float(config["grasp"].get("pick_z_offset_mm", 0.0))
        wrist_grasp_offset = get_wrist_grasp_offset_deg(config)
        if eye_in_hand:
            robot_xyz = camera_to_robot_eye_in_hand_mm(best["xyz_cam"], capture_pose, config)
            # Sign convention matches stack_grab.compute_pick (subtract vision yaw).
            robot_yaw = normalize_angle(
                capture_pose[5]
                - best["gripper_yaw_deg"]
                + float(config["calibration"].get("yaw_offset_deg", 0.0))
                + wrist_grasp_offset
            )
            print("  Transform: T_base_from_tcp_current @ T_tcp_from_camera @ point_camera")
            print(
                "  Capture TCP pose: "
                f"x={capture_pose[0]:.1f}, y={capture_pose[1]:.1f}, z={capture_pose[2]:.1f}, "
                f"roll={capture_pose[3]:.1f}, pitch={capture_pose[4]:.1f}, yaw={capture_pose[5]:.1f}"
            )
        else:
            calibration, calibration_file = load_calibration(config, config_dir)
            robot_xyz = camera_to_robot_mm(best["xyz_cam"], calibration)
            # Sign convention matches stack_grab.compute_pick (subtract vision yaw).
            robot_yaw = normalize_angle(
                -best["gripper_yaw_deg"] + float(calibration.get("yaw_offset_deg", 0.0))
                + wrist_grasp_offset
            )
            print(f"  Calibration: {calibration_file}")

        robot_xyz[2] += pick_z_offset
        print(f"  Pick XYZ: x={robot_xyz[0]:.1f}, y={robot_xyz[1]:.1f}, z={robot_xyz[2]:.1f} mm")
        print(f"  Pick Z offset from config: {pick_z_offset:.1f} mm")
        print(f"  Wrist long-side offset: {wrist_grasp_offset:.1f} deg")
        print(f"  Robot yaw: {robot_yaw:.1f} deg")

        print("\n[Step 4] Safety check")
        safe_range = config["safe_range"]
        place_xyz = config["grasp"]["place_xyz_mm"]
        if not is_position_safe(robot_xyz[0], robot_xyz[1], robot_xyz[2], safe_range):
            print("  Pick target is outside the configured safe range. Stop.")
            return 1
        if not is_position_safe(place_xyz[0], place_xyz[1], place_xyz[2], safe_range):
            print("  Place target is outside the configured safe range. Stop.")
            return 1
        print("  Safe range check passed.")

        print("\n[Confirmation]")
        print(f"  Robot will move to pick: ({robot_xyz[0]:.1f}, {robot_xyz[1]:.1f}, {robot_xyz[2]:.1f}) mm")
        print(f"  Robot will move to place: ({place_xyz[0]:.1f}, {place_xyz[1]:.1f}, {place_xyz[2]:.1f}) mm")
        confirm_before_motion = bool(config["grasp"].get("confirm_before_motion", True))
        if confirm_before_motion and not args.yes:
            answer = input("  Type y to execute real robot motion: ").strip().lower()
            if answer != "y":
                print("  Cancelled by user.")
                return 0

        if arm is None:
            arm = initialize_arm(config)
        execute_pick_and_place(arm, config, robot_xyz, place_xyz, robot_yaw)
    finally:
        if arm is not None:
            arm.disconnect()

    print("\nDone. One real Jenga block pick-and-place cycle completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
