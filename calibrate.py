"""
Real-hardware camera-to-robot calibration for the simple Jenga grab demo.

This script collects corresponding point pairs:
    - robot point in the xArm base frame, in millimeters
    - detected Jenga marker point in the RealSense camera frame, in meters

It then solves:
    robot_mm = rotation_matrix @ camera_mm + camera_position_mm

The generated calibration.json is used directly by simple_jenga_grab.py.
There is no simulation mode in this script.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI

from vision_pipeline import VisionPipeline, draw_detection


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


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


def get_arm_position_mm_deg(arm):
    try:
        code, pose = arm.get_position(is_radian=False)
    except TypeError:
        code, pose = arm.get_position()
    if code != 0:
        raise RuntimeError(f"Failed to read xArm position, code={code}")
    if len(pose) < 3:
        raise RuntimeError(f"xArm returned an invalid pose: {pose}")
    return [float(value) for value in pose]


def start_realsense(config):
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
    started = False
    try:
        profile = rs_pipeline.start(rs_config)
        started = True
        align = rs.align(rs.stream.color)

        for _ in range(int(camera_cfg.get("warmup_frames", 30))):
            rs_pipeline.wait_for_frames()

        frames = rs_pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        if not color_frame:
            raise RuntimeError("RealSense did not return a color frame.")

        intrinsics = get_realsense_intrinsics(color_frame)
        depth_scale_m = profile.get_device().first_depth_sensor().get_depth_scale()
        return rs_pipeline, align, intrinsics, float(depth_scale_m)
    except Exception:
        if started:
            rs_pipeline.stop()
        raise


def capture_aligned_frame(rs_pipeline, align):
    frames = rs_pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        raise RuntimeError("RealSense did not return both color and depth frames.")
    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())
    return color, depth


def corrected_robot_point_from_pose(robot_pose, offset_mm):
    tcp_xyz = np.array(robot_pose[:3], dtype=float)
    offset = np.array(offset_mm, dtype=float)
    return tcp_xyz + offset


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


def is_xyz_safe(xyz, safe_range):
    x, y, z = [float(value) for value in xyz]
    return (
        float(safe_range["x_min"]) <= x <= float(safe_range["x_max"])
        and float(safe_range["y_min"]) <= y <= float(safe_range["y_max"])
        and float(safe_range["z_min"]) <= z <= float(safe_range["z_max"])
    )


def format_pose(pose):
    return (
        f"x={pose[0]:.1f}, y={pose[1]:.1f}, z={pose[2]:.1f}, "
        f"roll={pose[3]:.1f}, pitch={pose[4]:.1f}, yaw={pose[5]:.1f}"
    )


def print_jog_help(step_mm, angle_step_deg, speed):
    print("""
  Jog commands:
    x+ / x-          move TCP along robot base X
    y+ / y-          move TCP along robot base Y
    z+ / z-          move TCP along robot base Z
    roll+ / roll-    rotate tool roll
    pitch+ / pitch-  rotate tool pitch
    yaw+ / yaw-      rotate tool yaw
    goto X Y Z       move to an absolute XYZ in mm, keeping current RPY
    safez            move current XYZ to robot.safe_z_mm
    rpy              apply calibration.jog_roll_pitch_yaw_deg if configured
    step N           set translation step in mm
    angle N          set rotation step in degrees
    speed N          set jog speed
    p                print current pose
    c                capture this calibration point
    q                finish point collection
    h                show this help
""")
    print(f"  Current jog settings: step={step_mm:g} mm, angle={angle_step_deg:g} deg, speed={speed:g}")


def parse_float(text, label):
    try:
        return float(text)
    except ValueError:
        print(f"  Invalid {label}: {text}")
        return None


def command_arm_pose(arm, target_pose, speed, safe_range, action):
    target_xyz = target_pose[:3]
    if not is_xyz_safe(target_xyz, safe_range):
        print(
            "  Refusing move outside safe_range: "
            f"({target_xyz[0]:.1f}, {target_xyz[1]:.1f}, {target_xyz[2]:.1f})"
        )
        return get_arm_position_mm_deg(arm)

    code = arm.set_position(
        x=float(target_pose[0]),
        y=float(target_pose[1]),
        z=float(target_pose[2]),
        roll=float(target_pose[3]),
        pitch=float(target_pose[4]),
        yaw=float(target_pose[5]),
        speed=float(speed),
        is_radian=False,
        wait=True,
    )
    require_xarm_ok(arm, code, action)
    return get_arm_position_mm_deg(arm)


def drag_to_reference_position(arm, config):
    calibration_cfg = config["calibration"]
    safe_range = config["safe_range"]
    sensitivity = calibration_cfg.get("teach_sensitivity", 3)
    settle_seconds = float(calibration_cfg.get("drag_settle_seconds", 0.5))

    print("""
  Drag mode:
    1. The script will put xArm into joint teaching mode.
    2. Drag the arm by hand until the TCP is at the calibration reference point.
    3. Release the arm and let it settle.
    4. Press Enter or type c to capture this point.

  Commands:
    Enter / c    capture this calibration point
    p            print current pose
    q            finish point collection
    h            show this help
""")

    enter_teach_mode(arm, sensitivity=sensitivity)
    try:
        while True:
            pose = get_arm_position_mm_deg(arm)
            print(f"\n  Current pose: {format_pose(pose)}")
            command = input("  drag> ").strip().lower()

            if command in {"", "c", "capture"}:
                enter_position_mode(arm)
                time.sleep(settle_seconds)
                pose = get_arm_position_mm_deg(arm)
                if not is_xyz_safe(pose[:3], safe_range):
                    print(
                        "  Current TCP is outside safe_range after capture: "
                        f"({pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f})"
                    )
                    answer = input("  Use this point anyway? [y/N]: ").strip().lower()
                    if answer != "y":
                        enter_teach_mode(arm, sensitivity=sensitivity)
                        continue
                return pose

            if command in {"q", "quit", "finish"}:
                enter_position_mode(arm)
                return None

            if command in {"p", "pose"}:
                continue

            if command in {"h", "help", "?"}:
                print("""
  Drag the robot by hand, then press Enter or type c to capture.
  Type p to print pose, q to finish point collection.
""")
                continue

            print("  Unknown command. Type h for help.")
    except Exception:
        enter_position_mode(arm)
        raise


def jog_to_reference_position(arm, config):
    calibration_cfg = config["calibration"]
    robot_cfg = config["robot"]
    safe_range = config["safe_range"]

    step_mm = float(calibration_cfg.get("jog_step_mm", 10.0))
    angle_step_deg = float(calibration_cfg.get("jog_angle_step_deg", 5.0))
    speed = float(calibration_cfg.get("jog_speed", 50.0))
    configured_rpy = calibration_cfg.get("jog_roll_pitch_yaw_deg")

    print_jog_help(step_mm, angle_step_deg, speed)

    while True:
        pose = get_arm_position_mm_deg(arm)
        print(f"\n  Current pose: {format_pose(pose)}")
        command = input("  jog> ").strip().lower()

        if command in {"c", "capture"}:
            return pose
        if command in {"q", "quit", "finish"}:
            return None
        if command in {"h", "help", "?"}:
            print_jog_help(step_mm, angle_step_deg, speed)
            continue
        if command in {"p", "pose"}:
            continue

        parts = command.split()
        if len(parts) == 2 and parts[0] == "step":
            value = parse_float(parts[1], "step")
            if value is None or value <= 0:
                print("  Step must be a positive number.")
                continue
            step_mm = value
            print(f"  Translation step set to {step_mm:g} mm")
            continue
        if len(parts) == 2 and parts[0] == "angle":
            value = parse_float(parts[1], "angle")
            if value is None or value <= 0:
                print("  Angle step must be a positive number.")
                continue
            angle_step_deg = value
            print(f"  Rotation step set to {angle_step_deg:g} deg")
            continue
        if len(parts) == 2 and parts[0] == "speed":
            value = parse_float(parts[1], "speed")
            if value is None or value <= 0:
                print("  Speed must be a positive number.")
                continue
            speed = value
            print(f"  Jog speed set to {speed:g}")
            continue

        target = list(pose[:6])
        if command == "x+":
            target[0] += step_mm
        elif command == "x-":
            target[0] -= step_mm
        elif command == "y+":
            target[1] += step_mm
        elif command == "y-":
            target[1] -= step_mm
        elif command == "z+":
            target[2] += step_mm
        elif command == "z-":
            target[2] -= step_mm
        elif command == "roll+":
            target[3] += angle_step_deg
        elif command == "roll-":
            target[3] -= angle_step_deg
        elif command == "pitch+":
            target[4] += angle_step_deg
        elif command == "pitch-":
            target[4] -= angle_step_deg
        elif command == "yaw+":
            target[5] += angle_step_deg
        elif command == "yaw-":
            target[5] -= angle_step_deg
        elif command == "safez":
            target[2] = float(robot_cfg["safe_z_mm"])
        elif command == "rpy":
            if configured_rpy is None:
                print("  calibration.jog_roll_pitch_yaw_deg is null; no configured RPY to apply.")
                continue
            target[3:6] = [float(value) for value in configured_rpy]
        elif len(parts) == 4 and parts[0] == "goto":
            values = [parse_float(value, "goto coordinate") for value in parts[1:4]]
            if any(value is None for value in values):
                continue
            target[0:3] = values
        else:
            print("  Unknown command. Type h for help.")
            continue

        command_arm_pose(arm, target, speed, safe_range, command)


def move_to_reference_position(arm, config):
    calibration_cfg = config["calibration"]
    mode = str(calibration_cfg.get("positioning_mode", "drag")).lower()
    if mode in {"drag", "teach", "manual"}:
        return drag_to_reference_position(arm, config)
    if mode == "jog":
        return jog_to_reference_position(arm, config)
    raise ValueError(
        "calibration.positioning_mode must be 'drag' or 'jog', "
        f"got {mode!r}"
    )


def solve_rigid_transform(camera_points_m, robot_points_mm):
    camera_points_mm = np.array(camera_points_m, dtype=float) * 1000.0
    robot_points_mm = np.array(robot_points_mm, dtype=float)

    if len(camera_points_mm) < 3:
        raise ValueError("At least 3 calibration points are required.")
    if np.linalg.matrix_rank(camera_points_mm - camera_points_mm.mean(axis=0)) < 2:
        raise ValueError("Calibration points are too close to a line. Spread them out.")

    camera_center = camera_points_mm.mean(axis=0)
    robot_center = robot_points_mm.mean(axis=0)
    camera_centered = camera_points_mm - camera_center
    robot_centered = robot_points_mm - robot_center

    h = camera_centered.T @ robot_centered
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    translation = robot_center - rotation @ camera_center
    return rotation, translation


def compute_errors(rotation, translation, camera_points_m, robot_points_mm):
    errors = []
    computed_points = []
    for cam_m, robot_mm in zip(camera_points_m, robot_points_mm):
        cam_mm = np.array(cam_m, dtype=float) * 1000.0
        actual = np.array(robot_mm, dtype=float)
        computed = rotation @ cam_mm + translation
        computed_points.append(computed)
        errors.append(float(np.linalg.norm(actual - computed)))
    return errors, computed_points


def collect_points(config, config_dir, min_points, points_path, debug_dir):
    robot_cfg = config["robot"]
    calibration_cfg = config["calibration"]
    tcp_offset = calibration_cfg.get("tcp_to_calibration_point_offset_mm", [0.0, 0.0, 0.0])

    print("=" * 62)
    print("  Real Camera-to-Robot Calibration")
    print("=" * 62)
    print(f"  Robot IP: {robot_cfg['ip']}")
    print(f"  TCP-to-calibration-point offset: {tcp_offset} mm")
    print("  Corrected robot point = current TCP XYZ + this offset.")
    print("  If the TCP is 30 mm above the real marker point, set Z offset to -30.")
    print()

    arm = None
    rs_pipeline = None
    points = []
    try:
        print("  Connecting to xArm...")
        arm = XArmAPI(robot_cfg["ip"], is_radian=False)
        require_xarm_ok(arm, arm.motion_enable(enable=True), "motion enable")
        require_xarm_ok(arm, arm.clean_error(), "clear errors")
        enter_position_mode(arm)
        require_xarm_ok(
            arm,
            arm.set_collision_sensitivity(int(robot_cfg.get("collision_sensitivity", 3))),
            "set collision sensitivity",
        )

        print("  Starting RealSense...")
        rs_pipeline, align, intrinsics, depth_scale_m = start_realsense(config)
        print(f"  RealSense intrinsics: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")
        print(f"  Depth scale: {depth_scale_m:.6f} m/unit")

        vision = VisionPipeline(intrinsics, depth_scale_m=depth_scale_m, config=config["vision"])

        while True:
            print("\n" + "-" * 62)
            print(f"  Calibration point {len(points) + 1}")
            print("  Place one Jenga block at a new workspace position.")
            print("  Move the robot TCP to the reference position.")
            robot_pose = move_to_reference_position(arm, config)
            if robot_pose is None:
                break

            corrected_robot = corrected_robot_point_from_pose(robot_pose, tcp_offset)

            color, depth = capture_aligned_frame(rs_pipeline, align)
            best = vision.get_best(color, depth)

            index = len(points)
            color_file = debug_dir / f"point_{index:03d}_color.png"
            mask_file = debug_dir / f"point_{index:03d}_mask.png"
            detection_file = debug_dir / f"point_{index:03d}_detection.png"
            cv2.imwrite(str(color_file), color)
            if vision.last_mask is not None:
                cv2.imwrite(str(mask_file), vision.last_mask)
            cv2.imwrite(str(detection_file), draw_detection(color, best))

            if best is None:
                print("  No marker block detected. Adjust lighting or position and retry.")
                print(f"  Debug image: {color_file}")
                continue

            print(f"  Raw TCP XYZ: ({robot_pose[0]:.1f}, {robot_pose[1]:.1f}, {robot_pose[2]:.1f}) mm")
            print(
                "  Corrected robot point: "
                f"({corrected_robot[0]:.1f}, {corrected_robot[1]:.1f}, {corrected_robot[2]:.1f}) mm"
            )
            cam_x, cam_y, cam_z = best["xyz_cam"]
            print(f"  Camera point: ({cam_x:.4f}, {cam_y:.4f}, {cam_z:.4f}) m")
            print(f"  Detection score: {best['score']:.3f}")
            print(f"  Debug detection: {detection_file}")

            accept = input("  Save this point? [Y/n]: ").strip().lower()
            if accept == "n":
                print("  Point discarded.")
                continue

            record = {
                "index": index,
                "robot_tcp_mm_deg": [round(float(value), 4) for value in robot_pose[:6]],
                "tcp_to_calibration_point_offset_mm": [round(float(value), 4) for value in tcp_offset],
                "robot_mm": [round(float(value), 4) for value in corrected_robot],
                "cam_m": [round(float(value), 6) for value in best["xyz_cam"]],
                "pixel_uv": [int(best["center_uv"][0]), int(best["center_uv"][1])],
                "score": float(best["score"]),
                "color_file": str(color_file),
                "detection_file": str(detection_file),
            }
            points.append(record)
            save_json(points_path, {
                "description": "Raw point pairs collected for camera-to-robot calibration.",
                "points": points,
            })
            print(f"  Saved point {len(points)} to {points_path}")

            if len(points) >= min_points:
                more = input("  Collect more points? More spread improves accuracy. [Y/n]: ").strip().lower()
                if more == "n":
                    break

    finally:
        if rs_pipeline is not None:
            rs_pipeline.stop()
        if arm is not None:
            arm.disconnect()

    return points


def main():
    parser = argparse.ArgumentParser(description="Calibrate camera-to-robot transform with real xArm and RealSense.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.json")
    parser.add_argument("--output", default="", help="Calibration output JSON. Defaults to config calibration.file")
    parser.add_argument("--points-output", default="", help="Collected point-pair JSON. Defaults to config calibration.points_file")
    parser.add_argument("--min-points", type=int, default=0, help="Minimum points before the script asks to stop")
    parser.add_argument("--debug-dir", default="", help="Debug image directory. Defaults to config calibration.debug_dir")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    config = load_json(config_path)

    calibration_cfg = config["calibration"]
    setup = str(calibration_cfg.get("setup", "fixed_camera")).lower()
    if setup in {"eye_in_hand_manual", "eye-in-hand-manual", "eye_in_hand"}:
        print(
            "This point-pair calibrate.py script is for a fixed camera setup.\n"
            "Your config is eye-in-hand manual, so do not use this script to solve "
            "rotation_matrix/camera_position_mm.\n"
            "Measure the camera mount and edit calibration.eye_in_hand_manual in "
            "config.json, then run simple_jenga_grab.py."
        )
        return 1

    output_path = resolve_path(config_dir, args.output or calibration_cfg["file"])
    points_path = resolve_path(config_dir, args.points_output or calibration_cfg["points_file"])
    debug_dir = resolve_path(config_dir, args.debug_dir or calibration_cfg["debug_dir"])
    debug_dir.mkdir(parents=True, exist_ok=True)

    min_points = args.min_points or int(calibration_cfg.get("min_points", 4))
    if min_points < 3:
        min_points = 3

    points = collect_points(config, config_dir, min_points, points_path, debug_dir)
    if len(points) < 3:
        print("\nNot enough points. Calibration was not solved.")
        return 1

    camera_points_m = [point["cam_m"] for point in points]
    robot_points_mm = [point["robot_mm"] for point in points]
    rotation, translation = solve_rigid_transform(camera_points_m, robot_points_mm)
    errors, computed_points = compute_errors(rotation, translation, camera_points_m, robot_points_mm)

    print("\n" + "=" * 62)
    print("  Calibration Result")
    print("=" * 62)
    print("  rotation_matrix:")
    for row in rotation:
        print(f"    [{row[0]: .6f}, {row[1]: .6f}, {row[2]: .6f}]")
    print(
        "  camera_position_mm: "
        f"[{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]"
    )

    print("\n  Verification errors:")
    for idx, (actual, computed, err) in enumerate(zip(robot_points_mm, computed_points, errors)):
        print(
            f"    {idx:02d}: actual=({actual[0]:.1f}, {actual[1]:.1f}, {actual[2]:.1f}) "
            f"computed=({computed[0]:.1f}, {computed[1]:.1f}, {computed[2]:.1f}) "
            f"error={err:.2f} mm"
        )
    avg_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    print(f"\n  Average error: {avg_error:.2f} mm")
    print(f"  Max error: {max_error:.2f} mm")

    yaw_default = float(calibration_cfg.get("yaw_offset_deg", 0.0))
    yaw_text = input(f"\n  Yaw offset in degrees [{yaw_default:.1f}]: ").strip()
    yaw_offset = float(yaw_text) if yaw_text else yaw_default

    output = {
        "description": "Camera-to-xArm-base transform generated by real_simple_jenga_grab/calibrate.py.",
        "rotation_matrix": [[round(float(value), 6) for value in row] for row in rotation],
        "camera_position_mm": [round(float(value), 3) for value in translation],
        "yaw_offset_deg": float(yaw_offset),
        "calibration_error_mm": round(avg_error, 3),
        "max_error_mm": round(max_error, 3),
        "num_points_used": len(points),
        "tcp_to_calibration_point_offset_mm": calibration_cfg.get(
            "tcp_to_calibration_point_offset_mm",
            [0.0, 0.0, 0.0],
        ),
        "points_file": str(points_path),
        "debug_dir": str(debug_dir),
        "test_points": [
            {
                "xyz_cam_m": point["cam_m"],
                "expected_robot_mm": point["robot_mm"],
            }
            for point in points
        ],
    }
    save_json(output_path, output)
    print(f"\nSaved calibration to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
