"""
Continuous Jenga stacking runner.

Press g to scan, grab one block, and place it on the stack.
Press q to quit.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import cv2

from simple_jenga_grab import (
    DEFAULT_CONFIG,
    VisionPipeline,
    camera_to_robot_eye_in_hand_mm,
    camera_to_robot_mm,
    capture_realsense_frame,
    draw_detection,
    enter_position_mode,
    get_arm_position_mm_deg,
    get_wrist_grasp_offset_deg,
    initialize_arm,
    is_eye_in_hand_setup,
    is_position_safe,
    load_calibration,
    load_json,
    require_arm_ok,
    resolve_path,
    tcp_from_camera_transform,
)
from vision_pipeline import normalize_angle


HOME_WRIST_YAW_DEG = 0.0
DEFAULT_BLOCK_LENGTH_MM = 75.0
DEFAULT_BLOCK_WIDTH_MM = 25.0
DEFAULT_BLOCK_HEIGHT_MM = 15.0


def yaw_delta_deg(actual_yaw, target_yaw):
    return (float(actual_yaw) - float(target_yaw) + 180.0) % 360.0 - 180.0


def get_scan_pose(config):
    pose = config["grasp"].get(
        "initial_scan_pose_mm_deg",
        [400.0, -100.0, 150.0, 180.0, 0.0, HOME_WRIST_YAW_DEG],
    )
    if len(pose) != 6:
        raise ValueError("grasp.initial_scan_pose_mm_deg must contain 6 numbers.")
    pose = [float(value) for value in pose]
    pose[5] = HOME_WRIST_YAW_DEG
    return pose


def move_to_scan_pose(arm, config):
    pose = get_scan_pose(config)
    safe_range = config["safe_range"]
    if not is_position_safe(pose[0], pose[1], pose[2], safe_range):
        raise RuntimeError(
            "Initial scan pose is outside the configured safe range: "
            f"({pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f})"
        )

    robot_cfg = config["robot"]
    print(
        "  Move to scan pose: "
        f"x={pose[0]:.1f}, y={pose[1]:.1f}, z={pose[2]:.1f}, "
        f"roll={pose[3]:.1f}, pitch={pose[4]:.1f}, yaw={pose[5]:.1f}"
    )
    code = arm.set_position(
        x=pose[0],
        y=pose[1],
        z=pose[2],
        roll=pose[3],
        pitch=pose[4],
        yaw=pose[5],
        speed=float(robot_cfg["move_speed"]),
        wait=True,
    )
    require_arm_ok(arm, "move to scan pose", code)

    actual_pose = get_arm_position_mm_deg(arm)
    yaw_error = yaw_delta_deg(actual_pose[5], pose[5])
    yaw_tolerance = float(config["grasp"].get("scan_yaw_tolerance_deg", 3.0))
    print(
        "  Actual scan pose: "
        f"x={actual_pose[0]:.1f}, y={actual_pose[1]:.1f}, z={actual_pose[2]:.1f}, "
        f"roll={actual_pose[3]:.1f}, pitch={actual_pose[4]:.1f}, yaw={actual_pose[5]:.1f}"
    )
    if abs(yaw_error) > yaw_tolerance:
        raise RuntimeError(
            "Scan pose yaw did not reach target. "
            f"target={pose[5]:.1f}, actual={actual_pose[5]:.1f}, "
            f"error={yaw_error:.1f}, tolerance={yaw_tolerance:.1f}"
        )
    return pose


def write_debug_images(debug_dir, index, color, depth, pipeline, best):
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / "color.png"), color)
    cv2.imwrite(str(debug_dir / "depth_raw.png"), depth)
    cv2.imwrite(str(debug_dir / f"color_{index:03d}.png"), color)
    cv2.imwrite(str(debug_dir / f"depth_raw_{index:03d}.png"), depth)

    if pipeline.last_mask is not None:
        cv2.imwrite(str(debug_dir / "mask.png"), pipeline.last_mask)
        cv2.imwrite(str(debug_dir / f"mask_{index:03d}.png"), pipeline.last_mask)

    detection = draw_detection(color, best)
    if best is None:
        cv2.putText(
            detection,
            "No valid Jenga block detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(debug_dir / "detection.png"), detection)
    cv2.imwrite(str(debug_dir / f"detection_{index:03d}.png"), detection)


def detect_block(config, config_dir, debug_index):
    print("  Capture RealSense frame")
    color, depth, intrinsics, depth_scale_m = capture_realsense_frame(config)

    print("  Detect best Jenga block")
    pipeline = VisionPipeline(intrinsics, depth_scale_m=depth_scale_m, config=config["vision"])
    best = pipeline.get_best(color, depth)

    debug_dir = resolve_path(config_dir, config["grasp"].get("debug_dir", "grab_debug"))
    write_debug_images(debug_dir, debug_index, color, depth, pipeline, best)
    if best is not None:
        cam_x, cam_y, cam_z = best["xyz_cam"]
        print(
            "  Detection: "
            f"score={best['score']:.3f}, "
            f"camera_xyz=({cam_x:.4f}, {cam_y:.4f}, {cam_z:.4f}) m, "
            f"vision_yaw={best['gripper_yaw_deg']:.1f}"
        )
    return best


def compute_pick(config, config_dir, best, capture_pose, verbose=True):
    pick_z_offset = float(config["grasp"].get("pick_z_offset_mm", 0.0))
    wrist_grasp_offset = get_wrist_grasp_offset_deg(config)
    table_z = float(config["grasp"].get("table_z_mm", config["grasp"]["place_xyz_mm"][2]))

    if is_eye_in_hand_setup(config):
        robot_xyz = camera_to_robot_eye_in_hand_mm(best["xyz_cam"], capture_pose, config)
        yaw_offset = float(config["calibration"].get("yaw_offset_deg", 0.0))
        robot_yaw = normalize_angle(
            capture_pose[5]
            - best["gripper_yaw_deg"]
            + yaw_offset
            + wrist_grasp_offset
        )
    else:
        calibration, _ = load_calibration(config, config_dir)
        robot_xyz = camera_to_robot_mm(best["xyz_cam"], calibration)
        yaw_offset = float(calibration.get("yaw_offset_deg", 0.0))
        robot_yaw = normalize_angle(
            -best["gripper_yaw_deg"]
            + yaw_offset
            + wrist_grasp_offset
        )

    robot_xyz_before_offset = [float(value) for value in robot_xyz]
    robot_xyz[2] += pick_z_offset
    if not verbose:
        return robot_xyz, robot_yaw
    print(
        "  Pick Z: "
        f"transformed_z={robot_xyz_before_offset[2]:.1f} mm, "
        f"pick_z_offset={pick_z_offset:.1f} mm, "
        f"final_z={robot_xyz[2]:.1f} mm, "
        f"table_z={table_z:.1f} mm, "
        f"above_table={robot_xyz[2] - table_z:.1f} mm"
    )
    if robot_xyz[2] < table_z - 5.0:
        print("  Warning: pick Z is below the configured table/home Z. Check camera Z offset or tilt.")
    print(
        "  Rotation: "
        f"vision_yaw={best['gripper_yaw_deg']:.1f}, "
        f"capture_yaw={capture_pose[5]:.1f}, "
        f"yaw_offset={yaw_offset:.1f}, "
        f"wrist_offset={wrist_grasp_offset:.1f}, "
        f"pick_yaw={robot_yaw:.1f}"
    )
    return robot_xyz, robot_yaw


def resolve_place_yaw(config):
    grasp_cfg = config.get("grasp", {})
    return float(grasp_cfg.get("place_yaw_deg", HOME_WRIST_YAW_DEG))


def get_block_size_mm(config):
    grasp_cfg = config.get("grasp", {})
    return (
        float(grasp_cfg.get("block_length_mm", DEFAULT_BLOCK_LENGTH_MM)),
        float(grasp_cfg.get("block_width_mm", DEFAULT_BLOCK_WIDTH_MM)),
        float(grasp_cfg.get("block_height_mm", DEFAULT_BLOCK_HEIGHT_MM)),
    )


def compute_stack_place(config, stack_count):
    base_xyz = [float(value) for value in config["grasp"]["place_xyz_mm"]]
    _, block_width, block_height = get_block_size_mm(config)
    layer_index = int(stack_count) // 2
    slot_index = int(stack_count) % 2

    layer_yaw = resolve_place_yaw(config) + 90.0 * (layer_index % 2)
    side_offset = block_width if slot_index == 1 else -block_width
    side_angle = math.radians(layer_yaw)

    place_xyz = [
        base_xyz[0] + side_offset * math.cos(side_angle),
        base_xyz[1] + side_offset * math.sin(side_angle),
        base_xyz[2] + layer_index * block_height,
    ]
    return place_xyz, layer_yaw, layer_index + 1, slot_index + 1


def get_stack_travel_z(config, pick_xyz, place_xyz):
    safe_range = config["safe_range"]
    safe_z = float(config["robot"]["safe_z_mm"])
    clearance = float(config["grasp"].get("stack_clearance_mm", 45.0))
    travel_z = max(safe_z, float(pick_xyz[2]) + clearance, float(place_xyz[2]) + clearance)
    if travel_z > float(safe_range["z_max"]):
        raise RuntimeError(
            "Not enough Z range for stack clearance. "
            f"travel_z={travel_z:.1f}, z_max={float(safe_range['z_max']):.1f}. "
            "Lower the stack, reduce grasp.stack_clearance_mm, or raise safe_range.z_max."
        )
    return travel_z


def print_stack_path_debug(config, stack_count, pick_xyz, place_xyz, travel_z):
    base_z = float(config["grasp"]["place_xyz_mm"][2])
    _, _, block_height = get_block_size_mm(config)
    completed_layers = int(stack_count) // 2
    stack_top_z = base_z + completed_layers * block_height
    clearance = travel_z - max(stack_top_z, float(place_xyz[2]), float(pick_xyz[2]))
    print(
        "  Path debug: "
        f"stack_top_z={stack_top_z:.1f}, travel_z={travel_z:.1f}, "
        f"clearance={clearance:.1f}"
    )


def execute_stack_pick_and_place(arm, config, pick_xyz_mm, place_xyz_mm, pick_yaw_deg, place_yaw_deg):
    robot_cfg = config["robot"]
    gripper_cfg = config["gripper"]
    move_speed = float(robot_cfg["move_speed"])
    approach_speed = float(robot_cfg["approach_speed"])
    gripper_open = int(gripper_cfg["open_position"])
    gripper_close = int(gripper_cfg["close_position"])

    pick_x, pick_y, pick_z = [float(value) for value in pick_xyz_mm]
    place_x, place_y, place_z = [float(value) for value in place_xyz_mm]
    travel_z = get_stack_travel_z(config, pick_xyz_mm, place_xyz_mm)

    print("\n[Motion] Execute smooth stack motion")
    print(f"  Travel Z: {travel_z:.1f} mm")

    print("  Motion 1/9: open gripper")
    code = arm.set_gripper_position(gripper_open, wait=True)
    require_arm_ok(arm, "open gripper", code)

    print("  Motion 2/9: move to travel height")
    code = arm.set_position(z=travel_z, speed=move_speed, wait=True)
    require_arm_ok(arm, "travel height", code)

    print("  Motion 3/9: move above the block")
    code = arm.set_position(
        x=pick_x,
        y=pick_y,
        z=travel_z,
        roll=180,
        pitch=0,
        yaw=float(pick_yaw_deg),
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
    code = arm.set_position(z=travel_z, speed=move_speed, wait=True)
    require_arm_ok(arm, "lift", code)

    print("  Motion 7/9: move above stack")
    code = arm.set_position(
        x=place_x,
        y=place_y,
        z=travel_z,
        roll=180,
        pitch=0,
        yaw=float(place_yaw_deg),
        speed=move_speed,
        wait=True,
    )
    require_arm_ok(arm, "move above stack", code)

    print("  Motion 8/9: descend to stack height")
    code = arm.set_position(z=place_z, speed=approach_speed, wait=True)
    require_arm_ok(arm, "descend to stack", code)

    print("  Motion 9/9: open gripper")
    code = arm.set_gripper_position(gripper_open, wait=True)
    time.sleep(0.3)
    require_arm_ok(arm, "release", code)

    print("  Finish: lift above stack")
    code = arm.set_position(z=travel_z, speed=move_speed, wait=True)
    require_arm_ok(arm, "finish lift", code)


def stack_once(arm, config, config_dir, stack_count, debug_index):
    move_to_scan_pose(arm, config)
    capture_pose = get_arm_position_mm_deg(arm)

    best = detect_block(config, config_dir, debug_index)
    if best is None:
        print("  No valid Jenga block detected. Try again.")
        return False

    robot_xyz, pick_yaw = compute_pick(config, config_dir, best, capture_pose)
    place_xyz, place_yaw, layer_number, slot_number = compute_stack_place(config, stack_count)

    safe_range = config["safe_range"]
    if not is_position_safe(robot_xyz[0], robot_xyz[1], robot_xyz[2], safe_range):
        print("  Pick target is outside the configured safe range. Skip.")
        return False
    if not is_position_safe(place_xyz[0], place_xyz[1], place_xyz[2], safe_range):
        print("  Place target is outside the configured safe range. Skip.")
        return False

    travel_z = get_stack_travel_z(config, robot_xyz, place_xyz)
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

    execute_stack_pick_and_place(arm, config, robot_xyz, place_xyz, pick_yaw, place_yaw)
    move_to_scan_pose(arm, config)
    return True


def main():
    parser = argparse.ArgumentParser(description="Continuously stack Jenga blocks with xArm.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.json")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    config = load_json(config_path)

    print("=" * 60)
    print("  Continuous Jenga Stack Grab")
    print("=" * 60)
    print(f"  Config: {config_path}")
    print("  Press g to grab/place one block. Press q to quit.")

    if is_eye_in_hand_setup(config):
        tcp_from_camera_transform(config)

    arm = initialize_arm(config)
    stack_count = 0
    debug_index = 1
    try:
        enter_position_mode(arm)
        move_to_scan_pose(arm, config)
        while True:
            command = input("\n[g] grab next block, [q] quit: ").strip().lower()
            if command == "q":
                print("  Quit requested.")
                break
            if command != "g":
                print("  Type g or q.")
                continue

            success = stack_once(arm, config, config_dir, stack_count, debug_index)
            debug_index += 1
            if success:
                stack_count += 1
                print(f"  Stack count: {stack_count}")
    finally:
        arm.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
