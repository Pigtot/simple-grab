"""
Continuous Jenga stacking with a responsive RealSense preview.

Automatically picks eligible blocks. Space pauses after the current cycle;
g requests one pick while paused; q/Esc stops motion and exits.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI

from simple_jenga_grab import (
    VisionPipeline,
    get_arm_position_mm_deg,
    get_realsense_intrinsics,
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
    get_block_size_mm,
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
DEFAULT_CONFIG = Path(__file__).resolve().with_name("config2.json")
OVERLAY_BAR_PX = 28

PICKABLE = "pickable"
# Status -> (BGR outline/text color, short label). "target" is the block the
# next automatic pick will use; the rest explain why a block is not chosen.
LABEL_STYLES = {
    "target": ((0, 220, 0), "TARGET"),
    PICKABLE: ((0, 200, 255), "pickable"),
    "stack_zone": ((255, 140, 0), "stack zone"),
    "out_of_range": ((0, 0, 255), "out of reach"),
    "invalid": ((150, 150, 150), "no 3D"),
}


class MotionStopped(RuntimeError):
    pass


class CameraStartupError(RuntimeError):
    """Native camera startup failed before the arm was connected."""


def report_camera_startup_error(error):
    print(f"\nCamera startup failed: {error}", file=sys.stderr, flush=True)
    print(
        "The robot was not connected or moved.\n"
        "Close other camera applications, unplug the D435 for 10 seconds,\n"
        "and reconnect directly to a USB 3 port. Then test with:\n"
        "  sudo .venv/bin/python live_stack_grab.py --camera-only\n"
        "If it still fails, add --camera-debug to show the native USB error.",
        file=sys.stderr, flush=True,
    )


class InterruptibleArm:
    """Prevent a stopped worker from issuing any subsequent SDK command."""

    def __init__(self, arm, stop_event):
        self.arm = arm
        self.stop_event = stop_event

    def __getattr__(self, name):
        value = getattr(self.arm, name)
        if not callable(value):
            return value

        def checked(*args, **kwargs):
            if self.stop_event.is_set():
                raise MotionStopped("Motion stopped by operator")
            result = value(*args, **kwargs)
            if self.stop_event.is_set():
                raise MotionStopped("Motion stopped by operator")
            return result

        return checked


def initialize_live_arm(config):
    """Configure the arm without automatically clearing controller faults."""
    arm = XArmAPI(config["robot"]["ip"], is_radian=False)
    try:
        code, errors = arm.get_err_warn_code()
        require_arm_ok(arm, "read controller status", code)
        if errors[0]:
            raise RuntimeError(f"Controller fault {errors[0]}; resolve it before restarting.")
        robot, gripper = config["robot"], config["gripper"]
        # Configure collision protection and payload before enabling motion.
        commands = (
            (arm.set_collision_sensitivity, (int(robot.get("collision_sensitivity", 3)),)),
            (arm.set_tcp_load, (float(robot.get("tcp_load_kg", 0.82)),
                                robot.get("tcp_load_center_mm", [0.0, 0.0, 48.0]))),
            (arm.motion_enable, (True,)),
            (arm.set_mode, (0,)),
            (arm.set_state, (0,)),
            (arm.set_gripper_mode, (0,)),
            (arm.set_gripper_enable, (True,)),
            (arm.set_gripper_speed, (int(gripper["speed"]),)),
        )
        for command, arguments in commands:
            require_arm_ok(arm, command.__name__, command(*arguments))
        return arm
    except BaseException:
        arm.disconnect()
        raise


def classify_candidates(candidates, config, config_dir, capture_pose):
    """Return (candidate, status) for every detection, keeping score order."""
    length, width, _ = get_block_size_mm(config)
    half_extent = max(length / 2.0, 1.5 * width) + 10.0
    stack_x, stack_y, _ = config["grasp"]["place_xyz_mm"]
    labeled = []
    for candidate in candidates:
        xyz, yaw = compute_pick(config, config_dir, candidate, capture_pose, verbose=False)
        if not np.isfinite([*xyz, yaw]).all():
            status = "invalid"
        elif not is_position_safe(*xyz, config["safe_range"]):
            status = "out_of_range"
        elif abs(xyz[0] - stack_x) <= half_extent and abs(xyz[1] - stack_y) <= half_extent:
            status = "stack_zone"
        else:
            status = PICKABLE
        labeled.append((candidate, status))
    return labeled


def first_pickable(labeled):
    return next((candidate for candidate, status in labeled if status == PICKABLE), None)


def select_pickable_candidate(candidates, config, config_dir, capture_pose):
    """Choose the best in-range block outside the reserved stack footprint."""
    return first_pickable(classify_candidates(candidates, config, config_dir, capture_pose))


def _draw_tag(image, text, origin, color):
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    height, width = image.shape[:2]
    x = int(np.clip(origin[0], 2, max(width - text_w - 4, 2)))
    y = int(np.clip(origin[1], OVERLAY_BAR_PX + text_h + 6, max(height - baseline - 2, 0)))
    cv2.rectangle(image, (x - 2, y - text_h - 4), (x + text_w + 2, y + baseline + 2), (20, 20, 20), -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_labeled_candidates(color_image, labeled, target=None):
    """Outline and tag every detected block; the pick target is drawn last, on top."""
    output = color_image.copy()
    ordered = sorted(enumerate(labeled, 1), key=lambda item: item[1][0] is target)
    for rank, (candidate, status) in ordered:
        style = "target" if candidate is target else status
        bgr, text = LABEL_STYLES.get(style, LABEL_STYLES["invalid"])
        u, v = (int(value) for value in candidate["center_uv"])
        tag_origin = (u + 8, v - 8)
        contour = candidate.get("contour")
        if contour is not None:
            cv2.drawContours(output, [contour], -1, bgr, 3 if style == "target" else 2)
            box_x, box_y, _, _ = cv2.boundingRect(contour)
            tag_origin = (box_x, box_y - 10)  # above the block, not over its outline
        cv2.circle(output, (u, v), 4, bgr, -1)
        label = f"#{rank} {text} {candidate['score']:.2f}"
        if style == "target":
            label += f" z={candidate['depth_m']:.3f}m yaw={candidate['gripper_yaw_deg']:.0f}"
        _draw_tag(output, label, tag_origin, bgr)
    return output


class DetectionGate:
    """Require a short run of consistent detections, using the newest frame."""

    def __init__(self, frames=2):
        self.frames = frames
        self.reset()

    def reset(self):
        self.previous = None
        self.count = 0

    def update(self, candidate):
        if candidate is None:
            self.reset()
            return False
        same = False
        if self.previous is not None:
            distance = np.linalg.norm(np.subtract(candidate["xyz_cam"], self.previous["xyz_cam"]))
            yaw_delta = abs((candidate["gripper_yaw_deg"] - self.previous["gripper_yaw_deg"] + 90) % 180 - 90)
            same = distance <= 0.01 and yaw_delta <= 10.0
        self.count = self.count + 1 if same else 1
        self.previous = candidate
        return self.count >= self.frames


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
        try:
            self._open_and_warm_up()
        except CameraStartupError:
            raise
        except RuntimeError as exc:
            # A D435 left mid-stream by a previous run (common on macOS) opens
            # but never sends frames. A hardware reset clears it.
            print(f"  {exc}\n  Resetting the camera once and retrying...")
            device = self.profile.get_device()
            self.stop()
            device.hardware_reset()
            time.sleep(3.0)  # the camera drops off USB and re-enumerates
            self._open_and_warm_up(start_attempts=10)

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

    def _open_and_warm_up(self, start_attempts=1):
        for attempt in range(1, start_attempts + 1):
            try:
                self.profile = self.pipeline.start(self.rs_config)
                break
            except RuntimeError as exc:
                if attempt == start_attempts:
                    raise CameraStartupError(str(exc)) from exc
                time.sleep(1.0)
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

    def read(self, include_frame=False):
        frame_timeout_ms = int(self.camera_cfg.get("frame_timeout_ms", 5000))
        frame_retries = int(self.camera_cfg.get("frame_retries", 3))
        frames = wait_for_realsense_frames(
            self.pipeline,
            frame_timeout_ms,
            frame_retries,
            "preview",
        )
        return self._decode(frames, include_frame)

    def poll(self):
        frames = self.pipeline.poll_for_frames()
        return self._decode(frames) if frames else None

    def _decode(self, frames, include_frame=False):
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


def side_by_side_mask(preview, mask):
    """Put the segmentation mask (black = blank while not processing) right of the preview."""
    if mask is None or mask.shape[:2] != preview.shape[:2]:
        mask_bgr = np.zeros_like(preview)
    else:
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return np.hstack((preview, mask_bgr))


def draw_live_overlay(preview, best, stack_count, busy=False, automatic=True, message=None):
    """One thin status line; key help is printed in the terminal at startup."""
    if best is None:
        status, color = "No valid Jenga block detected", (0, 0, 255)
    else:
        status, color = "Ready", (0, 200, 0)
    if busy:
        status = "Robot is moving..." if automatic else "Pausing after this cycle..."
        color = (0, 180, 255)
    if message:
        status = message

    text = f"{'AUTO' if automatic else 'PAUSED'} | Stack {stack_count} | {status}"
    cv2.rectangle(preview, (0, 0), (preview.shape[1], OVERLAY_BAR_PX), (20, 20, 20), -1)
    cv2.putText(preview, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
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
    save_debug=False,
):
    if best is None:
        print("  No eligible block in this frame; continue watching.")
        return False

    capture_pose = get_arm_position_mm_deg(arm)
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
        print(f"  Pick target {robot_xyz} is outside the configured workspace. Skip.")
        return False
    if not is_position_safe(place_xyz[0], place_xyz[1], place_xyz[2], safe_range):
        raise RuntimeError(f"Stack target {place_xyz} exceeds the configured workspace.")

    travel_z = get_stack_travel_z(config, robot_xyz, place_xyz)
    if save_debug:
        write_debug_images(
            resolve_path(config_dir, config["grasp"].get("debug_dir", "grab_debug")),
            debug_index, color, depth, pipeline, best,
        )
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


def run_preview(arm, stream, pipeline, config, config_dir, args):
    """Keep camera/UI on the main thread and serialize complete robot cycles."""
    stop_event = Event()
    guarded_arm = InterruptibleArm(arm, stop_event)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stack-motion")
    future = None
    automatic = not args.manual
    stack_count = args.stack_count
    gate = DetectionGate(args.stable_frames)
    phase = "scan"
    fault = None
    settle_until = float("inf")
    capture_pose = None
    last_frame_at = time.monotonic()
    preview = np.zeros((config["camera"]["height"], 2 * config["camera"]["width"], 3), dtype=np.uint8)
    best = None
    labeled = []
    mask = None
    stop_sent = False

    def stop_motion():
        nonlocal stop_sent
        stop_event.set()
        if not stop_sent:
            stop_sent = True
            code = arm.set_state(4)
            if code not in (0, None):
                print(f"  Stop command failed ({code}); use the physical emergency stop.")

    try:
        future = executor.submit(move_to_scan_pose, guarded_arm, config)
        while True:
            now = time.monotonic()
            if future is not None and future.done():
                try:
                    success = future.result()
                    if phase == "grab" and success:
                        stack_count += 1
                        print(f"  Stack count: {stack_count}")
                    capture_pose = get_arm_position_mm_deg(arm)
                    require_arm_ok(arm, "cycle complete")
                    settle_until = time.monotonic() + args.settle_seconds
                except Exception as exc:
                    fault = str(exc)
                    automatic = False
                    stop_motion()
                    print(f"  STOPPED: {fault}. Resolve the cause, then restart; no automatic retry.")
                future = None
                gate.reset()
                best = None
                labeled = []

            frame = stream.poll()
            ready = False
            if frame is not None:
                color, depth = frame
                last_frame_at = time.monotonic()
                best = None
                labeled = []
                mask = None
                if future is None and fault is None and now >= settle_until:
                    candidates = pipeline.process(color, depth)
                    mask = pipeline.last_mask
                    labeled = classify_candidates(candidates, config, config_dir, capture_pose)
                    best = first_pickable(labeled)
                    ready = gate.update(best)
                else:
                    gate.reset()
                preview = side_by_side_mask(draw_labeled_candidates(color, labeled, best), mask)
            elif now - last_frame_at > float(config["camera"].get("frame_timeout_ms", 5000)) / 1000.0:
                raise RuntimeError("Camera stopped delivering frames")

            message = None
            if fault:
                message = "FAULT: " + fault[:110] + " | q: exit"
            elif future is not None and phase == "scan":
                message = "Moving to scan pose..."
            elif future is None and now < settle_until:
                message = "Refreshing camera view..."
            elif best is None and future is None:
                message = (f"{len(labeled)} block(s) seen, none pickable outside the stack area"
                           if labeled else "Watching for a pickable block outside the stack area...")
            cv2.imshow(WINDOW_NAME, draw_live_overlay(
                preview.copy(), best, stack_count, busy=future is not None,
                automatic=automatic, message=message,
            ))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("  Quit requested.")
                break
            if key == ord(" ") and fault is None:
                automatic = not automatic
                gate.reset()
                ready = False
                print("  Automatic stacking enabled." if automatic else "  Paused after current cycle.")

            single_pick = key == ord("g") and not automatic
            if (future is None and fault is None and frame is not None and best is not None
                    and now >= settle_until and ((automatic and ready) or single_pick)):
                print(f"\n[Grab {stack_count + 1}] Use current live frame")
                # The worker receives a snapshot; the UI continues draining frames
                # while the wrist moves, and those frames never trigger a pick.
                snapshot = SimpleNamespace(last_mask=(pipeline.last_mask.copy()
                    if args.save_debug and pipeline.last_mask is not None else None))
                phase = "grab"
                future = executor.submit(
                    stack_current_detection, guarded_arm, config, config_dir,
                    stack_count, stack_count + 1, color, depth, snapshot, best,
                    args.fast_move_speed, args.fast_speed_min_clearance_mm, args.save_debug,
                )
                gate.reset()
        return 1 if fault else 0
    finally:
        # Set the guard before stopping the controller so a worker cannot advance
        # to the next move after an interrupted blocking SDK call returns.
        try:
            stop_motion()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(
        description="Continuously stack Jenga blocks with live RealSense preview.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Config path (default: config2.json)")
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_LIVE_WIDTH,
        help="Live RealSense stream width. Use 640 to match config2.json.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_LIVE_HEIGHT,
        help="Live RealSense stream height. Use 480 to match config2.json.",
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
    parser.add_argument("--manual", action="store_true", help="Start paused; g picks once, Space enables auto.")
    parser.add_argument("--stable-frames", type=int, default=2, help="Consistent frames before an automatic pick (default: 2).")
    parser.add_argument("--settle-seconds", type=float, default=0.3, help="Fresh-frame settling time after each cycle.")
    parser.add_argument("--stack-count", type=int, default=0, help="Number of blocks already placed in the stack.")
    parser.add_argument("--save-debug", action="store_true", help="Save images for each pick; off by default.")
    parser.add_argument("--camera-only", action="store_true", help="Verify color/depth frames, then exit without connecting to the arm.")
    parser.add_argument("--camera-debug", action="store_true", help="Print native RealSense driver diagnostics.")
    args = parser.parse_args()
    if args.stable_frames < 1 or args.stack_count < 0:
        parser.error("stable-frames must be positive and stack-count must be nonnegative")
    if not np.isfinite(args.settle_seconds) or args.settle_seconds < 0:
        parser.error("settle-seconds must be finite and nonnegative")
    if any(not np.isfinite(value) or value <= 0 for value in
           (args.fast_move_speed, args.fast_speed_min_clearance_mm, args.preview_scale)):
        parser.error("speed, clearance, and preview-scale must be finite and positive")

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
    print("  Mode: " + ("camera test only (no robot connection)" if args.camera_only else
                        "manual / paused" if args.manual else "continuous automatic stacking"))
    if not args.camera_only:
        print("  Space: pause after current cycle / resume; g: one pick while paused; q/Esc: stop and quit.")

    if not args.camera_only and is_eye_in_hand_setup(config):
        tcp_from_camera_transform(config)
    if args.camera_debug:
        rs.log_to_console(rs.log_severity.debug)

    arm = None
    stream = None
    try:
        # Prove the camera works before enabling or moving the robot.
        stream = RealSenseLiveStream(config)
        stream.start()
        if args.camera_only:
            color, depth = stream.read()
            valid_depth = int(np.count_nonzero(depth))
            if valid_depth == 0:
                raise RuntimeError("Camera streams opened but the depth frame contains no valid depth pixels")
            print(f"  Camera test PASSED: color={color.shape}, depth={depth.shape}, valid depth pixels={valid_depth}")
            return 0
        arm = initialize_live_arm(config)
        pipeline = VisionPipeline(
            stream.intrinsics,
            depth_scale_m=stream.depth_scale_m,
            config=config["vision"],
        )
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        # The preview is two frames wide; keep its aspect at one frame's width on screen.
        cv2.resizeWindow(
            WINDOW_NAME,
            int(stream.intrinsics["width"] * args.preview_scale),
            int(stream.intrinsics["height"] * args.preview_scale / 2),
        )
        return run_preview(arm, stream, pipeline, config, config_dir, args)
    finally:
        try:
            if stream is not None:
                stream.stop()
        finally:
            cv2.destroyAllWindows()
            if arm is not None:
                arm.disconnect()


if __name__ == "__main__":
    try:
        exit_code = main()
    except CameraStartupError as exc:
        report_camera_startup_error(exc)
        # This macOS native driver has crashed during destruction after a failed
        # USB power-state transition. No arm has been connected on this path.
        # Keep the exception/traceback alive and let the OS release USB handles.
        if sys.platform == "darwin" and "failed to set power state" in str(exc).lower():
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        exit_code = 1
    sys.exit(exit_code)
