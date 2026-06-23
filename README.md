# Jenga Grab — RealSense + OpenCV + xArm

This project picks up and stacks Jenga blocks with real hardware. An Intel
RealSense depth camera is mounted on the wrist of a UFactory xArm (an
**eye-in-hand** setup). OpenCV finds the best block in the color/depth frame, the
camera point is transformed into the robot base frame, and the xArm drives a
gripper through a pick-and-place (or stacking) motion.

There is no simulator and no mock robot. Everything here talks to the physical
camera and arm.

The tuned, tested configuration is **`config2.json`**. `config.json` is an older
single-grab config and is ignored.

## Hardware and environment

- **Intel RealSense** depth camera (eye-in-hand, on the wrist), via `pyrealsense2`.
- **UFactory xArm** + gripper, via the `xArm-Python-SDK` (`xarm.wrapper.XArmAPI`).
- **OpenCV** (`opencv-python`/`cv2`) for vision, overlays, and debug images.
- **NumPy / SciPy** for the linear algebra.

The dependencies are already installed in the bundled virtual environment, so run
everything with that interpreter:

```powershell
.\.venv\Scripts\python.exe <script> --config config2.json
```

## File map

| File | Role |
| --- | --- |
| `vision_pipeline.py` | OpenCV block detector + pixel/depth → 3D camera point. The vision core. |
| `simple_jenga_grab.py` | Shared library (RealSense capture, xArm helpers, eye-in-hand transform) **plus** a one-shot single-grab `main()`. |
| `stack_grab.py` | Keyboard-driven continuous stacking. Imports the shared helpers. |
| `live_stack_grab.py` | **Primary entry point.** Live OpenCV preview window; press `g` to grab/stack the currently shown block. |
| `calibrate.py` | Fixed-camera point-pair calibration only. Not used in eye-in-hand mode. |
| `config2.json` | **Tuned config** used by the stacking scripts. |
| `config.json` | Legacy single-grab config. Ignored. |
| `LIVE_STACK_GRAB_README.md` | Line-by-line deep dive on `live_stack_grab.py`. |
| `grab_debug/` | Debug images written each grab (`color`, `depth_raw`, `mask`, `detection`). |

## Quick start

### Primary: live preview stacking

Opens a live RealSense window with the detection overlay. Press `g` to pick the
shown block and place it on the growing stack; press `q` or `Esc` to quit.

```powershell
.\.venv\Scripts\python.exe live_stack_grab.py --config config2.json
```

The live stream defaults to 1280×720 (more pixels than the 640×480 capture config);
`prepare_live_config()` rescales the pixel-based vision thresholds to match. You can
override the stream and speeds:

```powershell
.\.venv\Scripts\python.exe live_stack_grab.py --config config2.json --width 640 --height 480
.\.venv\Scripts\python.exe live_stack_grab.py --config config2.json --fast-move-speed 220 --fast-speed-min-clearance-mm 45
```

### Keyboard stacking (no preview window)

Captures one fresh frame each time you type `g` at the prompt. Type `q` to quit.

```powershell
.\.venv\Scripts\python.exe stack_grab.py --config config2.json
```

### Single grab (one pick-and-place cycle)

Drags the wrist camera over a block, captures one frame, picks it, and places it at
`grasp.place_xyz_mm`. Add `--yes` to skip the final confirmation prompt.

```powershell
.\.venv\Scripts\python.exe simple_jenga_grab.py --config config2.json
```

### Fixed-camera calibration (only if the camera is NOT on the wrist)

`calibrate.py` solves a fixed camera-to-base transform from point pairs. With the
eye-in-hand config it exits with a warning, so it is not part of the normal flow.

```powershell
.\.venv\Scripts\python.exe calibrate.py --config config.json
```

## How it works end to end

```text
RealSense frame ──► OpenCV detect ──► camera→robot transform ──► xArm motion
 (color + depth)     (best block)      (eye-in-hand chain)        (pick / stack)
```

1. **Capture.** Grab one aligned RealSense color + depth frame (depth pixels mapped
   onto color pixels) and read the camera intrinsics and depth scale.
2. **Detect.** `VisionPipeline` segments the frame, scores block-shaped contours,
   picks the best, and back-projects its center pixel + depth into a 3D point in the
   camera frame (meters). It also reports a suggested gripper yaw.
3. **Transform.** Because the camera rides on the wrist, the camera point is mapped
   to the robot base frame using the current TCP pose and the measured camera mount.
4. **Move.** The xArm runs a 9-step pick-and-place. The stacking scripts compute the
   next slot/layer and add a travel-height clearance between moves.

## OpenCV skills used (`vision_pipeline.py`)

The detector is built to survive uneven lighting on a plain table:

- **Shadow normalization** — convert BGR→LAB, run **CLAHE** on the L channel, then
  divide by a heavily Gaussian-blurred copy (`cv2.divide`) to flatten illumination.
- **Color contrast mask** — **Otsu** thresholds on the LAB *b* channel and the HSV
  saturation channel, combined with `bitwise_or`, to catch the yellow block color.
- **Edge mask** — **Canny** edges, `dilate`, and `morphologyEx(MORPH_CLOSE)`, then
  fill the resulting contours.
- **Watershed refinement** — `distanceTransform` + `connectedComponents` markers
  feed `cv2.watershed` to split blocks that touch.
- **Contour analysis** — `findContours`, `contourArea`, `moments` (center),
  `minAreaRect` (size + angle), `boundingRect` (ROI).
- **Robust depth** — median of valid depth in a small window around the center,
  falling back to the median inside an **eroded** contour ROI so edge/background
  pixels do not skew the reading.
- **Pinhole back-projection** — `x=(u−cx)/fx·z`, `y=(v−cy)/fy·z`, `z=depth`, turning a
  pixel + depth into a camera-frame XYZ in meters.
- **Scoring** — a weighted blend of fill ratio, aspect ratio, size, and
  yellow/saturation, each shaped with a Gaussian around the expected block geometry.
- **Yaw** — the `minAreaRect` long-axis angle, normalized to `[-90, 90)`, becomes the
  suggested gripper yaw.
- **Debug drawing** — `drawContours`, `circle`, `putText`, `rectangle`, and `imwrite`
  for the images in `grab_debug/`.

## Camera skills used (`pyrealsense2`)

- Configure an `rs.pipeline` with a **z16 depth** stream and a **bgr8 color** stream
  at the configured resolution/FPS.
- **`rs.align(rs.stream.color)`** so each depth pixel lines up with its color pixel.
- Throw away **warmup frames** for auto-exposure, with timeout + retry on
  `wait_for_frames` (`wait_for_realsense_frames`).
- Pull **intrinsics** (`fx, fy, ppx, ppy`) and the **depth scale** (meters per raw
  unit) from the running profile.
- Two capture styles: a **one-shot** `capture_realsense_frame()` (single grab / stack
  prompt) and a **continuous** `RealSenseLiveStream` class that keeps the stream open
  for the live preview.
- When the live preview runs at a higher resolution than the capture config,
  `prepare_live_config()` rescales pixel thresholds (`edge_margin_px`, `min_area_px`,
  `max_long_px`, `max_short_px`).

## Calibration skills used

### Eye-in-hand manual (the mode this project uses)

The camera moves with the robot, so a single fixed transform is not valid. Instead
the point is transformed through the chain:

```text
point_base = T_base_from_tcp_current @ T_tcp_from_camera @ point_camera
```

- `point_camera` — the block in RealSense camera coordinates (converted m→mm).
- `T_tcp_from_camera` — the **measured** camera mount: `camera_origin_in_tcp_mm`
  (translation) plus `camera_axes_to_tcp_rpy_deg` (rotation, built with
  `Rz(yaw) @ Ry(pitch) @ Rx(roll)`). Set `verified: true` once measured or the script
  refuses to run.
- `T_base_from_tcp_current` — the live TCP pose read at the moment of capture.

The pick yaw is computed separately:

```text
pick_yaw = capture_yaw − vision_yaw + yaw_offset_deg + wrist_grasp_offset
```

where `wrist_grasp_offset` is 90° for `grab_from_side: "long"` (0° for `"short"`)
plus `wrist_yaw_offset_deg`. *(Note: `simple_jenga_grab.py` was aligned to this same
`− vision_yaw` convention so all three runners agree with `stack_grab.compute_pick`.)*

### Fixed-camera point pairs (`calibrate.py`, secondary)

For a world-fixed camera, `calibrate.py` collects corresponding
`(camera_m, robot_mm)` pairs while you drag/jog the TCP to a marker, then solves a
rigid transform `robot = R @ camera + t` with an **SVD / Kabsch** fit
(`solve_rigid_transform`) and reports per-point and average errors. It writes
`calibration.json`, which the fixed-camera path of `simple_jenga_grab.py` consumes.

## Robot / motion skills used (xArm SDK)

- Connect with `XArmAPI`, `motion_enable`, `clean_error`, and switch between
  **position mode** (`set_mode(0)`) and **joint teaching mode** (`set_mode(2)`).
- `set_collision_sensitivity` and `set_tcp_load` (gripper mass + center) for safety.
- Gripper control: `set_gripper_mode/enable/speed/position`.
- Cartesian moves via `set_position(x, y, z, roll, pitch, yaw, speed, wait)`; pose
  readback via `get_position`.
- **Drag-to-teach** posing — switch to teach mode, let the operator move the arm by
  hand, then lock the pose and read it (used to aim the wrist camera and to collect
  calibration points).
- **Safe-range box check** — every pick/place target must fall inside `safe_range`
  before any motion.
- **Travel clearance** — the stacking scripts lift to a computed travel height above
  the pick, the place, the stack top, and `robot.safe_z_mm` before moving across.
- **Stacking math** (`compute_stack_place`) — two blocks per layer (offset left/right
  by `block_width_mm`), each new layer raised by `block_height_mm` and rotated 90°.

## `config2.json` reference

| Section | Key fields | Notes |
| --- | --- | --- |
| `robot` | `ip`, `safe_z_mm`, `move_speed`, `approach_speed`, `collision_sensitivity`, `tcp_load_kg`, `tcp_load_center_mm` | `approach_speed` is the slow descent into pick/place. |
| `gripper` | `open_position`, `close_position`, `speed` | Position units are the xArm gripper scale. |
| `camera` | `width`, `height`, `fps`, `warmup_frames` | Capture size (640×480); the live script can upscale. |
| `vision` | `min_score`, `min_depth_m`, `max_depth_m`, `edge_margin_px`, `min_area_px` | Reject weak, too near/far, edge, or tiny detections. |
| `calibration` | `setup: eye_in_hand_manual`, `yaw_offset_deg`, `eye_in_hand_manual.{verified, camera_origin_in_tcp_mm, camera_axes_to_tcp_rpy_deg}` | The wrist-camera mount. `verified` must be `true`. |
| `grasp` | `place_xyz_mm` (stack base), `initial_scan_pose_mm_deg`, `place_yaw_deg`, `grab_from_side`, `pick_z_offset_mm`, `drag_to_view_before_capture` | `initial_scan_pose_mm_deg` is `[x, y, z, roll, pitch, yaw]`. |
| `safe_range` | `x/y/z_min`, `x/y/z_max` | The allowed motion box. |

Notes:

- `config2.json` omits the block dimensions, so the code uses defaults
  **length 75 / width 25 / height 15 mm** (`get_block_size_mm`).
- `grasp.stack_step_z_mm` is present in `config2.json` but **not read** by the code —
  the per-layer height step comes from `block_height_mm`. Treat it as informational.

## Safety checklist

- Keep the emergency stop within reach.
- Confirm `safe_range` and `initial_scan_pose_mm_deg` before any motion.
- Keep `eye_in_hand_manual.verified: true` only when the mount values are correct and
  the camera has not moved.
- Start at modest speeds; raise `--fast-move-speed` only after the basics work.
- Check `grab_debug/` if the robot misses or chooses the wrong block.

## Tuning

For a full, line-by-line walkthrough of the live runner and a config-tuning guide
(what to change when X/Y, Z, yaw, stacking, detection, or speed look wrong), see
[LIVE_STACK_GRAB_README.md](LIVE_STACK_GRAB_README.md).
