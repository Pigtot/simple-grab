# Live Stack Grab Student Guide

This guide explains what happens inside `live_stack_grab.py`. It is written for
new students who are learning how the RealSense camera, OpenCV vision code, and
xArm motion code work together.

## What This Program Does

`live_stack_grab.py` is a live-preview version of the Jenga stacking script.

It opens the RealSense camera, shows a live OpenCV window, detects the best
visible Jenga block, and waits for the operator to press `g`. When `g` is
pressed, the robot uses the current camera frame to pick that block and place it
onto the growing stack.

Controls:

- `g`: grab the currently detected block and place it on the stack.
- `q`: quit.
- `Esc`: quit.

Run example:

```powershell
.\.venv\Scripts\python.exe live_stack_grab.py --config config2.json
```

## Big Picture Flow

The program follows this order:

1. Load the JSON config.
2. Prepare a higher-resolution live camera config.
3. Connect to the xArm.
4. Move the robot to the scan pose.
5. Start the RealSense stream.
6. Show a live preview with detection overlays.
7. When `g` is pressed, compute the pick and place targets.
8. Run the pick-and-place motion.
9. Return to the scan pose and continue the live preview.
10. On quit, stop the camera, close the window, and disconnect the arm.

## Camera Preview

The script uses the `RealSenseLiveStream` class to keep the RealSense camera
open continuously. This is different from `stack_grab.py`, which captures one
new frame only when the user types a command.

By default, the live script requests:

```text
1280 x 720
```

This gives the preview more pixels than the normal config camera size. You can
change it from the command line:

```powershell
.\.venv\Scripts\python.exe live_stack_grab.py --config config2.json --width 640 --height 480
```

The preview window shows:

- the camera image,
- the best block contour,
- the block center,
- detection score,
- depth,
- suggested gripper yaw,
- current stack count.

## Vision Detection

Each frame is passed into `VisionPipeline` from `vision_pipeline.py`.

The pipeline:

1. segments possible block-shaped objects,
2. filters contours by area, aspect ratio, and depth,
3. estimates the center pixel,
4. converts the center pixel plus depth into a 3D camera point,
5. estimates a gripper yaw angle,
6. returns the best candidate.

Because the live preview uses a larger image, `prepare_live_config()` scales the
pixel-based vision settings. Without this, a good block could be rejected just
because it appears larger in a higher-resolution frame.

## Pressing `g`

When the operator presses `g`, the script does not capture a totally separate
frame. It uses the latest live frame and latest detection result already shown
in the preview window.

The function `stack_current_detection()` handles this step.

It does the following:

1. Checks that a valid block was detected.
2. Reads the current robot pose.
3. Saves debug images into `grab_debug`.
4. Converts the detected camera point into robot coordinates.
5. Computes the next stack location.
6. Checks that pick and place positions are inside `safe_range`.
7. Chooses a safe travel speed.
8. Runs the stack pick-and-place motion.
9. Moves the robot back to the scan pose.

## Coordinate Transform

The camera reports the block position in camera coordinates, measured in meters.
The robot needs a target in robot base coordinates, measured in millimeters.

For the eye-in-hand setup, the script uses:

```text
robot point = current robot pose * camera mount transform * camera point
```

That work is done through functions imported from `simple_jenga_grab.py` and
`stack_grab.py`, especially `compute_pick()`.

The important idea is that the RealSense camera is mounted on the robot wrist,
so the camera position changes whenever the robot moves.

## Stack Placement

The next place target is computed by `compute_stack_place()` from
`stack_grab.py`.

The script places two blocks per layer:

- slot 1 goes to one side of the stack center,
- slot 2 goes to the other side,
- the next layer moves up by one block height,
- each new layer rotates by 90 degrees.

The stack count starts at `0` and increases after every successful grab.

## Motion Speeds

The live script can move faster when it is safe.

There are two important speed types:

- `move_speed`: used for high, open travel moves.
- `approach_speed`: used when descending to pick or place.

The script may increase only the travel speed. It keeps the careful moments
before pick and place constant by leaving `approach_speed` unchanged.

Default live fast-speed settings:

```text
fast travel speed: 180 mm/s
minimum clearance: 35 mm
```

Example:

```powershell
.\.venv\Scripts\python.exe live_stack_grab.py --config config2.json --fast-move-speed 220 --fast-speed-min-clearance-mm 45
```

If there is not enough clearance, the script uses the normal configured
`move_speed`.

## Safety Checks

Before the robot moves to pick or place, the script checks both targets against
the configured `safe_range`.

If a target is outside the allowed range, the grab is skipped.

The script also computes a travel height that is above:

- the pick position,
- the place position,
- the configured safe travel height,
- the current stack height.

If there is not enough Z range for this clearance, the script raises an error
instead of trying the motion.

## Debug Images

Every time `g` is pressed and a valid detection exists, the script writes debug
images into the configured debug directory, usually `grab_debug`.

Useful files include:

- `color.png`: latest color frame,
- `depth_raw.png`: latest depth frame,
- `mask.png`: segmentation mask,
- `detection.png`: color frame with detection overlay,
- numbered copies such as `color_001.png` and `detection_001.png`.

These images are useful when the robot chooses the wrong block or the detector
misses a block.

## Important Functions

`prepare_live_config()`

Copies the config, applies the requested live camera resolution, and scales
vision settings for the larger image.

`RealSenseLiveStream`

Starts, reads from, and stops the RealSense camera stream.

`draw_live_overlay()`

Draws the text at the top of the preview window.

`choose_live_move_speed()`

Decides whether it is safe to use the faster live travel speed.

`stack_current_detection()`

Takes the current detection and performs one full stack pick-and-place cycle.

`main()`

Parses command-line arguments, initializes the robot and camera, runs the live
preview loop, and handles cleanup.

## Student Mental Model

Think of the program as two loops working together:

```text
Live loop:
    camera frame -> vision detection -> preview window

Grab event:
    current detection -> robot target -> safety check -> pick -> place
```

The robot only moves when the operator presses `g`. Until then, the program is
just watching the camera and updating the preview.

## Before Running With Hardware

Check these first:

- The emergency stop is nearby.
- The RealSense camera is connected.
- The robot IP in the config is correct.
- The eye-in-hand calibration values are verified.
- `safe_range` is conservative.
- The scan pose is safe.
- The gripper open and close positions are correct for the block.
- Start with modest speeds before increasing `--fast-move-speed`.

## Config And Value Flow

This section maps the important `config2.json` values to the code that uses
them. JSON files cannot safely contain comments, so this guide is the commented
version of the config.

## Main Tools Used By The Script

The live stack script connects several tools:

- `pyrealsense2`: starts the Intel RealSense depth and color streams.
- `OpenCV / cv2`: displays the live preview, draws overlays, and saves debug
  images.
- `numpy`: performs vector and matrix math.
- `xArm Python SDK`: connects to the robot and sends motion/gripper commands.
- `vision_pipeline.py`: finds a block in the color/depth frame.
- `simple_jenga_grab.py`: provides RealSense helpers, xArm helpers, and camera
  calibration transforms.
- `stack_grab.py`: provides stack placement math and pick-and-place motion.

## Full Runtime Flow With Config Values

Here is the whole flow in the order it happens.

1. `main()` loads `config2.json`.

   Important values:

   ```json
   "robot": { "ip": "192.168.1.217" }
   ```

   This tells the xArm SDK which robot to connect to.

2. `prepare_live_config()` prepares the live camera settings.

   Original config values:

   ```json
   "camera": {
     "width": 640,
     "height": 480,
     "fps": 30,
     "warmup_frames": 30
   }
   ```

   Live defaults from `live_stack_grab.py`:

   ```text
   width  = 1280
   height = 720
   fps    = config camera.fps
   ```

   Because the live image is larger than the config image, the script scales
   pixel-based vision values such as `edge_margin_px`, `min_area_px`,
   `max_long_px`, and `max_short_px`.

3. The script checks the eye-in-hand calibration.

   Important values:

   ```json
   "calibration": {
     "setup": "eye_in_hand_manual",
     "yaw_offset_deg": 0.0,
     "eye_in_hand_manual": {
       "verified": true,
       "camera_origin_in_tcp_mm": [67.05, -31.14, -137.61],
       "camera_axes_to_tcp_rpy_deg": [-0.2408, -0.4862, 91.0937]
     }
   }
   ```

   If `verified` is not `true`, the script stops. This prevents using an
   untrusted camera-to-tool transform.

4. `initialize_arm()` connects to the xArm.

   Important values:

   ```json
   "robot": {
     "ip": "192.168.1.217",
     "collision_sensitivity": 3,
     "tcp_load_kg": 0.82,
     "tcp_load_center_mm": [0.0, 0.0, 48.0]
   }
   ```

   These configure robot connection, collision sensitivity, and the gripper/tool
   load.

5. The gripper is configured.

   Important values:

   ```json
   "gripper": {
     "open_position": 840,
     "close_position": 600,
     "speed": 3000
   }
   ```

   `open_position` is used before picking and after placing.
   `close_position` is used to hold the block.

6. The robot moves to the scan pose.

   Important value:

   ```json
   "initial_scan_pose_mm_deg": [300.0, -50.0, 150.0, 180.0, 0.0, 0.0]
   ```

   Format:

   ```text
   [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]
   ```

   The script checks this pose against `safe_range` before moving.

7. The RealSense stream starts.

   The script reads aligned color and depth frames. Aligned means the depth
   image pixels match the color image pixels.

8. `VisionPipeline` detects the best block.

   Important values:

   ```json
   "vision": {
     "min_score": 0.3,
     "min_depth_m": 0.2,
     "max_depth_m": 1.0,
     "edge_margin_px": 15,
     "min_area_px": 500
   }
   ```

   These values reject weak detections, objects too close/far from the camera,
   objects near the image edge, and objects too small in pixels.

9. The preview window shows the current detection.

   The preview uses:

   ```python
   draw_detection()
   draw_live_overlay()
   cv2.imshow()
   ```

   The robot does not move during normal preview. It only moves after `g`.

10. When `g` is pressed, the script freezes the latest frame and detection.

    It uses the detection already visible in the preview, not a separate new
    still capture.

11. The script reads the current robot pose.

    This becomes `capture_pose`:

    ```text
    [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]
    ```

    This is required because the camera is mounted on the wrist, so camera
    coordinates depend on the robot pose at the moment of detection.

12. The script transforms camera XYZ into robot XYZ.

    This is the most important math in the program.

    The vision pipeline gives:

    ```text
    point_camera_m = [camera_x_m, camera_y_m, camera_z_m]
    ```

    The code converts meters to millimeters:

    ```text
    point_camera_mm = point_camera_m * 1000
    ```

    Then it applies:

    ```text
    point_base = T_base_from_tcp_current @ T_tcp_from_camera @ point_camera
    ```

    Meaning:

    - `point_camera`: where the block is in RealSense camera coordinates.
    - `T_tcp_from_camera`: where the camera is mounted relative to the gripper
      TCP.
    - `T_base_from_tcp_current`: where the gripper TCP is in robot base
      coordinates right now.
    - `point_base`: where the robot should move in robot base coordinates.

## Eye-In-Hand Transform Math

These two config values define the RealSense mount:

```json
"camera_origin_in_tcp_mm": [67.05, -31.14, -137.61],
"camera_axes_to_tcp_rpy_deg": [-0.2408, -0.4862, 91.0937]
```

`camera_origin_in_tcp_mm` is the camera optical origin measured in the xArm TCP
frame.

For your value:

```text
camera_x_in_tcp =  67.05 mm
camera_y_in_tcp = -31.14 mm
camera_z_in_tcp = -137.61 mm
```

`camera_axes_to_tcp_rpy_deg` describes how the RealSense camera axes are rotated
relative to the TCP axes.

For your value:

```text
roll  = -0.2408 deg
pitch = -0.4862 deg
yaw   = 91.0937 deg
```

The code converts that RPY into a rotation matrix using:

```python
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
```

Then `make_transform()` combines the rotation and translation into:

```text
T_tcp_from_camera =

[ R00 R01 R02 camera_x_in_tcp ]
[ R10 R11 R12 camera_y_in_tcp ]
[ R20 R21 R22 camera_z_in_tcp ]
[ 0   0   0   1               ]
```

This transform is reused every time the robot converts a detected block from
camera coordinates into robot coordinates.

## Pick Target Math

After the camera-to-robot conversion, `compute_pick()` applies the configured
pick Z offset:

```json
"pick_z_offset_mm": 0.0
```

Formula:

```text
pick_z = transformed_block_z + pick_z_offset_mm
```

If the robot is grabbing too high or too low, this is one of the first values to
check.

The pick yaw is computed separately from XYZ:

```text
pick_yaw = capture_pose_yaw - vision_gripper_yaw + yaw_offset + wrist_grasp_offset
```

Where:

- `capture_pose_yaw`: current robot wrist yaw.
- `vision_gripper_yaw`: block angle detected in the camera image.
- `yaw_offset_deg`: calibration yaw correction.
- `wrist_grasp_offset`: depends on `grab_from_side`.

Important config values:

```json
"yaw_offset_deg": 0.0,
"grab_from_side": "long",
"wrist_yaw_offset_deg": 0.0
```

In this code:

```text
grab_from_side = "long"  -> wrist_grasp_offset starts at 90 deg
grab_from_side = "short" -> wrist_grasp_offset starts at 0 deg
```

## Place Target Math

The base stack position comes from:

```json
"place_xyz_mm": [250.0, -50.0, -107.5]
```

The block dimensions come from:

```json
"block_length_mm": 75.0,
"block_width_mm": 25.0,
"block_height_mm": 15.0
```

The script places two blocks per layer.

Formulas:

```text
layer_index = stack_count // 2
slot_index  = stack_count % 2

layer_yaw = place_yaw_deg + 90 * (layer_index % 2)

side_offset = -block_width_mm for slot 1
side_offset = +block_width_mm for slot 2

place_x = base_x + side_offset * cos(layer_yaw)
place_y = base_y + side_offset * sin(layer_yaw)
place_z = base_z + layer_index * block_height_mm
```

If `place_yaw_deg` is not present in the config, the code defaults it to `0`.

## Safe Range

Before moving, both pick and place targets must pass this check:

```json
"safe_range": {
  "x_min": 100.0,
  "x_max": 600.0,
  "y_min": -250.0,
  "y_max": 250.0,
  "z_min": -111.0,
  "z_max": 300.0
}
```

The code checks:

```text
x_min <= x <= x_max
y_min <= y <= y_max
z_min <= z <= z_max
```

If either target is outside this box, the robot skips the grab.

## Travel Height And Speed

The robot does not move directly from pick to place at table height. It first
moves up to a travel height.

Important values:

```json
"robot": {
  "safe_z_mm": 150.0,
  "move_speed": 100,
  "approach_speed": 50
}
```

Optional value:

```json
"stack_clearance_mm": 45.0
```

If `stack_clearance_mm` is not in the config, the code uses `45.0`.

Formula:

```text
travel_z = max(
    safe_z_mm,
    pick_z + stack_clearance_mm,
    place_z + stack_clearance_mm
)
```

Then the script checks:

```text
travel_z <= safe_range.z_max
```

Speed behavior:

- High travel moves use `move_speed`, or the faster live speed if clearance is
  high enough.
- Descending into the block and descending onto the stack always use
  `approach_speed`.

This keeps the careful moments before pick and place constant.

## Motion Sequence

`execute_stack_pick_and_place()` runs the physical motion:

1. Open gripper.
2. Move to travel height.
3. Move above the block.
4. Descend to pick height using `approach_speed`.
5. Close gripper.
6. Lift to travel height.
7. Move above the stack.
8. Descend to stack height using `approach_speed`.
9. Open gripper.
10. Lift back to travel height.

After that, `live_stack_grab.py` moves the arm back to the scan pose and resumes
the camera preview.

## Debug Output Flow

When `g` is pressed and a detection exists, the script writes:

```text
grab_debug/color.png
grab_debug/depth_raw.png
grab_debug/mask.png
grab_debug/detection.png
grab_debug/color_001.png
grab_debug/depth_raw_001.png
grab_debug/mask_001.png
grab_debug/detection_001.png
```

Use these files to answer questions like:

- Did the camera see the block clearly?
- Did the mask isolate the correct object?
- Was the detection center reasonable?
- Was the block angle reasonable?

## Config Values To Tune First

If the robot misses the block in X/Y:

- Check `camera_origin_in_tcp_mm`.
- Check `camera_axes_to_tcp_rpy_deg`.
- Check the camera mount has not moved.

If the robot is too high or low at pick:

- Tune `pick_z_offset_mm`.
- Check `camera_origin_in_tcp_mm`, especially the Z value.

If the wrist angle is wrong:

- Tune `yaw_offset_deg`.
- Check `grab_from_side`.
- Tune `wrist_yaw_offset_deg`.

If the stack placement is wrong:

- Tune `place_xyz_mm`.
- Check `block_width_mm` and `block_height_mm`.
- Add or tune `place_yaw_deg`.

If detection is unstable:

- Tune `vision.min_score`.
- Tune `vision.edge_margin_px`.
- Tune `vision.min_area_px`.
- Check lighting and debug masks.

If motion is too slow:

- Increase `--fast-move-speed`.
- Keep `approach_speed` conservative.

If motion is too close to the stack:

- Increase `stack_clearance_mm`.
- Make sure `safe_range.z_max` allows the higher travel height.
