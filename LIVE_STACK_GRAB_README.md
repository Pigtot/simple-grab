# Continuous Live Stack Grab

`live_stack_grab.py` automatically picks eligible Jenga blocks and places them
on a growing stack. It defaults to `config2.json`. A single motion worker runs
one complete cycle at a time while the main thread keeps the RealSense preview
and keyboard controls responsive.

## Run

```bash
.venv/bin/python live_stack_grab.py
```

On this Mac, the RealSense driver required administrator access:

```bash
sudo .venv/bin/python live_stack_grab.py
```

If startup reports `failed to set power state` even with `sudo`, close other
camera applications, unplug the D435 for 10 seconds, and reconnect it directly
to a USB 3 port. Verify camera access separately before restarting stacking:

```bash
sudo .venv/bin/python live_stack_grab.py --camera-only
```

Run only one camera session at a time. If startup says `UVC device is already
opened`, quit the existing preview with `q`/Esc or stop the old Terminal run
before starting another instance or a camera-only test. A failed partial startup
can also retain a camera stream until its process exits.

This opens the color/depth streams, checks for valid depth pixels, and exits
without connecting to or moving the arm. Add `--camera-debug` if it still fails.
On macOS camera-startup failures, the command exits without running native-driver
destructors, which have crashed or left streaming threads during shutdown. After
a hardware reset, startup retries only a missing device while USB re-enumerates;
it does not retry busy or partially opened pipelines. This
error handling does not repair a USB connection or reset the device.

Startup checks the camera, connects to the arm without clearing existing
controller faults, configures the payload and collision sensitivity, and moves
to the scan pose. Automatic picking starts as soon as eligible detections are
consistent across two frames. Use `--manual` to start paused; startup still
moves the arm to the scan pose.

## Controls

| Key | Action |
| --- | --- |
| Space | Pause after the current cycle, or resume automatic picking |
| `g` | Pick once while paused |
| `q` / Esc / close preview | Send controller stop, cancel subsequent moves, and exit |

Space lets the current pick/place/return cycle finish. Quit sends a controller
stop immediately when the UI processes the key, including during a motion.
The hardware emergency stop remains the way to stop if the application or
network is unresponsive.

## Selecting blocks

The detector produces candidates ranked by score. Each candidate is transformed
to robot coordinates. The runner chooses the highest-scoring candidate inside
`safe_range` and outside the reserved stack footprint. This allows a lower-ranked
reachable block to be picked when the top candidate is out of range.

The reserved footprint is centered at `grasp.place_xyz_mm`. Its half-width in
both X and Y is `max(block_length_mm / 2, 1.5 * block_width_mm) + 10` mm. It covers
both orientations of the two-block layers and prevents the runner from picking
its own stack. Put source blocks outside this area.

Automatic picks need two consecutive detections within 10 mm and 10 degrees of
each other. No detection means keep watching. The score, depth, and geometry
thresholds still come from `config2.json`; the runner does not lower them.

During motion, the preview continues reading frames but does not select targets.
After returning to the scan pose, it drains frames for 0.3 seconds, resets the
detection history, and selects from fresh frames. The scan pose is read again
after each completed cycle for the eye-in-hand coordinate transform.

## Motion and faults

The shared stack motion in `stack_grab.py` still opens the gripper, lifts,
approaches the block, closes the gripper, lifts, approaches the stack, releases,
and lifts again. The arm then returns to the scan pose.

Travel speed defaults to 180 mm/s when the existing clearance check permits it;
approach speed comes from the config. Workspace boundaries, stack-height
clearance, collision sensitivity, and command-result checks remain active.

A controller fault or a full stack stops automatic operation. The preview shows
the fault, and Space cannot resume it. Resolve the cause and restart. The runner
never automatically clears errors or retries an interrupted pick. Controller
error 31 is reported by the SDK as “Collision Caused Abnormal Current.” Check the
physical setup, payload, and path before restarting after this fault.

If a cycle is interrupted after releasing a block, inspect the stack and supply
the actual count with `--stack-count` on restart. The counter advances only after
a complete pick/place/return cycle, and it is not persisted between runs.

## Options

| Option | Default | Purpose |
| --- | --- | --- |
| `--config` | `config2.json` beside the script | Robot, camera, calibration, and workspace configuration |
| `--manual` | Off | Start paused |
| `--stable-frames` | 2 | Consecutive consistent detections before automatic picking |
| `--settle-seconds` | 0.3 | Fresh-frame interval after each cycle |
| `--stack-count` | 0 | Number of blocks already placed |
| `--save-debug` | Off | Save color, depth, mask, and detection images for each pick |
| `--camera-only` | Off | Test color/depth streaming and exit without connecting to the arm |
| `--camera-debug` | Off | Print native RealSense driver diagnostics |
| `--width`, `--height` | 1280, 720 | Camera resolution; pixel thresholds scale automatically |
| `--fps` | Config value | Camera frame rate |
| `--preview-scale` | 1.0 | Window size relative to camera resolution |
| `--fast-move-speed` | 180 | Travel speed in mm/s when clearance permits |
| `--fast-speed-min-clearance-mm` | 35 | Clearance needed for the faster travel speed |

Example for a stack with four blocks already placed:

```bash
sudo .venv/bin/python live_stack_grab.py --stack-count 4 --save-debug
```

With debug saving enabled, output goes to `grasp.debug_dir` (normally
`grab_debug`). Existing numbered files can be overwritten when counts repeat.

## Offline verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v
```

These tests use simulated device responses. They cover continuous scheduling,
pause and single-pick behavior, target filtering, detection stability, stack
limits, controller faults, camera failure, and quit during a blocking motion.
They do not establish that the physical calibration or placement path is correct.
