"""Offline regression tests: never connect to a robot or camera."""
import unittest
from concurrent.futures import Future
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import live_stack_grab as live


class ImmediateExecutor:
    def __init__(self, **kwargs):
        pass

    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, **kwargs):
        pass


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.config = live.load_json(Path(__file__).resolve().parents[1] / 'config2.json')
        self.candidate = dict(xyz_cam=(0.0, 0.0, 0.35), gripper_yaw_deg=0,
                              center_uv=(100, 100), score=0.8, depth_m=0.35)

    def test_default_config_is_config2(self):
        self.assertEqual(live.DEFAULT_CONFIG.name, 'config2.json')
        self.assertTrue(live.DEFAULT_CONFIG.exists())

    def test_camera_only_never_initializes_arm(self):
        stream = Mock()
        stream.read.return_value = (np.zeros((2, 2, 3)), np.ones((2, 2)))
        with patch.object(live.sys, 'argv', ['live_stack_grab.py', '--camera-only']), \
             patch.object(live, 'RealSenseLiveStream', return_value=stream), \
             patch.object(live, 'initialize_live_arm') as initialize, \
             patch.object(live.cv2, 'destroyAllWindows'):
            self.assertEqual(live.main(), 0)
        initialize.assert_not_called()
        stream.stop.assert_called_once()

    def test_native_startup_failure_is_classified_without_stopping_unstarted_pipeline(self):
        with patch.object(live.rs, 'pipeline') as pipeline_factory, \
             patch.object(live.rs, 'config'), patch.object(live.rs, 'align'):
            pipeline = pipeline_factory.return_value
            pipeline.start.side_effect = RuntimeError('failed to set power state')
            stream = live.RealSenseLiveStream(self.config)
            with self.assertRaises(live.CameraStartupError):
                stream.start()
            stream.stop()
            pipeline.stop.assert_not_called()

    def test_camera_with_no_frames_is_reset_once_then_started(self):
        with patch.object(live.rs, 'pipeline') as pipeline_factory, \
             patch.object(live.rs, 'config'), patch.object(live.rs, 'align'), \
             patch.object(live.time, 'sleep'), \
             patch.object(live, 'wait_for_realsense_frames',
                          side_effect=[RuntimeError('RealSense frame timeout')] + [Mock()] * 100), \
             patch.object(live, 'get_realsense_intrinsics', return_value=dict(fx=1, fy=1)), \
             patch.object(live.RealSenseLiveStream, '_decode', return_value=(np.zeros((2, 2, 3)), None, None)):
            pipeline = pipeline_factory.return_value
            device = pipeline.start.return_value.get_device.return_value
            device.first_depth_sensor.return_value.get_depth_scale.return_value = 0.001
            live.RealSenseLiveStream(self.config).start()
        device.hardware_reset.assert_called_once()
        pipeline.stop.assert_called_once()
        self.assertEqual(pipeline.start.call_count, 2)

    def test_failed_camera_does_not_connect_arm(self):
        stream = Mock()
        stream.start.side_effect = live.CameraStartupError('failed to set power state')
        with patch.object(live.sys, 'argv', ['live_stack_grab.py']), \
             patch.object(live, 'RealSenseLiveStream', return_value=stream), \
             patch.object(live, 'initialize_live_arm') as initialize, \
             patch.object(live.cv2, 'destroyAllWindows'):
            with self.assertRaises(live.CameraStartupError):
                live.main()
        initialize.assert_not_called()

    def test_gate_requires_consistent_fresh_frames_and_resets_on_loss(self):
        gate = live.DetectionGate(2)
        self.assertFalse(gate.update(self.candidate))
        self.assertTrue(gate.update(self.candidate))
        self.assertFalse(gate.update(None))
        self.assertFalse(gate.update(self.candidate))
        changed = dict(self.candidate, xyz_cam=(0.03, 0, 0.35))
        self.assertFalse(gate.update(changed))
        self.assertTrue(gate.update(changed))
        rotated = dict(changed, gripper_yaw_deg=40)
        self.assertFalse(gate.update(rotated))

    def test_gate_accepts_equivalent_gripper_angles(self):
        gate = live.DetectionGate(2)
        self.assertFalse(gate.update(dict(self.candidate, gripper_yaw_deg=89)))
        self.assertTrue(gate.update(dict(self.candidate, gripper_yaw_deg=-89)))

    def test_chooses_lower_ranked_reachable_candidate_and_excludes_stack(self):
        candidates = [dict(self.candidate, score=s) for s in (0.95, 0.9, 0.8, 0.7)]
        with patch.object(live, 'compute_pick', side_effect=[
            ([700, 0, 0], 0), ([250, -50, -90], 0),
            ([float('nan'), 0, 0], 0), ([450, 0, 0], 0),
        ]):
            self.assertIs(live.select_pickable_candidate(candidates, self.config, Path('.'), [0]*6), candidates[3])

    def test_classifies_every_candidate_with_a_reason(self):
        candidates = [dict(self.candidate, score=s) for s in (0.95, 0.9, 0.8, 0.7)]
        stack_x, stack_y, _ = self.config['grasp']['place_xyz_mm']
        with patch.object(live, 'compute_pick', side_effect=[
            ([700, 0, 0], 0), ([stack_x, stack_y, 0], 0),
            ([float('nan'), 0, 0], 0), ([450, 0, 0], 0),
        ]):
            labeled = live.classify_candidates(candidates, self.config, Path('.'), [0]*6)
        self.assertEqual([status for _, status in labeled],
                         ['out_of_range', 'stack_zone', 'invalid', live.PICKABLE])
        self.assertIs(live.first_pickable(labeled), candidates[3])

    def test_draws_every_labeled_candidate(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        contour = np.array([[[60, 120]], [[140, 120]], [[140, 150]], [[60, 150]]], dtype=np.int32)
        target = dict(self.candidate, center_uv=(100, 135), contour=contour)
        other = dict(self.candidate, center_uv=(250, 200))
        output = live.draw_labeled_candidates(
            image, [(other, 'stack_zone'), (target, live.PICKABLE)], target)
        self.assertEqual(image.sum(), 0)  # input frame is not modified
        self.assertTrue((output[120, 60:140] == live.LABEL_STYLES['target'][0]).all(axis=-1).any())
        self.assertTrue((output == live.LABEL_STYLES['stack_zone'][0]).all(axis=-1).any())

    def test_existing_fault_is_not_cleared_or_enabled(self):
        arm = Mock(error_code=31)
        arm.get_err_warn_code.return_value = (0, [31, 0])
        with patch.object(live, 'XArmAPI', return_value=arm):
            with self.assertRaises(RuntimeError):
                live.initialize_live_arm(self.config)
        arm.clean_error.assert_not_called()
        arm.motion_enable.assert_not_called()
        arm.disconnect.assert_called_once()

    def test_stop_prevents_following_commands(self):
        event = Event()
        arm = Mock()
        guarded = live.InterruptibleArm(arm, event)
        arm.set_position.side_effect = lambda **kwargs: event.set()
        with self.assertRaises(live.MotionStopped):
            guarded.set_position(z=150)
        with self.assertRaises(live.MotionStopped):
            guarded.set_gripper_position(600)
        arm.set_gripper_position.assert_not_called()

    def test_full_stack_does_not_send_motion(self):
        arm = Mock()
        with patch.object(live, 'get_arm_position_mm_deg', return_value=[0]*6), \
             patch.object(live, 'compute_pick', return_value=([450, 0, 0], 0)), \
             patch.object(live, 'execute_stack_pick_and_place') as motion:
            with self.assertRaisesRegex(RuntimeError, 'Stack target'):
                live.stack_current_detection(arm, self.config, Path('.'), 100,
                    1, None, None, None, self.candidate, 180, 35)
            motion.assert_not_called()

    def run_loop(self, keys, manual=False, failure=None, camera_failure=None):
        args = SimpleNamespace(manual=manual, stack_count=0, stable_frames=2,
            settle_seconds=0, save_debug=False, fast_move_speed=180,
            fast_speed_min_clearance_mm=35)
        stream = Mock()
        stream.poll.return_value = (np.zeros((120, 160, 3), dtype=np.uint8),
                                    np.ones((120, 160), dtype=np.uint16))
        stream.poll.side_effect = camera_failure
        pipeline = Mock(last_mask=None)
        pipeline.process.return_value = [self.candidate]
        arm = Mock(error_code=0)
        arm.set_state.return_value = 0
        with patch.object(live, 'ThreadPoolExecutor', ImmediateExecutor), \
             patch.object(live, 'move_to_scan_pose'), \
             patch.object(live, 'get_arm_position_mm_deg', return_value=[400, -100, 150, 180, 0, 0]), \
             patch.object(live, 'classify_candidates', return_value=[(self.candidate, live.PICKABLE)]), \
             patch.object(live, 'stack_current_detection', return_value=True, side_effect=failure) as grab, \
             patch.object(live.time, 'monotonic', return_value=100), \
             patch.object(live.cv2, 'imshow'), \
             patch.object(live.cv2, 'getWindowProperty', return_value=1), \
             patch.object(live.cv2, 'waitKey', side_effect=keys):
            result = live.run_preview(arm, stream, pipeline, self.config, Path('.'), args)
        return result, grab, arm

    def test_camera_failure_stops_motion(self):
        with patch.object(live, 'InterruptibleArm') as guarded:
            with self.assertRaisesRegex(RuntimeError, 'camera disconnected'):
                self.run_loop([], camera_failure=RuntimeError('camera disconnected'))
            real_arm, stop_event = guarded.call_args.args
            real_arm.set_state.assert_called_once_with(4)
            self.assertTrue(stop_event.is_set())

    def test_quit_during_blocking_motion_keeps_ui_responsive(self):
        started, released = Event(), Event()
        arm = Mock(error_code=0)

        def blocking_move(**kwargs):
            started.set()
            if not released.wait(2):
                raise RuntimeError('test stop did not release blocking move')
            return 0

        def stop(state):
            released.set()
            return 0

        def scan(guarded, config):
            guarded.set_position(z=150)
            guarded.set_gripper_position(600)

        def quit_key(delay):
            self.assertTrue(started.wait(2))
            return ord('q')

        arm.set_position.side_effect = blocking_move
        arm.set_state.side_effect = stop
        args = SimpleNamespace(manual=False, stack_count=0, stable_frames=2, settle_seconds=0.3)
        stream = Mock()
        stream.poll.return_value = (np.zeros((120, 160, 3), dtype=np.uint8), None)
        with patch.object(live, 'move_to_scan_pose', side_effect=scan), \
             patch.object(live.cv2, 'imshow'), \
             patch.object(live.cv2, 'waitKey', side_effect=quit_key):
            self.assertEqual(live.run_preview(arm, stream, Mock(), self.config, Path('.'), args), 0)
        arm.set_state.assert_called_once_with(4)
        arm.set_gripper_position.assert_not_called()

    def test_auto_runs_multiple_cycles_without_g_and_advances_slots(self):
        result, grab, arm = self.run_loop([-1, -1, -1, -1, ord('q')])
        self.assertEqual(result, 0)
        self.assertEqual([c.args[3] for c in grab.call_args_list], [0, 1])
        arm.set_state.assert_called_once_with(4)

    def test_pause_prevents_new_cycles_and_manual_g_runs_once(self):
        _, grab, _ = self.run_loop([ord(' '), -1, ord('g'), -1, -1, ord('q')])
        self.assertEqual(grab.call_count, 1)
        _, grab, _ = self.run_loop([-1, -1, ord('q')], manual=True)
        grab.assert_not_called()

    def test_fault_latches_and_space_cannot_restart_motion(self):
        result, grab, arm = self.run_loop(
            [-1, -1, -1, ord(' '), ord('g'), -1, ord('q')],
            failure=RuntimeError('controller fault 31'))
        self.assertEqual(result, 1)
        self.assertEqual(grab.call_count, 1)
        arm.set_state.assert_called_once_with(4)
        arm.clean_error.assert_not_called()


if __name__ == '__main__':
    unittest.main()
