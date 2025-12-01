"""
Data collection script using IK-based joint control.
Simplified version using modular components.
"""
import os
import sys

# Remove ROS paths to avoid conflicts
sys.path = [p for p in sys.path if not any(x in p for x in ['/opt/ros', 'franka_ros2_ws', '/opt/openrobots'])]

import time
import json
import numpy as np
import tyro
from dataclasses import dataclass
from transforms3d.euler import euler2mat, mat2euler

from franky import Robot, JointWaypointMotion, JointWaypoint, RelativeDynamicsFactor

from franky_control.robot import FC, GripperController, IKController
from franky_control.driver import SpaceMouse, RealsenseAPI, SPACEMOUSE_AVAILABLE, REALSENSE_AVAILABLE
from franky_control.data_collection.data_collector_v2 import DataCollector
from franky_control.utils import get_next_episode_idx, ensure_dir


@dataclass 
class Args:
    """Data collection arguments."""
    task_name: str  # Task name for dataset
    instruction: str  # Task instruction
    robot_ip: str = FC.ROBOT_IP
    dataset_dir: str = FC.DEFAULT_DATASET_DIR
    min_steps: int = FC.MIN_EPISODE_STEPS
    max_steps: int = FC.MAX_EPISODE_STEPS
    episode_idx: int = -1  # -1 for auto-increment
    pos_scale: float = FC.DEFAULT_POS_SCALE
    rot_scale: float = FC.DEFAULT_ROT_SCALE
    control_freq: float = FC.VLA_CONTROL_FREQUENCY
    # IK options
    use_gpu_ik: bool = False
    use_joint_filter: bool = False
    filter_alpha: float = 0.4
    # Hardware options
    use_space_mouse: bool = True
    use_cameras: bool = True
    use_gripper: bool = True


class DataCollection:
    """Simplified data collection using IK-based joint control."""
    
    def __init__(self, args: Args, robot: Robot, cameras: RealsenseAPI=None, gripper: GripperController=None, space_mouse: SpaceMouse=None):
        self.args = args
        self.robot = robot
        self.cameras = cameras
        self.gripper = gripper
        self.space_mouse = space_mouse
        
        # Control state
        self.command_xyz = None
        self.command_rotation = None
        self.gripper_closed = False
        self.step_count = 0
        self.control_dt = 1.0 / args.control_freq
        
        # Initialize IK controller
        self.ik = IKController(
            device="cuda" if args.use_gpu_ik else "cpu",
            use_joint_filter=args.use_joint_filter,
            filter_alpha=args.filter_alpha,
        )
        
        # Initialize data collector
        self.collector = DataCollector(robot, cameras, gripper, is_image_encoded=True)
    
    def init_pose(self):
        """Initialize command pose from current robot state."""
        time.sleep(0.5)
        ee = self.robot.current_pose.end_effector_pose
        self.command_xyz = np.array(ee.translation)
        self.command_rotation = np.array(ee.matrix[:3, :3])
        
        joints = np.array(self.robot.current_joint_state.position)
        self.ik.reset(joints)
        
        print(f"[INFO] Initial position: {self.command_xyz}")
        print(f"[INFO] Initial joints: {joints}")
    
    def read_spacemouse(self):
        """Read and process SpaceMouse input."""
        if self.space_mouse is None:
            return np.zeros(3), np.zeros(3), 1.0, False
        
        control = self.space_mouse.control
        gripper = self.space_mouse.gripper_status
        quit_sig = self.space_mouse.quit_signal
        
        # Process with deadzone
        xyz = self._apply_deadzone(control[:3])
        euler = self._apply_deadzone(control[3:6][[1, 0, 2]] * [-1, -1, 1])
        
        return xyz, euler, gripper, quit_sig
    
    def _apply_deadzone(self, ctrl, deadzone=FC.SPACEMOUSE_DEADZONE):
        """Apply deadzone scaling to control input."""
        ctrl = np.clip(ctrl, -1.0, 1.0)
        scaled = np.zeros_like(ctrl)
        pos = ctrl >= deadzone
        neg = ctrl <= -deadzone
        if deadzone < 1.0:
            scaled[pos] = (ctrl[pos] - deadzone) / (1.0 - deadzone)
            scaled[neg] = (ctrl[neg] + deadzone) / (1.0 - deadzone)
        return np.clip(scaled, -1.0, 1.0)
    
    def control_gripper(self, gripper_action: float):
        """Control gripper based on action (< 0.5 = close)."""
        if self.gripper is None:
            return
        should_close = gripper_action < 0.5
        if should_close != self.gripper_closed:
            try:
                if should_close:
                    self.gripper.grasp(async_call=True, force=FC.GRIPPER_DEFAULT_FORCE,
                                      epsilon_inner=0.06, epsilon_outer=0.06)
                else:
                    self.gripper.open(async_call=True)
                self.gripper_closed = should_close
            except Exception as e:
                print(f"[WARNING] Gripper error: {e}")
    
    def run(self):
        """Main data collection loop."""
        input("Press Enter to start data collection...")
        print("[INFO] Starting collection (Ctrl+C or SpaceMouse button to stop)")
        
        try:
            while True:
                t0 = time.time()
                
                try:
                    # Read control input
                    ctrl_xyz, ctrl_euler, ctrl_gripper, quit_sig = self.read_spacemouse()
                    if quit_sig:
                        print("[INFO] Stopped by user")
                        break
                    
                    # Compute deltas
                    delta_xyz = ctrl_xyz * self.args.pos_scale
                    delta_euler = ctrl_euler * self.args.rot_scale
                    delta_rot = euler2mat(delta_euler[0], delta_euler[1], delta_euler[2], 'sxyz')
                    
                    # Update command pose
                    self.command_xyz += delta_xyz
                    self.command_rotation = self.command_rotation @ delta_rot
                    
                    # Compute IK
                    target_joints = self.ik.compute_ik(self.command_xyz, self.command_rotation)
                    if target_joints is None:
                        print(f"[WARNING] IK failed at step {self.step_count}")
                        time.sleep(self.control_dt)
                        continue
                    
                    # Record data
                    self.collector.record_observation(self.args.instruction)
                    self.collector.record_action(
                        delta_xyz=delta_xyz,
                        delta_euler=delta_euler,
                        abs_position=self.command_xyz,
                        abs_rotation=self.command_rotation,
                        gripper_control=ctrl_gripper,
                        abs_joints=target_joints,
                    )
                    
                    # Send joint command
                    waypoint = JointWaypoint(target_joints.tolist())
                    motion = JointWaypointMotion(
                        [waypoint],
                        relative_dynamics_factor=RelativeDynamicsFactor(
                            velocity=FC.DEFAULT_VELOCITY_FACTOR, 
                            acceleration=FC.DEFAULT_ACCELERATION_FACTOR, 
                            jerk=FC.DEFAULT_JERK_FACTOR,
                        )
                    )
                    self.robot.move(motion, asynchronous=True)
                    
                    # Control gripper
                    self.control_gripper(ctrl_gripper)
                    
                    self.step_count += 1
                    
                    # Maintain control frequency
                    elapsed = time.time() - t0
                    if elapsed < self.control_dt:
                        time.sleep(self.control_dt - elapsed)
                
                except KeyboardInterrupt:
                    print("[INFO] Stopped by keyboard")
                    break
                except Exception as e:
                    print(f"[ERROR] An error occurred during data collection: {e}")
                    
                    # Try to recover from errors (e.g., Reflex mode after collision)
                    try:
                        print("[INFO] Attempting to recover from error...")
                        self.robot.recover_from_errors()
                        time.sleep(0.05)  # Wait for recovery
                        print("[INFO] Error recovery successful")
                    except Exception as recover_error:
                        print(f"[WARNING] Error recovery failed: {recover_error}")
                    
                    time.sleep(self.control_dt)
                    self.init_pose()  # Re-initialize pose after recovery
                    continue
                    
        except KeyboardInterrupt:
            print("[INFO] Stopped by keyboard")
        finally:
            self.ik.print_stats()
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources."""
        if self.space_mouse:
            self.space_mouse.close()
        if self.cameras:
            self.cameras.close()
    
    def save(self, base_dir: str) -> bool:
        """Save collected data."""
        if self.step_count > self.args.max_steps:
            print(f"[ERROR] Too many steps ({self.step_count} > {self.args.max_steps})")
            return False
        
        metadata = {
            "task_name": self.args.task_name,
            "instruction": self.args.instruction,
            "steps": self.step_count,
            "control_type": "ik_joint_control",
            "ik_mode": self.ik.alignment_mode,
        }
        
        save_path = self.collector.save(
            base_dir,
            episode_idx=self.args.episode_idx,
            metadata=metadata,
            save_video=True,
            use_hdf5=True,
        )
        print(f"[INFO] Saved to {save_path}")
        return True


def main(args: Args):
    """Main function."""
    # Initialize robot
    print(f"[INFO] Connecting to robot at {args.robot_ip}")
    robot = Robot(args.robot_ip)
    robot.recover_from_errors()
    
    # Initialize cameras
    cameras = None
    if args.use_cameras and REALSENSE_AVAILABLE:
        try:
            cameras = RealsenseAPI(FC.CAMERA_HEIGHT, FC.CAMERA_WIDTH, FC.CAMERA_FPS)
            print(f"[INFO] Cameras: {cameras.get_num_cameras()}")
        except Exception as e:
            print(f"[WARNING] Camera init failed: {e}")
    
    # Initialize SpaceMouse
    space_mouse = None
    if args.use_space_mouse and SPACEMOUSE_AVAILABLE:
        try:
            space_mouse = SpaceMouse(FC.SPACEMOUSE_VENDOR_ID, FC.SPACEMOUSE_PRODUCT_ID)
        except Exception as e:
            print(f"[WARNING] SpaceMouse init failed: {e}")
    
    # Move to home
    print("[INFO] Moving to home...")
    home = JointWaypointMotion(
        [JointWaypoint(FC.RESET_JOINTS)],
        relative_dynamics_factor=RelativeDynamicsFactor(
            velocity=FC.DEFAULT_VELOCITY_FACTOR,
            acceleration=FC.DEFAULT_ACCELERATION_FACTOR,
            jerk=FC.DEFAULT_JERK_FACTOR
        )
    )
    robot.move(home)

    # Initialize gripper
    gripper = None
    if args.use_gripper:
        try:
            gripper = GripperController(args.robot_ip)
            if gripper.home():
                gripper.open()
                print("[INFO] Gripper ready")
        except Exception as e:
            print(f"[WARNING] Gripper init failed: {e}")
            gripper = None
    
    # Run collection
    collector = DataCollection(args, robot, cameras, gripper, space_mouse)
    collector.init_pose()
    collector.run()
    
    # Check and save
    if collector.step_count < args.min_steps:
        print(f"[ERROR] Not enough steps ({collector.step_count} < {args.min_steps})")
        return -1
    
    task_dir = os.path.join(args.dataset_dir, args.task_name)
    ensure_dir(task_dir)
    
    if collector.save(task_dir):
        print(f"\033[32mSuccess! Saved {collector.step_count} steps.\033[0m")
        return 0
    return -1


if __name__ == "__main__":
    args = tyro.cli(Args)
    sys.exit(main(args))
