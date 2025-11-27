"""
Real robot data collection script using direct Cartesian pose control.
This implementation uses franky.Robot with CartesianMotion (no IK solver needed).
"""
import os
import sys

# Remove ROS2 and OpenRobots paths to avoid interference with conda packages
sys.path = [p for p in sys.path if not any(x in p for x in ['/opt/ros', 'franka_ros2_ws', '/opt/openrobots'])]

import time
import json
import copy
import numpy as np
import tyro
from dataclasses import dataclass
from transforms3d.euler import euler2quat, euler2mat, mat2euler
from transforms3d.quaternions import mat2quat, quat2mat

# Import franky robot control
from franky import (
    Robot, Affine, ReferenceType, CartesianMotion,
    JointWaypointMotion, JointWaypoint, 
    RelativeDynamicsFactor
)

# Import local drivers
from franky_control.driver import SpaceMouse, RealsenseAPI, SPACEMOUSE_AVAILABLE, REALSENSE_AVAILABLE
from franky_control.data_collection import DataCollector


@dataclass 
class Args:
    """Data collection script arguments."""
    task_name: str  # Task name for the dataset
    instruction: str  # Instruction for data collection
    robot_ip: str = "172.16.0.2"  # Robot IP address
    dataset_dir: str = "datasets"  # Directory to save dataset
    min_action_steps: int = 200  # Minimum action_steps for data collection
    max_action_steps: int = 500  # Maximum action_steps for data collection  
    episode_idx: int = -1  # Episode index to save data (-1 for auto-increment)
    pos_scale: float = 0.015  # The scale of xyz action
    rot_scale: float = 0.025  # The scale of rotation action
    control_frequency: float = 10.0  # Control frequency in Hz
    use_space_mouse: bool = True  # Use SpaceMouse for control
    use_cameras: bool = False  # Use RealSense cameras
    use_gripper: bool = True  # Use Franka gripper


class FrankyDataCollectionCartesian:
    """Data collection class using direct Cartesian pose control.
    
    Uses franky's CartesianMotion or ImpedanceMotion with Affine transforms.
    Supports both position control and compliant impedance control.
    No IK solver needed - franky handles the inverse kinematics internally.
    """
    
    def __init__(self, args: Args, robot: Robot, cameras=None):
        self.robot: Robot = robot
        self.cameras = cameras

        self.args = args
        self.data_collector = DataCollector(robot, cameras)
        self.use_space_mouse = args.use_space_mouse
        if self.use_space_mouse:
            if not SPACEMOUSE_AVAILABLE:
                print("[WARNING] SpaceMouse not available, using dummy control")
                self.space_mouse = None
            else:
                self.space_mouse = SpaceMouse(vendor_id=0x256f, product_id=0xc635)
        else:
            self.space_mouse = None
        
        self.episode_idx = 0
        self.action_steps = 0
        self.instruction = args.instruction
        self.init_xyz = None
        self.init_rotation = None
        self.command_xyz = None
        self.command_rotation = None
        self.control_frequency = args.control_frequency
        self.control_time_step = 1.0 / self.control_frequency
        self.last_gripper_state = 1.0  # Start with open
        self.init_time = time.time()
        
        # Initialize gripper controller if available
        self.gripper_controller = None
        if args.use_gripper:
            try:
                from franky_control.robot import GripperController
                self.gripper_controller = GripperController(args.robot_ip)
                print("[INFO] Initializing gripper...")
                if self.gripper_controller.home():
                    print("[INFO] Gripper ready")
                    self.gripper_controller.open()
                else:
                    print("[WARNING] Gripper homing failed, continuing without gripper")
                    self.gripper_controller = None
            except Exception as e:
                print(f"[WARNING] Failed to initialize gripper: {e}")
                self.gripper_controller = None

        self.pos_scale = args.pos_scale
        self.rot_scale = args.rot_scale
        
        # Track previous pose for velocity calculation
        self.prev_xyz = None
        self.prev_rotation = None
        self.prev_time = None

    def ee_pose_init(self):
        """Initialize the end-effector pose."""
        time.sleep(0.5)
        current_pose = self.robot.current_pose
        
        # Extract position and rotation from franky's RobotPose
        ee_affine = current_pose.end_effector_pose
        self.init_xyz = np.array(ee_affine.translation)
        self.init_rotation = np.array(ee_affine.matrix[:3, :3])  # Extract 3x3 rotation
        
        self.command_xyz = self.init_xyz.copy()
        self.command_rotation = self.init_rotation.copy()
        
        # Initialize previous pose for velocity calculation
        self.prev_xyz = self.init_xyz.copy()
        self.prev_rotation = self.init_rotation.copy()
        self.prev_time = time.time()
        
        # Get initial joint positions for logging
        joint_state = self.robot.current_joint_state
        self.current_joints = np.array(joint_state.position)
        
        print(f"[INFO] Initial joints: {self.current_joints}")
        print(f"[INFO] Initial EE position: {self.init_xyz}")

    def _apply_control_data_clip_and_scale(self, control_tensor, offset=0.0):
        """Clip and scale control data."""
        control_tensor = np.clip(control_tensor, -1.0, 1.0)
        scaled_tensor = np.zeros_like(control_tensor)
        positive_mask = (control_tensor >= offset)
        negative_mask = (control_tensor <= -offset)
        if offset < 1.0 and offset >= 0.0:
            scaled_tensor[positive_mask] = (control_tensor[positive_mask] - offset) / (1.0 - offset)
            scaled_tensor[negative_mask] = (control_tensor[negative_mask] + offset) / (1.0 - offset)
        else:
            raise ValueError(f"offset should in the range of 0-1, while the offset is set to be {offset}.")
        return np.clip(scaled_tensor, -1.0, 1.0)

    def collect_data(self):
        """Main data collection loop using Cartesian pose control."""
        input("Press enter to start collection with Cartesian control")
        print("[INFO] Starting data collection with direct Cartesian control...")
        
        try:
            while True:
                loop_start_time = time.time()
                
                try:
                    # Read SpaceMouse controls
                    if self.use_space_mouse and self.space_mouse is not None:
                        control = self.space_mouse.control
                        control_gripper = self.space_mouse.gripper_status
                        control_quit = self.space_mouse.quit_signal
                    else:
                        control = np.zeros((6,))
                        control_gripper = 1
                        control_quit = False

                    if control_quit:
                        print("[INFO] Data collection stopped by user.")
                        break

                    # Process control signals
                    control_xyz = control[:3]
                    control_euler = control[3:6][[1,0,2]] * np.array([-1,-1,1])
                    control_xyz = self._apply_control_data_clip_and_scale(control_xyz, 0.35)
                    control_euler = self._apply_control_data_clip_and_scale(control_euler, 0.35)

                    # Compute delta pose
                    delta_xyz = control_xyz * self.pos_scale
                    delta_euler = control_euler * self.rot_scale
                    delta_rotation = euler2mat(delta_euler[0], delta_euler[1], delta_euler[2], 'sxyz')
                    
                    # Update command pose (for logging/saving only)
                    self.command_xyz += delta_xyz
                    self.command_rotation = np.matmul(self.command_rotation, delta_rotation)

                    timestamp = time.time() - self.init_time

                    # Create target pose as Affine transform
                    # Build 4x4 transformation matrix
                    target_matrix = np.eye(4)
                    target_matrix[:3, :3] = self.command_rotation
                    target_matrix[:3, 3] = self.command_xyz
                    
                    # Create Affine object from matrix
                    target_affine = Affine(target_matrix)
                    
                    # Get current joint positions for logging
                    joint_state = self.robot.current_joint_state
                    current_joints = np.array(joint_state.position)

                    # Save action data
                    save_action = {
                        "delta": {
                            "position": delta_xyz,
                            "orientation": euler2quat(delta_euler[0], delta_euler[1], delta_euler[2], 'sxyz'),
                            "euler_angle": delta_euler,
                        },
                        "abs": {
                            "position": copy.deepcopy(self.command_xyz),
                            "euler_angle": np.array([mat2euler(self.command_rotation, 'sxyz')])[0],
                            "joints": current_joints.copy(),
                        },
                        "gripper_control": control_gripper
                    }

                    # Collect data
                    self.data_collector.update_data_dict(
                        instruction=self.instruction,
                        action=save_action,
                        timestamp=timestamp,
                    )

                    # Send Cartesian position command using RELATIVE motion
                    # Key insight: Send motion commands EVERY iteration for smooth real-time control
                    # Use very small dynamics factors (like examples: 0.05) for 20Hz updates
                    if self.action_steps % 50 == 0:
                        print(f"[DEBUG] Target pos: {self.command_xyz}, delta: {delta_xyz}")
                    
                    # Create RELATIVE motion (delta from current position)
                    # Build 4x4 transformation matrix for delta
                    delta_matrix = np.eye(4)
                    delta_matrix[:3, :3] = delta_rotation
                    delta_matrix[:3, 3] = delta_xyz
                    
                    # Create Affine from matrix
                    delta_affine = Affine(delta_matrix)
                    
                    cartesian_motion = CartesianMotion(
                        delta_affine,
                        ReferenceType.Relative,  # Relative to current position
                        relative_dynamics_factor=RelativeDynamicsFactor(
                            velocity=0.4,   # Very small for real-time control (like examples)
                            acceleration=0.1,
                            jerk=0.1,
                        ),
                        return_when_finished=False  # Don't wait for motion to finish
                    )
                    self.robot.move(cartesian_motion, asynchronous=True)

                    # Control gripper (if available)
                    if hasattr(self, 'gripper_controller') and self.gripper_controller is not None:
                        if abs(control_gripper - self.last_gripper_state) > 0.02:
                            gripper_width = 0.08 * control_gripper  # Scale to gripper range
                            try:
                                if control_gripper < 0.5:
                                    # Close/grasp
                                    self.gripper_controller.grasp(
                                        force=50.0, 
                                        epsilon_inner=0.08, 
                                        epsilon_outer=0.08, 
                                        async_call=True
                                    )
                                else:
                                    # Open to specified width
                                    self.gripper_controller.move(
                                        gripper_width, 
                                        speed=0.12, 
                                        async_call=True
                                    )
                                self.last_gripper_state = control_gripper
                            except Exception as e:
                                print(f"[WARNING] Gripper control failed: {e}")

                    self.action_steps += 1
                    
                    # Log progress
                    if not self.use_space_mouse and not self.args.use_cameras:
                        if self.action_steps % 10 == 0:
                            print(f"[INFO] Step {self.action_steps}/{self.args.max_action_steps}")
                    
                    # Sleep to maintain control frequency
                    elapsed = time.time() - loop_start_time
                    if elapsed < self.control_time_step:
                        time.sleep(self.control_time_step - elapsed)
                    
                except KeyboardInterrupt:
                    print("[INFO] Data collection stopped by keyboard interrupt.")
                    break
                except Exception as e:
                    print(f"[ERROR] An error occurred during data collection: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Try to recover from errors (e.g., Reflex mode after collision)
                    try:
                        print("[INFO] Attempting to recover from error...")
                        self.robot.recover_from_errors()
                        time.sleep(0.5)  # Wait for recovery
                        print("[INFO] Error recovery successful")
                    except Exception as recover_error:
                        print(f"[WARNING] Error recovery failed: {recover_error}")
                    
                    time.sleep(self.control_time_step)
                    self.ee_pose_init()
                    continue
        finally:
            # Clean up resources
            if self.space_mouse is not None:
                self.space_mouse.close()
            if self.cameras is not None:
                self.cameras.close()

    def get_next_episode_idx(self, task_dir):
        """Find the next episode index by identifying the highest existing episode number."""
        if not os.path.exists(task_dir):
            return 0

        all_items = os.listdir(task_dir)
        episode_dirs = [item for item in all_items 
                       if os.path.isdir(os.path.join(task_dir, item)) and item.startswith("episode_")]

        if not episode_dirs:
            return 0

        episode_numbers = []
        for dir_name in episode_dirs:
            try:
                episode_number = int(dir_name.split("_")[1])
                episode_numbers.append(episode_number)
            except (IndexError, ValueError):
                continue

        if not episode_numbers:
            return 0

        return max(episode_numbers) + 1

    def save_data(self, task_dir):
        """Save collected data to disk."""
        if self.action_steps > self.args.max_action_steps:
            print("action_steps too large, data not saved")
            return False
            
        if self.args.episode_idx < 0:
            self.episode_idx = self.get_next_episode_idx(task_dir)
        else:
            self.episode_idx = self.args.episode_idx
            
        episode_dir = os.path.join(task_dir, f"episode_{self.episode_idx}")

        os.makedirs(episode_dir, exist_ok=True)
        metadata_path = os.path.join(episode_dir, "metadata.json")
        self.data_collector.save_data(episode_dir, self.episode_idx)

        metadata = {
            "task_name": self.args.task_name,
            "episode_idx": self.episode_idx,
            "action_steps": self.action_steps,
            "instruction": self.instruction,
            "control_type": "cartesian_pose_control",
            "robot_interface": "franky",
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[INFO] Data saved to {episode_dir}")
        print(f"[INFO] Metadata saved to {metadata_path}")
        return True


def main(args: Args):
    """Main function with tyro argument parsing.
    
    Args:
        args: Arguments from tyro CLI
    """
    # Initialize robot
    print(f"[INFO] Connecting to robot at {args.robot_ip}")
    robot = Robot(args.robot_ip)
    robot.recover_from_errors()
    
    # Initialize cameras
    cameras = None
    if args.use_cameras:
        if REALSENSE_AVAILABLE:
            try:
                cameras = RealsenseAPI()
                print(f"[INFO] Initialized {cameras.get_num_cameras()} camera(s)")
            except Exception as e:
                print(f"[WARNING] Failed to initialize cameras: {e}")
        else:
            print("[WARNING] RealSense not available, data collection without cameras")

    # Move to home position
    print("[INFO] Moving to home position...")
    home_motion = JointWaypointMotion([
        JointWaypoint([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], 
                      relative_dynamics_factor=RelativeDynamicsFactor(velocity=0.18, acceleration=0.13, jerk=0.11))
    ])
    robot.move(home_motion)
    
    # Set global relative dynamics factor for real-time control
    # This is multiplied with the motion-specific factor for smooth real-time updates
    robot.relative_dynamics_factor = RelativeDynamicsFactor(0.5)

    # For Cartesian POSITION control, use CARTESIAN impedance
    # Lower stiffness = more compliant/smoother, less vibration
    # Higher stiffness = more precise but potentially more vibration
    # Translational stiffness [x, y, z]: 10-3000 N/m
    # Rotational stiffness [rx, ry, rz]: 1-300 Nm/rad
    robot.set_cartesian_impedance([600.0, 600.0, 600.0, 60.0, 60.0, 60.0])

    # Set collision behavior to reasonable values
    robot.set_collision_behavior(
        # Lower torque thresholds (contact detection)
        [25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.0],
        # Upper torque thresholds (collision detection)
        [35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0],
        # Lower force thresholds
        [30.0, 30.0, 30.0, 25.0, 25.0, 25.0],
        # Upper force thresholds
        [40.0, 40.0, 40.0, 35.0, 35.0, 35.0]
    )
    
    # Create data collection instance
    collection = FrankyDataCollectionCartesian(
        args, 
        robot, 
        cameras, 
    )
    
    # Initialize pose and start collection
    collection.ee_pose_init()
    collection.collect_data()

    # Check if enough data collected
    if collection.action_steps < args.min_action_steps:
        print(f"[Error] Save failure (#step < {args.min_action_steps}), please try again.")
        return -1

    # Save data
    task_dir = os.path.join(args.dataset_dir, args.task_name)
    os.makedirs(task_dir, exist_ok=True)
    
    result = collection.save_data(task_dir)
    if result:
        print(f"\033[32m\nSave success, saved {collection.action_steps} action_steps of data.\033[0m\n")
        return 0
    else:
        print(f"\033[31m\nSave failed.\033[0m\n")
        return -1


if __name__ == "__main__":
    args = tyro.cli(Args)
    sys.exit(main(args))
