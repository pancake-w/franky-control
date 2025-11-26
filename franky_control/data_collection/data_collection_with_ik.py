"""
Real robot data collection script using IK to convert EE pose commands to joint commands.
This implementation uses franky.Robot instead of FrankaArm (FrankaPy).
"""
import os
import sys

# Remove ROS2 and OpenRobots paths to avoid interference with conda packages
# This prevents ROS2's Pinocchio from conflicting with our pytorch_kinematics setup
sys.path = [p for p in sys.path if not any(x in p for x in ['/opt/ros', 'franka_ros2_ws', '/opt/openrobots'])]

import time
import json
import copy
import numpy as np
import tyro
from dataclasses import dataclass
from transforms3d.euler import euler2quat, euler2mat, mat2euler
from transforms3d.quaternions import mat2quat, quat2mat
import torch

# Import franky robot control
from franky import Robot, JointWaypointMotion, JointWaypoint, ReferenceType

# Import local drivers
from franky_control.driver import SpaceMouse, RealsenseAPI, SPACEMOUSE_AVAILABLE, REALSENSE_AVAILABLE
from franky_control.data_collection import DataCollector
from franky_control.kinematics import PANDA_URDF_PATH, ensure_assets_downloaded
from franky_control.kinematics.panda_ik_solver import create_sim_aligned_ik_solver, SimPose


@dataclass 
class Args:
    """Data collection script arguments."""
    task_name: str  # Task name for the dataset
    instruction: str  # Instruction for data collection
    robot_ip: str = "172.16.0.2"  # Robot IP address
    dataset_dir: str = "datasets"  # Directory to save dataset
    min_action_steps: int = 200  # Minimum action_steps for data collection
    max_action_steps: int = 1000  # Maximum action_steps for data collection  
    episode_idx: int = -1  # Episode index to save data (-1 for auto-increment)
    pos_scale: float = 0.015  # The scale of xyz action
    rot_scale: float = 0.025  # The scale of rotation action
    use_gpu_ik: bool = False  # Use GPU for IK solver
    verify_ik: bool = False  # Verify IK solution with FK (for debugging)
    control_frequency: float = 5.0  # Control frequency in Hz
    use_space_mouse: bool = False  # Use SpaceMouse for control
    use_cameras: bool = False  # Use RealSense cameras
    use_gripper: bool = True  # Use Franka gripper


class FrankyDataCollectionWithIK:
    """Data collection class using IK with franky.Robot.
    
    Uses SimAlignedPandaIKSolver - fully aligned with ManiSkill simulation kinematics.
    """
    
    def __init__(self, args: Args, robot: Robot, cameras=None, ):
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

        # Initialize IK solver
        self._init_ik_solver()

    def _init_ik_solver(self):
        """Initialize the simulation-aligned Panda IK solver."""
        urdf_path = PANDA_URDF_PATH
        
        # Use CPU for IK solver (more stable for real-time control)
        device = "cuda" if self.args.use_gpu_ik else "cpu"
        
        print(f"[INFO] Initializing Simulation-Aligned Panda IK solver on device: {device}")
        
        # Use pytorch_kinematics instead (CPU or GPU)
        self.ik_solver = create_sim_aligned_ik_solver(
            urdf_path=urdf_path,
            device=device,
        )
        print(f"[INFO] IK solver initialized successfully")
        print(f"[INFO] Solver mode: {self.ik_solver.alignment_mode}")
        print(f"[INFO] Max iterations: {self.ik_solver.max_iterations}")

    def _compute_ik_for_pose(self, position, rotation_matrix, initial_joints=None):
        """
        Compute IK for given pose using the simulation-aligned Panda IK solver.
        
        Args:
            position: np.array of shape (3,) - xyz position
            rotation_matrix: np.array of shape (3,3) - rotation matrix
            initial_joints: np.array of shape (7,) - initial joint positions (warm start)
            
        Returns:
            np.array of shape (7,) - joint positions, or None if IK fails
        """
        # Create target pose using SimPose (aligned with ManiSkill)
        # Convert rotation matrix to quaternion (wxyz format) using transforms3d
        quat_wxyz = mat2quat(rotation_matrix)  # transforms3d returns [w,x,y,z]
        
        target_pose = SimPose.from_pq(
            position=position,
            quaternion=quat_wxyz,
            device=self.ik_solver.device
        )
        
        # Solve IK with warm start
        ik_solution = self.ik_solver.compute_ik(
            target_pose,
            initial_qpos=initial_joints  # Warm start for better convergence
        )
        
        if ik_solution is not None:
            # Return numpy array (remove batch dimension)
            return ik_solution[0].cpu().numpy()
        else:
            return None

    def ee_pose_init(self):
        """Initialize the end-effector pose."""
        time.sleep(0.5)
        current_pose = self.robot.current_pose
        
        # Extract position and rotation from franky's RobotPose
        # current_pose is RobotPose, which has end_effector_pose (Affine)
        ee_affine = current_pose.end_effector_pose
        self.init_xyz = np.array(ee_affine.translation)
        self.init_rotation = np.array(ee_affine.matrix[:3, :3])  # Extract 3x3 rotation from 4x4 matrix
        
        self.command_xyz = self.init_xyz.copy()
        self.command_rotation = self.init_rotation.copy()
        
        # Get initial joint positions for IK warm start
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
        """Main data collection loop using IK-based joint control."""
        input("Press enter to start collection with IK-based control")
        print("[INFO] Starting data collection with IK solver...")
        
        # Statistics
        ik_success_count = 0
        ik_failure_count = 0
        
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

                    # Update command pose
                    self.command_xyz += delta_xyz
                    self.command_rotation = np.matmul(self.command_rotation, delta_rotation)

                    timestamp = time.time() - self.init_time

                    # Compute IK to get target joint positions
                    target_joints = self._compute_ik_for_pose(
                        self.command_xyz,
                        self.command_rotation,
                        initial_joints=self.current_joints
                    )

                    if target_joints is None:
                        print(f"[WARNING] IK failed at step {self.action_steps}, skipping this command")
                        ik_failure_count += 1
                        time.sleep(self.control_time_step)
                        continue
                    
                    ik_success_count += 1
                    
                    # Verify IK solution (optional, for debugging)
                    if self.args.verify_ik and self.action_steps % 10 == 0:
                        quat_wxyz = mat2quat(self.command_rotation)
                        target_pose_for_verify = SimPose.from_pq(
                            position=self.command_xyz,
                            quaternion=quat_wxyz,
                            device=self.ik_solver.device
                        )
                        pos_error, ori_error = self.ik_solver.verify_ik_solution(
                            torch.tensor([target_joints], device=self.ik_solver.device),
                            target_pose_for_verify
                        )
                        print(f"[DEBUG] Step {self.action_steps}: "
                              f"pos_err={pos_error*1000:.3f}mm, "
                              f"ori_err={np.rad2deg(ori_error):.3f}deg")

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
                            "joints": target_joints.copy(),
                        },
                        "gripper_control": control_gripper
                    }

                    # Collect data
                    self.data_collector.update_data_dict(
                        instruction=self.instruction,
                        action=save_action,
                        timestamp=timestamp,
                    )

                    # Send joint command to robot using asynchronous waypoint control
                    waypoint = JointWaypoint(target_joints.tolist(), relative_dynamics_factor=0.2)
                    waypoint_motion = JointWaypointMotion([waypoint])
                    self.robot.move(waypoint_motion, asynchronous=True)
                    
                    # Update current joints for next IK warm start
                    self.current_joints = target_joints

                    # Control gripper (if available)
                    if hasattr(self, 'gripper_controller') and self.gripper_controller is not None:
                        if abs(control_gripper - self.last_gripper_state) > 0.02:
                            gripper_width = 0.08 * control_gripper  # Scale to gripper range
                            try:
                                if control_gripper < 0.5:
                                    # Close/grasp
                                    self.gripper_controller.grasp(force=50.0, epsilon_inner=0.08, epsilon_outer=0.08, async_call=True)
                                else:
                                    # Open to specified width
                                    self.gripper_controller.move(gripper_width, speed=0.12, async_call=True)
                                self.last_gripper_state = control_gripper
                            except Exception as e:
                                print(f"[WARNING] Gripper control failed: {e}")

                    self.action_steps += 1
                    
                    # Log progress when no camera/space mouse (for non-interactive mode)
                    if not self.use_space_mouse and not self.args.use_cameras:
                        if self.action_steps % 10 == 0:
                            print(f"[INFO] Step {self.action_steps}/{self.args.max_action_steps} | "
                                  f"IK success: {ik_success_count}/{ik_success_count + ik_failure_count}")
                    
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
                        time.sleep(0.1)  # Wait for recovery
                        print("[INFO] Error recovery successful")
                    except Exception as recover_error:
                        print(f"[WARNING] Error recovery failed: {recover_error}")
                    
                    time.sleep(self.control_time_step)
                    self.ee_pose_init()
                    continue
        finally:
            # Print statistics
            total_ik = ik_success_count + ik_failure_count
            if total_ik > 0:
                success_rate = 100.0 * ik_success_count / total_ik
                print(f"\n[INFO] IK Statistics:")
                print(f"  Success: {ik_success_count}")
                print(f"  Failure: {ik_failure_count}")
                print(f"  Success Rate: {success_rate:.2f}%")
            
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
            "control_type": "ik_based_joint_control",
            "ik_solver": "SimAlignedPandaIKSolver",
            "ik_solver_mode": self.ik_solver.alignment_mode,
            "ik_max_iterations": self.ik_solver.max_iterations,
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
    # Ensure robot assets are downloaded
    ensure_assets_downloaded("panda")
    
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
        JointWaypoint([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], relative_dynamics_factor=0.1)
    ])
    robot.move(home_motion)

    # Create data collection instance
    collection = FrankyDataCollectionWithIK(
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
