"""
VLA deployment script with IK-based joint control.
This script uses IK to convert EE pose commands to joint commands,
aligning the low-level control interface between real robot and simulation.

Adapted from frankapy/deploy/query_vla_with_ik.py for franky robot interface.
"""
import os
import sys

# Remove ROS2 and OpenRobots paths to avoid interference with conda packages
sys.path = [p for p in sys.path if not any(x in p for x in ['/opt/ros', 'franka_ros2_ws', '/opt/openrobots'])]

import time
import json
import cv2
import imageio
import requests
import numpy as np
import torch
import tyro
from pathlib import Path
from dataclasses import dataclass
from collections import deque
from PIL import Image as PImage
from transforms3d.euler import euler2quat, euler2mat, quat2euler, mat2euler
from transforms3d.quaternions import mat2quat, quat2mat

# Import franky robot control
from franky import (
    Robot, Affine,
    JointWaypointMotion, JointWaypoint,
    RelativeDynamicsFactor
)

# Import local modules
from franky_control.driver import RealsenseAPI, REALSENSE_AVAILABLE
from franky_control.robot import GripperController
from franky_control.robot.constants import FC
from franky_control.kinematics import PANDA_URDF_PATH, ensure_assets_downloaded
from franky_control.kinematics.panda_ik_solver import create_sim_aligned_ik_solver, SimPose

# Try to import json_numpy for numpy array serialization
try:
    import json_numpy
except ImportError:
    print("[WARNING] json_numpy not available, using custom serialization")
    json_numpy = None


@dataclass
class Args:
    """VLA deployment with IK arguments."""
    instruction: str = "test"  # Task instruction
    robot_ip: str = FC.ROBOT_IP  # Robot IP address
    ctrl_freq: float = FC.DEFAULT_CONTROL_FREQUENCY  # Control frequency in Hz
    record_dir: str = f"{FC.DEFAULT_LOG_DIR}/vla_deploy_ik"  # Directory to save logs
    max_steps: int = FC.MAX_EPISODE_STEPS  # Maximum deployment steps
    vla_server_ip: str = "localhost"  # VLA server IP address
    vla_server_port: int = 9876  # VLA server port
    episode_idx: str = "0"  # Episode index (string for directory naming)
    chunk_size: int = 16  # Action chunk size for VLA
    use_cameras: bool = True  # Use RealSense cameras
    use_gripper: bool = True  # Use Franka gripper
    pos_scale: float = 1.0  # Position action scale (1.0 = raw VLA output)
    rot_scale: float = 1.0  # Rotation action scale (1.0 = raw VLA output)
    # IK-related arguments
    use_gpu_ik: bool = False  # Use GPU for IK solver
    verify_ik: bool = False  # Verify IK solution with FK (for debugging)
    use_joint_filter: bool = False  # Use low-pass filter for joint commands
    filter_alpha: float = 0.4  # Low-pass filter smoothing factor (0-1, lower=smoother)


class VLADeployWithIK:
    """VLA deployment class that uses IK to convert EE pose to joint positions.
    
    Uses SimAlignedPandaIKSolver - fully aligned with ManiSkill simulation kinematics.
    This ensures consistency between simulation training and real robot deployment.
    """
    
    def __init__(self, args: Args):
        self.args = args
        self.observation_window = deque(maxlen=2)
        
        # Initialize robot
        print(f"[INFO] Connecting to robot at {args.robot_ip}")
        self.robot = Robot(args.robot_ip)
        self.robot.recover_from_errors()
        
        # Initialize cameras
        self.cameras = None
        if args.use_cameras:
            if REALSENSE_AVAILABLE:
                try:
                    self.cameras = RealsenseAPI(height=FC.CAMERA_HEIGHT, width=FC.CAMERA_WIDTH, fps=FC.CAMERA_FPS)
                    print(f"[INFO] Initialized {self.cameras.get_num_cameras()} camera(s)")
                except Exception as e:
                    print(f"[WARNING] Failed to initialize cameras: {e}")
            else:
                print("[WARNING] RealSense not available")
        
        # Initialize gripper
        self.gripper_controller = None
        if args.use_gripper:
            try:
                self.gripper_controller = GripperController(args.robot_ip)
                print("[INFO] Initializing gripper...")
                if self.gripper_controller.home():
                    print("[INFO] Gripper ready")
                    self.gripper_controller.open()
                else:
                    print("[WARNING] Gripper homing failed")
                    self.gripper_controller = None
            except Exception as e:
                print(f"[WARNING] Failed to initialize gripper: {e}")
                self.gripper_controller = None
        
        # Record settings
        self.record_dir = args.record_dir
        self.episode_idx = args.episode_idx
        self.chunk_size = args.chunk_size
        os.makedirs(self.record_dir, exist_ok=True)
        
        # Pose tracking
        self.init_xyz = None
        self.init_rotation = None
        self.command_xyz = None
        self.command_rotation = None
        self.current_joints = None  # For IK warm start
        self.filtered_joints = None  # For joint filtering
        self.actions_list = []
        self.actions_record_list = []
        
        # Control parameters
        self.max_steps = args.max_steps
        self.ctrl_freq = args.ctrl_freq
        self.control_time_step = 1.0 / self.ctrl_freq
        self.act_url = f"http://{args.vla_server_ip}:{args.vla_server_port}/act"
        self.init_time = time.time()
        self.record_image = []
        self.gripper_is_closed = False  # Simple open/close state tracking
        
        # Scaling factors
        self.pos_scale = args.pos_scale
        self.rot_scale = args.rot_scale
        
        # Joint filtering parameters
        self.use_joint_filter = args.use_joint_filter
        self.filter_alpha = args.filter_alpha
        
        # IK statistics
        self.ik_success_count = 0
        self.ik_failure_count = 0
        
        # Initialize IK solver
        self._init_ik_solver()

    def _init_ik_solver(self):
        """Initialize the simulation-aligned Panda IK solver."""
        urdf_path = PANDA_URDF_PATH
        
        # Use CPU for IK solver (more stable for real-time control)
        device = "cuda" if self.args.use_gpu_ik else "cpu"
        
        print(f"[INFO] Initializing Simulation-Aligned Panda IK solver on device: {device}")
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

    def _jpeg_mapping(self, img):
        """Apply JPEG compression/decompression for training alignment."""
        img = cv2.imencode('.jpg', img)[1].tobytes()
        return cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

    def update_observation_window(self):
        """Update observation window with current robot state and images."""
        # Get camera images
        if self.cameras is not None:
            images = self.cameras.get_rgb()
            # Concatenate images for visualization
            if len(images) >= 2:
                mix_image = np.concatenate([images[1], images[0]], axis=1)
            else:
                mix_image = images[0]
            self.record_image.append(mix_image)
        else:
            images = np.zeros((1, FC.CAMERA_HEIGHT, FC.CAMERA_WIDTH, 3), dtype=np.uint8)
            mix_image = images[0]
            self.record_image.append(mix_image)
        
        # Get robot state
        current_pose = self.robot.current_pose
        ee_affine = current_pose.end_effector_pose
        ee_matrix = np.array(ee_affine.matrix)
        
        joint_state = self.robot.current_joint_state
        joints = np.array(joint_state.position)
        
        # Get gripper width (read actual state from gripper)
        if self.gripper_controller is not None:
            gripper_width = np.array([self.gripper_controller.width])
        else:
            gripper_width = np.array([FC.GRIPPER_MAX_WIDTH])
        
        self.observation_window.append({
            'ee_pose_T': ee_matrix,  # np shape (4,4)
            'joints': joints,  # np shape (7,)
            'gripper_width': gripper_width,  # np shape (1,)
            'instruction': self.args.instruction,  # string
            'images': images.astype(np.uint8)  # support multi camera
        })

    def ee_pose_init(self):
        """Initialize the end-effector pose and joint positions."""
        time.sleep(0.5)
        current_pose = self.robot.current_pose
        
        ee_affine = current_pose.end_effector_pose
        self.init_xyz = np.array(ee_affine.translation)
        self.init_rotation = np.array(ee_affine.matrix[:3, :3])
        
        self.command_xyz = self.init_xyz.copy()
        self.command_rotation = self.init_rotation.copy()
        
        # Get initial joint positions for IK warm start
        joint_state = self.robot.current_joint_state
        self.current_joints = np.array(joint_state.position)
        
        # Initialize filtered joints
        if self.use_joint_filter and self.filtered_joints is None:
            self.filtered_joints = self.current_joints.copy()
            print(f"[INFO] Joint filter enabled with alpha={self.filter_alpha}")
        
        print(f"[INFO] Initial joints: {self.current_joints}")
        print(f"[INFO] Initial EE position: {self.init_xyz}")

    def robot_init(self):
        """Initialize robot to home position and setup joint control mode."""
        # Move to home position
        print("[INFO] Moving to home position...")
        home_motion = JointWaypointMotion([
            JointWaypoint(list(FC.HOME_JOINTS))
        ], relative_dynamics_factor=RelativeDynamicsFactor(
            velocity=FC.DEFAULT_VELOCITY_FACTOR, 
            acceleration=FC.DEFAULT_ACCELERATION_FACTOR, 
            jerk=FC.DEFAULT_JERK_FACTOR
        ))
        self.robot.move(home_motion)
        
        # Open gripper
        if self.gripper_controller is not None:
            self.gripper_controller.open()
        
        input("[INFO] Press enter to continue")
        
        # Set joint impedance for compliant control
        # Using custom gains similar to FrankaArm for IK-based control
        # Set collision behavior
        self.robot.set_collision_behavior(
            list(FC.COLLISION_TORQUE_LOWER),
            list(FC.COLLISION_TORQUE_UPPER),
            list(FC.COLLISION_FORCE_LOWER),
            list(FC.COLLISION_FORCE_UPPER)
        )

    def _send_action_request(self, observation):
        """Send observation to VLA server and get action."""
        payload = {
            "ee_pose_T": observation['ee_pose_T'],
            "joints": observation['joints'],
            "gripper_width": observation['gripper_width'],
            "images": observation['images'],
            "instruction": observation['instruction'],
        }
        
        if json_numpy is not None:
            payload_string = json_numpy.dumps(payload)
        else:
            # Fallback: convert numpy arrays to lists
            payload_serializable = {}
            for k, v in payload.items():
                if isinstance(v, np.ndarray):
                    payload_serializable[k] = v.tolist()
                else:
                    payload_serializable[k] = v
            payload_string = json.dumps(payload_serializable)
        
        response = requests.post(
            self.act_url,
            data=payload_string,
            headers={"Content-Type": "application/json"},
            timeout=1000,
        )
        
        if response.status_code == 200:
            if json_numpy is not None:
                response_data = json_numpy.loads(response.text)
            else:
                response_data = json.loads(response.text)
            action = np.array(response_data['actions'])
            return action
        else:
            raise TimeoutError(f"VLA server error: {response.status_code}")

    def run_inference_loop(self):
        """Main inference loop using IK-based joint control."""
        step = 0
        self.ee_pose_init()
        print("[INFO] Starting inference loop with IK-based joint control...")
        
        try:
            while step < self.max_steps:
                loop_start_time = time.time()
                
                self.update_observation_window()
                observation = self.observation_window[-1]
                
                # Get action from VLA server
                if len(self.actions_list) == 0:
                    t1 = time.time()
                    
                    try:
                        action = self._send_action_request(observation)
                    except Exception as e:
                        print(f"[ERROR] VLA request failed: {e}")
                        time.sleep(self.control_time_step)
                        continue
                    
                    # Handle action chunking
                    if len(action.shape) == 1:
                        self.actions_list.append(action)
                    else:
                        for idx in range(min(action.shape[0], self.chunk_size)):
                            self.actions_list.append(action[idx])
                    
                    print(f"[INFO] VLA inference time: {time.time() - t1:.3f}s, action shape: {action.shape}")
                
                action = self.actions_list.pop(0)
                
                # Parse action: [delta_xyz (3), delta_euler (3), gripper (1)]
                delta_xyz = action[:3] * self.pos_scale
                delta_euler = action[3:6] * self.rot_scale
                gripper = action[-1]
                
                # Compute delta rotation
                delta_rotation = euler2mat(delta_euler[0], delta_euler[1], delta_euler[2], 'sxyz')
                
                # Update command pose
                self.command_xyz = self.command_xyz + delta_xyz
                self.command_rotation = np.matmul(self.command_rotation, delta_rotation)
                
                try:
                    # Compute IK to get target joint positions
                    target_joints = self._compute_ik_for_pose(
                        self.command_xyz,
                        self.command_rotation,
                        initial_joints=self.current_joints
                    )
                    
                    if target_joints is None:
                        print(f"[WARNING] IK failed at step {step}, skipping this command")
                        self.ik_failure_count += 1
                        time.sleep(self.control_time_step)
                        continue
                    
                    self.ik_success_count += 1
                    
                    # Apply low-pass filter to joint commands for smoothness
                    if self.use_joint_filter:
                        self.filtered_joints = (self.filter_alpha * target_joints + 
                                               (1 - self.filter_alpha) * self.filtered_joints)
                        filtered_target_joints = self.filtered_joints.copy()
                    else:
                        filtered_target_joints = target_joints.copy()
                    
                    # Verify IK solution (optional, for debugging)
                    if self.args.verify_ik and step % 10 == 0:
                        quat_wxyz = mat2quat(self.command_rotation)
                        target_pose_for_verify = SimPose.from_pq(
                            position=self.command_xyz,
                            quaternion=quat_wxyz,
                            device=self.ik_solver.device
                        )
                        target_joints_tensor = torch.from_numpy(target_joints).unsqueeze(0).to(self.ik_solver.device)
                        pos_error, ori_error = self.ik_solver.verify_ik_solution(
                            target_joints_tensor,
                            target_pose_for_verify
                        )
                        print(f"[DEBUG] Step {step}: "
                              f"pos_err={pos_error*1000:.3f}mm, "
                              f"ori_err={np.rad2deg(ori_error):.3f}deg")
                    
                    # Send joint command to robot using asynchronous waypoint control
                    waypoint = JointWaypoint(filtered_target_joints.tolist())
                    waypoint_motion = JointWaypointMotion(
                        [waypoint],
                        relative_dynamics_factor=RelativeDynamicsFactor(
                            velocity=0.18,
                            acceleration=0.13,
                            jerk=0.11
                        )
                    )
                    self.robot.move(waypoint_motion, asynchronous=True)
                    
                    # Update current joints for next IK warm start (use unfiltered for IK)
                    self.current_joints = target_joints
                    
                    # Control gripper: simple open/close based on gripper action value
                    # gripper < 0.5 -> close, gripper >= 0.5 -> open
                    if self.gripper_controller is not None:
                        should_close = gripper < 0.5
                        if should_close != self.gripper_is_closed:
                            try:
                                if should_close:
                                    self.gripper_controller.grasp(async_call=True, force=FC.GRIPPER_DEFAULT_FORCE, 
                                                                  epsilon_inner=0.06, epsilon_outer=0.06)
                                else:
                                    self.gripper_controller.open(async_call=True)
                                self.gripper_is_closed = should_close
                            except Exception as e:
                                print(f"[WARNING] Gripper control failed: {e}")
                    
                except KeyboardInterrupt:
                    print("[INFO] Deployment stopped by user")
                    break
                except Exception as e:
                    print(f"[WARNING] Move failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Try to recover
                    try:
                        self.robot.recover_from_errors()
                    except:
                        pass
                    
                    self.ee_pose_init()
                    time.sleep(self.control_time_step)
                    continue
                
                # Log action
                print(f"[STEP {step}] delta_xyz: {delta_xyz}, delta_euler: {delta_euler}, gripper: {gripper:.3f}")
                step_record = {
                    "step": step,
                    "delta_xyz": delta_xyz.tolist(),
                    "delta_euler": delta_euler.tolist(),
                    "gripper": float(gripper) if np.isscalar(gripper) else gripper.tolist(),
                    "target_joints": target_joints.tolist(),  # Also record joint positions
                }
                self.actions_record_list.append(step_record)
                step += 1
                
                # Maintain control frequency
                elapsed = time.time() - loop_start_time
                if elapsed < self.control_time_step:
                    time.sleep(self.control_time_step - elapsed)
                    
        except Exception as e:
            print(f"[ERROR] Inference loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Stop robot
            try:
                self.robot.stop()
            except:
                pass
        
        # Print IK statistics
        total_ik = self.ik_success_count + self.ik_failure_count
        if total_ik > 0:
            success_rate = 100.0 * self.ik_success_count / total_ik
            print(f"\n[INFO] IK Statistics:")
            print(f"  Success: {self.ik_success_count}")
            print(f"  Failure: {self.ik_failure_count}")
            print(f"  Success Rate: {success_rate:.2f}%")
        else:
            success_rate = 0
        
        print('[INFO] Inference loop finished')
        
        # Save logs
        self._save_logs(success_rate)
    
    def _save_logs(self, ik_success_rate: float):
        """Save deployment logs."""
        # Save video
        video_logpath = os.path.join(self.record_dir, "log_policy_deploy_ik.mp4")
        print(f"[INFO] Saving video to {video_logpath}")
        if len(self.record_image) > 0:
            imageio.mimsave(video_logpath, self.record_image, fps=self.ctrl_freq)
        
        # Save actions
        actions_logpath = os.path.join(self.record_dir, "log_policy_output_ik.json")
        with open(actions_logpath, 'w') as f:
            json.dump(self.actions_record_list, f, indent=4)
        
        # Save IK statistics
        ik_stats = {
            "ik_success_count": self.ik_success_count,
            "ik_failure_count": self.ik_failure_count,
            "ik_success_rate": ik_success_rate,
            "ik_solver_mode": self.ik_solver.alignment_mode,
            "ik_max_iterations": self.ik_solver.max_iterations,
        }
        ik_stats_path = os.path.join(self.record_dir, "ik_statistics.json")
        with open(ik_stats_path, 'w') as f:
            json.dump(ik_stats, f, indent=4)
        
        print(f"[INFO] Logs saved to {self.record_dir}")
    
    def close(self):
        """Clean up resources."""
        if self.cameras is not None:
            self.cameras.close()


def main(args: Args):
    """Main function."""
    # Ensure robot assets are downloaded
    ensure_assets_downloaded("panda")
    
    # Setup record directory
    args.record_dir = os.path.join(args.record_dir, args.episode_idx)
    os.makedirs(args.record_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.record_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Save git commit (if in git repo)
    try:
        os.system(f'git rev-parse HEAD > {os.path.join(args.record_dir, "git_commit.txt")} 2>/dev/null')
    except:
        pass
    
    # Create and run agent
    agent = VLADeployWithIK(args)
    try:
        agent.robot_init()
        agent.run_inference_loop()
    finally:
        agent.close()


if __name__ == '__main__':
    args = tyro.cli(Args)
    main(args)
