"""
High-frequency robot data collection script using multiprocessing.

This implementation uses a separate process for robot control to avoid
interference from franky's real-time control thread, enabling 20-30Hz control.

Architecture:
- Main Process: SpaceMouse input, IK computation, data collection
- Control Process: Robot motion commands (isolated from real-time interference)
"""
import os
import sys

# Remove ROS2 and OpenRobots paths to avoid interference with conda packages
sys.path = [p for p in sys.path if not any(x in p for x in ['/opt/ros', 'franka_ros2_ws', '/opt/openrobots'])]

import time
import json
import copy
import traceback
import numpy as np
import tyro
from dataclasses import dataclass
from multiprocessing import Process, Queue, Value
import ctypes
from transforms3d.euler import euler2quat, euler2mat, mat2euler
from transforms3d.quaternions import mat2quat, quat2mat
import torch

# Import local modules
from franky_control.driver import SpaceMouse, RealsenseAPI, SPACEMOUSE_AVAILABLE, REALSENSE_AVAILABLE
from franky_control.data_collection.data_collector_v2 import DataCollector
from franky_control.robot.constants import FC
from franky_control.utils import get_next_episode_idx, ensure_dir
from franky_control.kinematics import PANDA_URDF_PATH, ensure_assets_downloaded
from franky_control.kinematics.panda_ik_solver import create_sim_aligned_ik_solver, SimPose

# Import franky modules
from franky import Robot, JointWaypointMotion, JointWaypoint, RelativeDynamicsFactor, Gripper

@dataclass 
class Args:
    """Data collection script arguments."""
    task_name: str = "test" # Task name for the dataset
    instruction: str = "test" # Instruction for data collection
    robot_ip: str = FC.ROBOT_IP  # Robot IP address
    dataset_dir: str = FC.DEFAULT_DATASET_DIR  # Directory to save dataset
    min_action_steps: int = FC.MIN_EPISODE_STEPS  # Minimum action_steps for data collection
    max_action_steps: int = FC.MAX_EPISODE_STEPS  # Maximum action_steps for data collection  
    episode_idx: int = -1  # Episode index to save data (-1 for auto-increment)
    pos_scale: float = FC.DEFAULT_POS_SCALE  # The scale of xyz action
    rot_scale: float = FC.DEFAULT_ROT_SCALE  # The scale of rotation action
    use_gpu_ik: bool = False  # Use GPU for IK solver
    verify_ik: bool = False  # Verify IK solution with FK (for debugging)
    control_frequency: float = 10.0  # Control frequency in Hz (10-15Hz recommended for smooth motion)
    use_joint_filter: bool = False  # Use low-pass filter for joint commands
    filter_alpha: float = 0.5  # Low-pass filter smoothing factor (0-1)
    use_space_mouse: bool = True  # Use SpaceMouse for control
    use_cameras: bool = True  # Use RealSense cameras
    use_gripper: bool = True  # Use Franka gripper

def robot_control_process(robot_ip: str, command_queue: Queue, state_queue: Queue, 
                          running: Value, gripper_queue: Queue = None, error_queue: Queue = None):
    """
    Separate process for robot control.
    
    This runs in isolation from the main Python process to avoid
    being preempted by franky's real-time control thread.
    """
    try:
        
        print("[Control Process] Connecting to robot...")
        robot = Robot(robot_ip)
        robot.recover_from_errors()
        print("[Control Process] Robot connected")
        
        # Initialize gripper if requested
        gripper = None
        if gripper_queue is not None:
            try:
                gripper = Gripper(robot_ip)
                gripper.homing()
                print("[Control Process] Gripper initialized")
            except Exception as e:
                print(f"[Control Process] Gripper init failed: {e}")
        
        current_joints = list(robot.current_joint_positions)
        print(f"[Control Process] Started with joints: {current_joints[:3]}...")
        
        # Send initial state
        state_queue.put({
            'joints': current_joints,
            'gripper_width': gripper.width if gripper else 0.04
        })
        print("[Control Process] Initial state sent")
        
        # Start with return_when_finished=False to keep control session alive
        dynamics = RelativeDynamicsFactor(
            velocity=FC.DEFAULT_VELOCITY_FACTOR,
            acceleration=FC.DEFAULT_ACCELERATION_FACTOR,
            jerk=FC.DEFAULT_JERK_FACTOR
        )
        motion = JointWaypointMotion(
            [JointWaypoint(current_joints)],
            relative_dynamics_factor=dynamics,
            return_when_finished=False, # When the motion is finished, do not return
        )
        robot.move(motion, asynchronous=True)
        print("[Control Process] Control loop started")
        
        gripper_is_closed = False
        cmd_count = 0
        error_count = 0
        last_error_time = 0
        loop_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 100
        
        while running.value:
            loop_count += 1
            if loop_count % 100 == 1:
                print(f"[Control Process] Loop #{loop_count}, cmd_count={cmd_count}, queue_size~={command_queue.qsize()}")
            
            # Check robot state more frequently (every 10 loops) for faster error detection
            if loop_count % 10 == 0:
                try:
                    robot_state = robot.state
                    # franky.Errors is non-iterable; rely on string repr '[]' when empty
                    raw_errors = getattr(robot_state, "current_errors", None)
                    errors_str = str(raw_errors) if raw_errors is not None else "[]"
                    has_errors = errors_str not in ("[]", "None", "")

                    if has_errors:
                        current_time = time.time()
                        if current_time - last_error_time > 0.5:
                            print(f"[Control Process] Robot has errors: {errors_str}, recovering...")
                            last_error_time = current_time
                            
                            # Attempt recovery
                            robot.recover_from_errors()
                            error_count += 1
                            consecutive_errors += 1
                            
                            if consecutive_errors >= max_consecutive_errors:
                                print(f"[Control Process] Too many consecutive errors ({consecutive_errors}), may need manual intervention")
                            
                            # Get current position and restart motion
                            current_joints = list(robot.current_joint_positions)
                            motion = JointWaypointMotion(
                                [JointWaypoint(current_joints)],
                                relative_dynamics_factor=dynamics,
                                return_when_finished=False,
                            )
                            robot.move(motion, asynchronous=True)
                            print("[Control Process] Motion restarted after recovery")
                            
                            # Skip command processing this iteration
                            continue
                    else:
                        # Reset consecutive error counter on successful state check
                        consecutive_errors = 0
                except Exception as state_err:
                    # State check failed, try to continue
                    print(f"[Control Process] State check error: {state_err}")
            
            # Get joint command (non-blocking with short timeout)
            try:
                target_joints = command_queue.get(timeout=0.001)
                cmd_count += 1
                
                # Debug: print every 50 commands
                if cmd_count % 50 == 0:
                    print(f"[Control Process] Received cmd #{cmd_count}, joints[:3]={target_joints[:3]}")
                
                # Send motion command
                motion = JointWaypointMotion(
                    [JointWaypoint(target_joints)],
                    relative_dynamics_factor=dynamics,
                    return_when_finished=False,
                )
                robot.move(motion, asynchronous=True)
                
                # Reset consecutive error counter on successful move
                consecutive_errors = 0
                
            except Exception as move_err:
                # Check if it's a control exception
                if "ControlException" in str(type(move_err).__name__) or \
                   "reflex" in str(move_err).lower() or \
                   "control" in str(move_err).lower():
                    import time
                    current_time = time.time()
                    if current_time - last_error_time > 0.5:
                        print(f"[Control Process] Motion error: {move_err}, attempting recovery...")
                        last_error_time = current_time
                    
                    try:
                        # Recover from error
                        robot.recover_from_errors()
                        error_count += 1
                        consecutive_errors += 1
                        
                        # Restart motion at current position
                        current_joints = list(robot.current_joint_positions)
                        motion = JointWaypointMotion(
                            [JointWaypoint(current_joints)],
                            relative_dynamics_factor=dynamics,
                            return_when_finished=False,
                        )
                        robot.move(motion, asynchronous=True)
                        print("[Control Process] Recovered and restarted motion")
                    except Exception as recovery_err:
                        print(f"[Control Process] Recovery failed: {recovery_err}")
                        consecutive_errors += 1
                elif "Empty" not in str(type(move_err).__name__):
                    # Other non-empty queue exceptions
                    pass
            
            # Handle gripper commands
            if gripper is not None and gripper_queue is not None:
                try:
                    gripper_cmd = gripper_queue.get_nowait()
                    should_close = gripper_cmd < 0.5
                    if should_close != gripper_is_closed:
                        if should_close:
                            gripper.grasp_async(width=0.0, speed=0.1, force=50.0, epsilon_inner=0.08, epsilon_outer=0.08)
                        else:
                            gripper.open_async(speed=0.1)
                        gripper_is_closed = should_close
                except:
                    pass
            
            # Send state update periodically
            try:
                state_queue.put_nowait({
                    'joints': list(robot.current_joint_positions),
                    'gripper_width': gripper.width if gripper else 0.04,
                    'error_count': error_count,
                })
            except:
                pass  # Queue full
        
        robot.stop()
        print(f"[Control Process] Stopped normally (errors recovered: {error_count})")
        
    except Exception as e:
        print(f"[Control Process] ERROR: {e}")
        traceback.print_exc()

class HighFreqDataCollection:
    """
    High-frequency data collection using multiprocessing.
    
    Achieves 20-30Hz by isolating robot control in a separate process.
    """
    
    def __init__(self, args: Args, cameras=None):
        self.args = args
        self.cameras = cameras
        self.robot_ip = args.robot_ip
        
        # SpaceMouse will be initialized AFTER control process starts
        self.space_mouse = None
        self.use_space_mouse = args.use_space_mouse
        
        # Control parameters
        self.control_frequency = args.control_frequency
        self.control_time_step = 1.0 / self.control_frequency
        
        # State
        self.episode_idx = 0
        self.action_steps = 0
        self.instruction = args.instruction
        self.init_xyz = None
        self.init_rotation = None
        self.command_xyz = None
        self.command_rotation = None
        self.current_joints = None
        self.filtered_joints = None
        self.gripper_is_closed = False
        self.init_time = time.time()
        
        # Joint filtering
        self.use_joint_filter = args.use_joint_filter
        self.filter_alpha = args.filter_alpha
        
        # Multiprocessing
        self.command_queue = Queue(maxsize=10)
        self.state_queue = Queue(maxsize=10)
        self.gripper_queue = Queue(maxsize=10) if args.use_gripper else None
        self.running = Value(ctypes.c_bool, False)
        self.control_process = None
        
        # Initialize IK solver
        self._init_ik_solver()
        
        # Data collector - disable encoding in control loop for speed
        self.data_collector = DataCollector(
            robot=None,  # We don't have direct robot access in main process
            cameras=cameras,
            gripper=None,  # Gripper in control process
            is_image_encoded=False,  # Encode at save time for speed
        )
        
        # Track gripper state locally
        self.gripper_width = 0.04

    def _init_ik_solver(self):
        """Initialize the IK solver."""
        device = "cuda" if self.args.use_gpu_ik else "cpu"
        print(f"[INFO] Initializing IK solver on device: {device}")
        
        self.ik_solver = create_sim_aligned_ik_solver(
            urdf_path=PANDA_URDF_PATH,
            device=device,
        )
        print(f"[INFO] IK solver initialized: {self.ik_solver.alignment_mode}")

    def _compute_ik_for_pose(self, position, rotation_matrix, initial_joints=None):
        """Compute IK for given pose."""
        quat_wxyz = mat2quat(rotation_matrix)
        target_pose = SimPose.from_pq(
            position=position,
            quaternion=quat_wxyz,
            device=self.ik_solver.device
        )
        
        ik_solution = self.ik_solver.compute_ik(
            target_pose,
            initial_qpos=initial_joints
        )
        
        if ik_solution is not None:
            return ik_solution[0].cpu().numpy()
        return None

    def start_control_process(self):
        """Start the robot control process.
        
        IMPORTANT: ee_pose_init() must be called before this method
        to release the robot connection.
        """
        self.running.value = True
        self.control_process = Process(
            target=robot_control_process,
            args=(self.robot_ip, self.command_queue, self.state_queue, 
                  self.running, self.gripper_queue)
        )
        self.control_process.start()
        
        # Wait for control process to initialize and send initial state
        # Gripper homing can take 10+ seconds, so use longer timeout
        print("[INFO] Waiting for control process to initialize (gripper homing may take time)...")
        try:
            state = self.state_queue.get(timeout=30.0)  # Increased timeout for gripper homing
            # Update current_joints from control process (should match ee_pose_init)
            control_joints = np.array(state['joints'])
            print(f"[INFO] Control process started, joints: {control_joints[:3]}...")
        except:
            raise RuntimeError("Failed to get initial state from control process")
        
        # Initialize SpaceMouse AFTER control process starts (to avoid multiprocessing issues)
        if self.use_space_mouse and SPACEMOUSE_AVAILABLE:
            self.space_mouse = SpaceMouse(
                vendor_id=FC.SPACEMOUSE_VENDOR_ID, 
                product_id=FC.SPACEMOUSE_PRODUCT_ID
            )
            print("[INFO] SpaceMouse initialized")

    def stop_control_process(self):
        """Stop the robot control process."""
        self.running.value = False
        if self.control_process is not None:
            self.control_process.join(timeout=2.0)
            if self.control_process.is_alive():
                self.control_process.terminate()
            self.control_process = None

    def ee_pose_init(self):
        """Initialize end-effector pose from current robot state.
        
        IMPORTANT: This must be called BEFORE start_control_process()
        to avoid having two simultaneous robot connections.
        """
        from franky import Robot
        
        # Connect to robot to get initial pose and joints
        robot = Robot(self.robot_ip)
        robot.recover_from_errors()
        
        current_pose = robot.current_pose
        ee_affine = current_pose.end_effector_pose
        
        self.init_xyz = np.array(ee_affine.translation)
        self.init_rotation = np.array(ee_affine.matrix[:3, :3])
        self.command_xyz = self.init_xyz.copy()
        self.command_rotation = self.init_rotation.copy()
        
        # Also get initial joints for IK warm start
        self.current_joints = np.array(robot.current_joint_positions)
        
        # Initialize filtered joints
        if self.use_joint_filter:
            self.filtered_joints = self.current_joints.copy()
            print(f"[INFO] Joint filter enabled with alpha={self.filter_alpha}")
        
        print(f"[INFO] Initial EE position: {self.init_xyz}")
        print(f"[INFO] Initial joints: {self.current_joints[:3]}...")
        
        # IMPORTANT: Release robot connection before starting control process
        del robot

    def _apply_control_data_clip_and_scale(self, control_tensor, deadzone=0.0):
        """Clip and scale control data."""
        control_tensor = np.clip(control_tensor, -1.0, 1.0)
        scaled_tensor = np.zeros_like(control_tensor)
        positive_mask = (control_tensor >= deadzone)
        negative_mask = (control_tensor <= -deadzone)
        if deadzone < 1.0 and deadzone >= 0.0:
            scaled_tensor[positive_mask] = (control_tensor[positive_mask] - deadzone) / (1.0 - deadzone)
            scaled_tensor[negative_mask] = (control_tensor[negative_mask] + deadzone) / (1.0 - deadzone)
        return np.clip(scaled_tensor, -1.0, 1.0)

    def collect_data(self):
        """Main data collection loop."""
        input("Press enter to start high-frequency data collection")
        print(f"[INFO] Starting data collection at {self.control_frequency:.1f} Hz")
        print("[INFO] Using multiprocessing for high-frequency control")
        
        # Statistics
        ik_success_count = 0
        ik_failure_count = 0
        loop_times = []
        ik_times = []
        processing_times = []
        last_loop_start_time = None
        
        try:
            while True:
                loop_start_time = time.time()
                
                # Record loop interval
                if last_loop_start_time is not None:
                    loop_times.append(loop_start_time - last_loop_start_time)
                last_loop_start_time = loop_start_time
                
                # Read SpaceMouse
                if self.space_mouse is not None:
                    control = self.space_mouse.control
                    control_gripper = self.space_mouse.gripper_status
                    control_quit = self.space_mouse.quit_signal
                    # Debug: print raw spacemouse values every 50 steps
                    if self.action_steps % 50 == 0:
                        print(f"[DEBUG] Raw SpaceMouse control: {control}")
                        print(f"[DEBUG] Command EE position: {self.command_xyz}")
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
                # Use smaller deadzone (FC.SPACEMOUSE_DEADZONE=0.35 is too large)
                deadzone = 0.35  # Reduced from 0.35
                control_xyz = self._apply_control_data_clip_and_scale(control_xyz, deadzone)
                control_euler = self._apply_control_data_clip_and_scale(control_euler, deadzone)

                # Compute delta pose
                delta_xyz = control_xyz * self.args.pos_scale
                delta_euler = control_euler * self.args.rot_scale
                delta_rotation = euler2mat(delta_euler[0], delta_euler[1], delta_euler[2], 'sxyz')

                # Save previous command pose (for rollback on IK failure)
                prev_command_xyz = self.command_xyz.copy()
                prev_command_rotation = self.command_rotation.copy()

                # Update command pose
                self.command_xyz += delta_xyz
                self.command_rotation = np.matmul(self.command_rotation, delta_rotation)

                # Compute IK
                ik_start_time = time.time()
                target_joints = self._compute_ik_for_pose(
                    self.command_xyz,
                    self.command_rotation,
                    initial_joints=self.current_joints
                )
                ik_times.append(time.time() - ik_start_time)

                if target_joints is None:
                    # Rollback command pose to last valid position
                    self.command_xyz = prev_command_xyz
                    self.command_rotation = prev_command_rotation
                    print(f"[WARNING] IK failed at step {self.action_steps}, rolling back target pose")
                    ik_failure_count += 1
                    time.sleep(self.control_time_step)
                    continue
                
                ik_success_count += 1
                
                # Apply low-pass filter
                if self.use_joint_filter:
                    self.filtered_joints = (self.filter_alpha * target_joints + 
                                           (1 - self.filter_alpha) * self.filtered_joints)
                    filtered_target_joints = self.filtered_joints.copy()
                else:
                    filtered_target_joints = target_joints.copy()

                # Send command to control process (non-blocking)
                try:
                    self.command_queue.put_nowait(filtered_target_joints.tolist())
                    # Debug: print every 20 steps to see if commands are being sent
                    if self.action_steps % 20 == 0:
                        print(f"[DEBUG] Step {self.action_steps}: EE={self.command_xyz}, delta={delta_xyz}, joints[:3]={filtered_target_joints[:3]}")
                except Exception as e:
                    print(f"[DEBUG] Queue put failed: {e}")  # Queue full, skip

                # Send gripper command
                if self.gripper_queue is not None:
                    try:
                        self.gripper_queue.put_nowait(control_gripper)
                    except:
                        pass

                # Update state from control process
                try:
                    while not self.state_queue.empty():
                        state = self.state_queue.get_nowait()
                        self.current_joints = np.array(state['joints'])
                except:
                    pass

                # Update current joints for IK warm start
                self.current_joints = target_joints
                
                # Record observation using external state (robot is in another process)
                # We use command_xyz/rotation as EE pose since we don't have real robot state here
                ee_quat = mat2quat(self.command_rotation)  # [w, x, y, z]
                gripper_width = 0.04 if control_gripper < 0.5 else 0.08
                
                self.data_collector.record_observation_external(
                    instruction=self.args.instruction,
                    ee_position=self.command_xyz,
                    ee_orientation=ee_quat,
                    joints=filtered_target_joints,
                    gripper_width=gripper_width,
                )
                
                # Record action
                delta_quat = mat2quat(delta_rotation)
                self.data_collector.record_action(
                    delta_xyz=delta_xyz,
                    delta_euler=delta_euler,
                    delta_orientation=delta_quat,
                    abs_position=self.command_xyz,
                    abs_rotation=self.command_rotation,
                    gripper_control=control_gripper,
                    abs_joints=filtered_target_joints,
                )

                self.action_steps += 1
                processing_times.append(time.time() - loop_start_time)
                
                # Sleep to maintain frequency
                elapsed = time.time() - loop_start_time
                remaining = self.control_time_step - elapsed
                if remaining > 0:
                    time.sleep(remaining)
                    
        finally:
            # Print statistics
            total_ik = ik_success_count + ik_failure_count
            if total_ik > 0:
                print(f"\n[INFO] IK Statistics:")
                print(f"  Success: {ik_success_count}")
                print(f"  Failure: {ik_failure_count}")
                print(f"  Success Rate: {100.0 * ik_success_count / total_ik:.2f}%")
            
            if loop_times:
                loop_times_ms = np.array(loop_times) * 1000
                ik_times_ms = np.array(ik_times) * 1000
                processing_times_ms = np.array(processing_times) * 1000
                
                actual_freq = 1000.0 / np.mean(loop_times_ms)
                print(f"\n[INFO] Frequency Statistics:")
                print(f"  Target frequency: {self.control_frequency:.1f} Hz")
                print(f"  Actual frequency: {actual_freq:.2f} Hz")
                print(f"  Loop time: mean={np.mean(loop_times_ms):.2f}ms, std={np.std(loop_times_ms):.2f}ms")
                print(f"  IK time: mean={np.mean(ik_times_ms):.2f}ms")
                print(f"  Processing time: mean={np.mean(processing_times_ms):.2f}ms")

            # Cleanup
            if self.space_mouse is not None:
                self.space_mouse.close()
            if self.cameras is not None:
                self.cameras.close()


def main(args: Args):
    """Main function."""
    from franky import Robot, JointWaypointMotion, JointWaypoint, RelativeDynamicsFactor
    
    ensure_assets_downloaded("panda")
    
    # Initialize cameras
    cameras = None
    if args.use_cameras and REALSENSE_AVAILABLE:
        try:
            cameras = RealsenseAPI(FC.CAMERA_HEIGHT, FC.CAMERA_WIDTH, FC.CAMERA_FPS)
            print(f"[INFO] Initialized {cameras.get_num_cameras()} camera(s)")
        except Exception as e:
            print(f"[WARNING] Failed to initialize cameras: {e}")

    # Move robot to home position first
    print(f"[INFO] Connecting to robot at {args.robot_ip}")
    robot = Robot(args.robot_ip)
    robot.recover_from_errors()
    
    print("[INFO] Moving to home position...")
    home_motion = JointWaypointMotion([
        JointWaypoint(FC.RESET_JOINTS)
    ], relative_dynamics_factor=RelativeDynamicsFactor(
        velocity=FC.DEFAULT_VELOCITY_FACTOR, 
        acceleration=FC.DEFAULT_ACCELERATION_FACTOR, 
        jerk=FC.DEFAULT_JERK_FACTOR
    ))
    robot.move(home_motion)
    print("[INFO] Robot at home position")
    
    # Release robot connection before creating collection instance
    del robot

    # Create data collection instance
    collection = HighFreqDataCollection(args, cameras)
    
    # Initialize pose FIRST (before starting control process)
    # This gets initial EE pose and releases robot connection
    collection.ee_pose_init()
    
    # Start control process (after main process has released robot)
    collection.start_control_process()
    
    try:
        # Start collection
        collection.collect_data()
        
    finally:
        # Stop control process
        collection.stop_control_process()

    print(f"\n[INFO] Collected {collection.action_steps} steps")
    
    # Save data if enough steps collected
    if collection.action_steps >= args.min_action_steps:
        task_dir = os.path.join(args.dataset_dir, args.task_name)
        ensure_dir(task_dir)
        
        # Save data using DataCollector's save method
        episode_dir = collection.data_collector.save(
            save_dir=task_dir,
            episode_idx=args.episode_idx,
            metadata={
                "task_name": args.task_name,
                "instruction": args.instruction,
                "steps": collection.action_steps,
                "control_frequency": args.control_frequency,
            },
            use_hdf5=True,
            save_video=True,
        )
        print(f"\033[32m[SUCCESS] Saved {collection.action_steps} steps to {episode_dir}\033[0m")
    else:
        print(f"\033[33m[WARNING] Not enough steps ({collection.action_steps} < {args.min_action_steps}), data not saved\033[0m")
    
    return 0


if __name__ == "__main__":
    args = tyro.cli(Args)
    sys.exit(main(args))
