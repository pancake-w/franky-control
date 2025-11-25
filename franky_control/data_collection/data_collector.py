"""Data collector for robot teleoperation and data collection.

This module provides a simplified data collector that works with franky.Robot
without ROS dependencies.
"""

import cv2
import os
import time
import numpy as np
import imageio
from typing import Optional, Dict, Any
from franky import Robot


def get_numpy(data, device='cpu'):
    """Convert data to numpy array."""
    if hasattr(data, 'cpu'):
        return data.cpu().numpy()
    return np.array(data)


def to_numpy(data_dict, device='cpu'):
    """Recursively convert all data in dictionary to numpy arrays."""
    if isinstance(data_dict, dict):
        return {k: to_numpy(v, device) for k, v in data_dict.items()}
    elif isinstance(data_dict, list):
        return [to_numpy(item, device) for item in data_dict]
    else:
        return get_numpy(data_dict, device)


class DataCollector:
    """
    Data collector for robot teleoperation.
    
    Simplified version without ROS dependencies, compatible with franky.Robot.
    """
    
    def __init__(
        self,
        robot: Robot,
        cameras=None,
        is_image_encode: bool = False,
        *args,
        **kwargs
    ):
        """
        Initialize data collector.
        
        Args:
            robot: franky.Robot instance
            cameras: Camera wrapper instance (optional)
            is_image_encode: Whether to encode images as JPEG
        """
        self.robot = robot
        self.cameras = cameras
        self.is_image_encode = is_image_encode
        self.data_dict = self.get_empty_data_dict()
        self.device = 'cpu'
    
    def get_empty_data_dict(self) -> Dict[str, Any]:
        """Create empty data dictionary structure."""
        data_dict = {
            "task_info": {
                "instruction": [],
            },
            "action": {
                "end_effector": {
                    "delta_orientation": [],
                    "delta_position": [],
                    "delta_euler": [],
                    "abs_position": [],
                    "abs_euler": [],
                    "abs_joints": [],  # Added for IK-based control
                    "gripper_width": [],
                },
                "joint": {
                    "position": [],
                    "gripper_width": [],
                },
            },
            "observation": {
                "is_image_encode": self.is_image_encode,
                "rgb": [],
                "rgb_timestamp": [],
                "depth": [],
                "depth_timestamp": [],
            },
            "state": {
                "end_effector": {
                    "orientation": [],
                    "position": [],
                    "gripper_width": [],
                },
                "joint": {
                    "position": [],
                },
            },
        }
        return data_dict
    
    def clear_data(self):
        """Clear all collected data."""
        self.data_dict = self.get_empty_data_dict()
    
    def get_data(self) -> Dict[str, Any]:
        """Get collected data as numpy arrays."""
        return to_numpy(self.data_dict, self.device)
    
    def save_multi_cam_videos(self, rgb_array, base_path="videos", fps=30):
        """Save multi-camera videos using imageio.
        
        Args:
            rgb_array: Shape [frame_num, cam_num, H, W, 3]
            base_path: Directory to save videos
            fps: Frames per second
        """
        os.makedirs(base_path, exist_ok=True)
        
        if isinstance(rgb_array, list):
            rgb_array = np.stack(rgb_array, axis=0)
        
        if rgb_array.ndim != 5 or rgb_array.shape[-1] != 3:
            raise ValueError("rgb_array must be shape [frame_num, cam_num, H, W, 3]")
        
        frame_num, cam_num, H, W, _ = rgb_array.shape
        
        for cam_idx in range(cam_num):
            video_filename = os.path.join(base_path, f"cam_{cam_idx}.mp4")
            
            # Extract frames for this camera
            frames = [rgb_array[frame_idx, cam_idx].astype(np.uint8) 
                     for frame_idx in range(frame_num)]
            
            # Save video using imageio
            imageio.mimsave(video_filename, frames, fps=fps, codec='libx264')
            print(f"Saved camera {cam_idx} video to: {video_filename}")
    
    def save_data(
        self,
        save_path: str,
        episode_idx: int,
        is_compressed: bool = False,
        is_save_video: bool = True
    ):
        """
        Save data to disk.
        
        Args:
            save_path: Directory to save data
            episode_idx: Episode index
            is_compressed: Whether to use compressed format
            is_save_video: Whether to save video files
        """
        saving_data = to_numpy(self.data_dict, self.device)
        
        # Save numpy data
        save_func = np.savez_compressed if is_compressed else np.save
        np_path_data = os.path.join(save_path, f"data")
        save_func(np_path_data, saving_data)
        print(f"Saved data to {np_path_data}.{'npz' if is_compressed else 'npy'}")
        
        # Save videos
        if is_save_video:
            if self.is_image_encode:
                print("Warning: Images are encoded, cannot save video")
            elif self.cameras is not None and len(saving_data["observation"]["rgb"]) > 0:
                self.save_multi_cam_videos(saving_data["observation"]["rgb"], save_path)
        
        self.clear_data()
    
    def update_instruction(self, instruction: str):
        """Update instruction."""
        self.data_dict["task_info"]["instruction"].append(instruction)
    
    def update_rgb(self, timestamp: Optional[float] = None):
        """
        Update RGB and depth observations from cameras.
        
        Args:
            timestamp: Current timestamp (uses time.time() if None)
        """
        if self.cameras is None:
            return
        
        rgb = self.cameras.get_rgb()
        depth = self.cameras.get_depth()
        timestamp = time.time() if timestamp is None else timestamp
        
        if self.is_image_encode:
            # Encode as JPEG
            success, encoded_rgb = cv2.imencode(
                '.jpeg',
                get_numpy(rgb, self.device),
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            if not success:
                raise ValueError("JPEG encode error")
            rgb = np.frombuffer(encoded_rgb.tobytes(), dtype=np.uint8)
        
        self.data_dict["observation"]["rgb"].append(rgb)
        self.data_dict["observation"]["rgb_timestamp"].append(timestamp)
        self.data_dict["observation"]["depth"].append(depth)
        self.data_dict["observation"]["depth_timestamp"].append(timestamp)
    
    def update_state(self):
        """Update robot state (joints, pose, gripper)."""
        # Get joint positions
        joint_state = self.robot.current_joint_state
        joint_pos = np.array(joint_state.position)
        
        # Get end-effector pose
        current_pose = self.robot.current_pose
        ee_affine = current_pose.end_effector_pose
        ee_position = np.array(ee_affine.translation)
        ee_orientation = np.array(ee_affine.quaternion)  # [w, x, y, z]
        
        # Get gripper state (approximate, franky doesn't have direct gripper width API)
        # You may need to track this separately in your control loop
        gripper_state = 0.04  # Placeholder
        
        self.data_dict["state"]["joint"]["position"].append(joint_pos)
        self.data_dict["state"]["end_effector"]["position"].append(ee_position)
        self.data_dict["state"]["end_effector"]["orientation"].append(ee_orientation)
        self.data_dict["state"]["end_effector"]["gripper_width"].append(gripper_state)
    
    def update_action(self, save_action: Dict[str, Any]):
        """
        Update action data.
        
        Args:
            save_action: Dictionary containing action information with keys:
                - "delta": {"position", "orientation", "euler_angle"}
                - "abs": {"position", "euler_angle", "joints"}
                - "gripper_width"
        """
        action = self.data_dict['action']["end_effector"]
        action["delta_position"].append(save_action["delta"]["position"])
        action["delta_orientation"].append(save_action["delta"]["orientation"])
        action["delta_euler"].append(save_action["delta"]["euler_angle"])
        action["abs_position"].append(save_action["abs"]["position"])
        action["abs_euler"].append(save_action["abs"]["euler_angle"])
        
        # Store computed joint positions from IK
        if "joints" in save_action["abs"]:
            action["abs_joints"].append(save_action["abs"]["joints"])
        
        action["gripper_width"].append(save_action["gripper_width"])
    
    def update_data_dict(
        self,
        instruction: str,
        action: Dict[str, Any],
        timestamp: Optional[float] = None
    ):
        """
        Update all data at once.
        
        Args:
            instruction: Task instruction
            action: Action dictionary
            timestamp: Current timestamp
        """
        self.update_rgb(timestamp)
        self.update_instruction(instruction)
        self.update_state()
        self.update_action(action)
