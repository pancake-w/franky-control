"""Data collector for robot teleoperation and data collection.

Simplified data collector that handles:
- Robot state recording (joints, EE pose, gripper)
- Camera image recording (raw or JPEG encoded)
- Action recording
- Data saving to disk (numpy, npz, or HDF5)
"""

import os
import json
import time
import copy
import cv2
import numpy as np
import imageio
from typing import Optional, Dict, Any, List

from transforms3d.euler import euler2quat, mat2euler
from transforms3d.quaternions import mat2quat

from franky import Robot
from franky_control.utils import get_next_episode_idx, ensure_dir

# Optional HDF5 support
import h5py


def to_numpy(data, device='cpu'):
    """Recursively convert data to numpy arrays."""
    if isinstance(data, dict):
        return {k: to_numpy(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_numpy(item, device) for item in data]
    elif hasattr(data, 'cpu'):
        return data.cpu().numpy()
    return np.array(data)


def encode_image_jpeg(image: np.ndarray, quality: int = 95) -> np.ndarray:
    """Encode image as JPEG bytes.
    
    Args:
        image: RGB image array [H, W, 3]
        quality: JPEG quality (0-100)
        
    Returns:
        1D array of JPEG bytes
    """
    # OpenCV uses BGR, so convert from RGB
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("JPEG encoding failed")
    return encoded.flatten()


def decode_image_jpeg(encoded: np.ndarray) -> np.ndarray:
    """Decode JPEG bytes to image.
    
    Args:
        encoded: 1D array of JPEG bytes
        
    Returns:
        RGB image array [H, W, 3]
    """
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class DataCollector:
    """Simplified data collector for robot teleoperation.
    
    Collects:
    - Observations: RGB images (raw or JPEG encoded), depth, timestamps
    - States: joint positions, EE pose, gripper width
    - Actions: delta/absolute pose, gripper control
    
    Supports saving to:
    - NumPy (.npy / .npz)
    - HDF5 (.h5) for better compression and partial loading
    """
    
    def __init__(
        self,
        robot: Robot,
        cameras=None,
        gripper=None,
        is_image_encoded: bool = True,
        jpeg_quality: int = 95,
    ):
        """Initialize data collector.
        
        Args:
            robot: franky.Robot instance
            cameras: Camera wrapper (RealsenseAPI or similar)
            gripper: GripperController for reading gripper width
            is_image_encoded: If True, store images as JPEG bytes (saves space)
            jpeg_quality: JPEG quality when is_image_encoded=True (0-100)
        """
        self.robot = robot
        self.cameras = cameras
        self.gripper = gripper
        self.is_image_encoded = is_image_encoded
        self.jpeg_quality = jpeg_quality
        self.init_time = time.time()
        
        # Video recording buffer (always raw for video saving)
        self.record_images: List[np.ndarray] = []
        
        # Initialize data storage
        self.clear()
    
    def clear(self):
        """Clear all collected data."""
        self.data = {
            "task_info": {"instruction": []},
            "action": {
                "delta_position": [],
                "delta_euler": [],
                "abs_position": [],
                "abs_euler": [],
                "abs_joints": [],
                "gripper_control": [],
            },
            "observation": {
                "is_image_encoded": self.is_image_encoded,
                "rgb": [],
                "depth": [],
                "timestamp": [],
            },
            "state": {
                "abs_position": [],
                "abs_euler": [],  # quaternion [w,x,y,z]
                "abs_joints": [],
                "gripper_width": [],
            },
        }
        self.record_images = []
    
    # ==================== Recording Methods ====================
    
    def _encode_images(self, images: np.ndarray) -> List[np.ndarray]:
        """Encode images as JPEG if is_image_encoded is True.
        
        Args:
            images: [num_cams, H, W, 3] or [H, W, 3]
            
        Returns:
            List of encoded images (each is 1D bytes array) or original images
        """
        if not self.is_image_encoded:
            return images
        
        if images.ndim == 3:
            # Single image
            return [encode_image_jpeg(images, self.jpeg_quality)]
        else:
            # Multiple cameras
            return [encode_image_jpeg(images[i], self.jpeg_quality) 
                    for i in range(images.shape[0])]
    
    def record_observation(self, instruction: str = ""):
        """Record current observation (images, robot state).
        
        Args:
            instruction: Task instruction string
        """
        timestamp = time.time() - self.init_time
        
        # Get camera images
        if self.cameras is not None:
            images = self.cameras.get_rgb()  # [num_cams, H, W, 3]
            depth = self.cameras.get_depth()  # [num_cams, H, W]
            
            # Store raw concatenated image for video recording
            if len(images) >= 2:
                self.record_images.append(np.concatenate(images, axis=1))
            elif len(images) > 0:
                self.record_images.append(images[0])
            
            # Encode images if needed
            images_to_store = self._encode_images(images)
        else:
            images_to_store = np.zeros((1, 480, 640, 3), dtype=np.uint8)
            depth = np.zeros((1, 480, 640), dtype=np.float32)
        
        # Get robot state
        ee_affine = self.robot.current_pose.end_effector_pose
        joints = np.array(self.robot.current_joint_state.position)
        gripper_width = self.gripper.width if self.gripper else 0.08
        
        # Store data
        self.data["observation"]["rgb"].append(images_to_store)
        self.data["observation"]["depth"].append(depth)
        self.data["observation"]["timestamp"].append(timestamp)
        self.data["state"]["abs_position"].append(np.array(ee_affine.translation))
        self.data["state"]["abs_euler"].append(np.array(mat2euler(ee_affine.matrix[:3, :3], 'sxyz')))
        self.data["state"]["abs_joints"].append(joints)
        self.data["state"]["gripper_width"].append(gripper_width)
        self.data["task_info"]["instruction"].append(instruction)
    
    def record_action(
        self,
        delta_xyz: np.ndarray,
        delta_euler: np.ndarray,
        abs_position: np.ndarray,
        abs_rotation: np.ndarray,
        gripper_control: float,
        abs_joints: Optional[np.ndarray] = None,
    ):
        """Record action data.
        
        Args:
            delta_xyz: Position delta [3]
            delta_euler: Euler angle delta [3] (sxyz)
            abs_position: Absolute position [3]
            abs_rotation: Absolute rotation matrix [3,3]
            gripper_control: Gripper action (0=close, 1=open)
            abs_joints: Absolute joint positions [7] (for IK-based control)
        """
        self.data["action"]["delta_position"].append(delta_xyz.copy())
        self.data["action"]["delta_euler"].append(delta_euler.copy())
        self.data["action"]["abs_position"].append(abs_position.copy())
        self.data["action"]["abs_euler"].append(np.array(mat2euler(abs_rotation, 'sxyz')))
        self.data["action"]["gripper_control"].append(gripper_control)
        
        if abs_joints is not None:
            self.data["action"]["abs_joints"].append(abs_joints.copy())
    
    # ==================== Convenience Methods ====================
    
    def get_observation(self, instruction: str = "") -> Dict[str, Any]:
        """Get current observation for VLA inference.
        
        Also records the observation internally.
        
        Returns:
            Dictionary with ee_pose_T, joints, gripper_width, images, instruction
        """
        self.record_observation(instruction)
        
        ee_affine = self.robot.current_pose.end_effector_pose
        return {
            'ee_pose_T': np.array(ee_affine.matrix),
            'joints': np.array(self.robot.current_joint_state.position),
            'gripper_width': np.array([self.gripper.width if self.gripper else 0.08]),
            'instruction': instruction,
            'images': self.data["observation"]["rgb"][-1].astype(np.uint8),
        }
    
    @property
    def num_steps(self) -> int:
        """Number of recorded steps."""
        return len(self.data["observation"]["rgb"])
    
    # ==================== Saving Methods ====================
    
    def save(
        self,
        save_dir: str,
        episode_idx: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        fps: float = 30.0,
        compressed: bool = True,
        use_hdf5: bool = False,
        save_video: bool = True,
    ) -> str:
        """Save collected data to disk.
        
        Args:
            save_dir: Base directory for saving
            episode_idx: Episode index (auto-increment if None or -1)
            metadata: Optional metadata dictionary
            fps: Video frame rate
            compressed: Use compressed npz format (ignored if use_hdf5=True)
            use_hdf5: Save as HDF5 format (better for large datasets)
            save_video: Whether to save video files
            
        Returns:
            Path to saved episode directory
        """
        # Determine episode index
        if episode_idx is None or episode_idx < 0:
            episode_idx = get_next_episode_idx(save_dir)
        
        episode_dir = os.path.join(save_dir, f"episode_{episode_idx}")
        ensure_dir(episode_dir)
        
        # Convert data to numpy
        data_np = to_numpy(self.data)
        
        # Save data
        if use_hdf5:
            self._save_hdf5(data_np, episode_dir)
        else:
            self._save_numpy(data_np, episode_dir, compressed)
        
        # Save videos
        if save_video:
            self._save_videos(data_np, episode_dir, fps)
        
        # Save metadata
        if metadata:
            metadata["is_image_encoded"] = self.is_image_encoded
            metadata["jpeg_quality"] = self.jpeg_quality if self.is_image_encoded else None
            meta_path = os.path.join(episode_dir, "metadata.json")
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"Saved metadata to {meta_path}")
        
        return episode_dir
    
    def _save_numpy(self, data_np: Dict, episode_dir: str, compressed: bool):
        """Save data as numpy format."""
        data_path = os.path.join(episode_dir, "data")
        if compressed:
            np.savez_compressed(data_path, **{k: v for k, v in self._flatten_dict(data_np).items()})
            print(f"Saved data to {data_path}.npz")
        else:
            np.save(data_path + ".npy", data_np, allow_pickle=True)
            print(f"Saved data to {data_path}.npy")
    
    def _save_hdf5(self, data_np: Dict, episode_dir: str):
        """Save data as HDF5 format."""
        h5_path = os.path.join(episode_dir, "data.h5")
        
        with h5py.File(h5_path, 'w') as f:
            self._save_dict_to_hdf5(f, data_np)
        
        print(f"Saved data to {h5_path}")
    
    def _save_dict_to_hdf5(self, h5_group, data_dict: Dict, path: str = ""):
        """Recursively save dictionary to HDF5 group."""
        for key, value in data_dict.items():
            current_path = f"{path}/{key}" if path else key
            
            if isinstance(value, dict):
                group = h5_group.create_group(key)
                self._save_dict_to_hdf5(group, value, current_path)
            elif isinstance(value, list):
                # Handle list of varying-length arrays (e.g., encoded images)
                if len(value) > 0 and isinstance(value[0], np.ndarray):
                    if self.is_image_encoded and "rgb" in current_path:
                        # Variable-length encoded images: use special dtype
                        dt = h5py.special_dtype(vlen=np.uint8)
                        dset = h5_group.create_dataset(key, (len(value),), dtype=dt)
                        for i, img in enumerate(value):
                            if isinstance(img, list):
                                # Multiple cameras: flatten to single bytes
                                dset[i] = np.concatenate(img)
                            else:
                                dset[i] = img
                    else:
                        # Fixed-size arrays: stack and save
                        try:
                            arr = np.stack(value)
                            h5_group.create_dataset(key, data=arr, compression="gzip")
                        except ValueError:
                            # Variable shapes, save as object
                            dt = h5py.special_dtype(vlen=np.float32)
                            dset = h5_group.create_dataset(key, (len(value),), dtype=dt)
                            for i, v in enumerate(value):
                                dset[i] = v.flatten()
                else:
                    # List of scalars or strings
                    arr = np.array(value)
                    if arr.dtype.kind == 'U':
                        # String array
                        dt = h5py.special_dtype(vlen=str)
                        dset = h5_group.create_dataset(key, (len(value),), dtype=dt)
                        for i, s in enumerate(value):
                            dset[i] = s
                    else:
                        h5_group.create_dataset(key, data=arr, compression="gzip")
            else:
                # Scalar or single array
                h5_group.create_dataset(key, data=value)
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '/') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _save_videos(self, data_np: Dict, episode_dir: str, fps: float):
        """Save video files."""
        # Save concatenated video (raw images)
        if self.record_images:
            video_path = os.path.join(episode_dir, "video_concat.mp4")
            imageio.mimsave(video_path, self.record_images, fps=fps)
            print(f"Saved video to {video_path}")
        
        # # Save per-camera videos (only if not encoded)
        # if not self.is_image_encoded:
        #     rgb_data = data_np["observation"]["rgb"]
        #     if len(rgb_data) > 0:
        #         self._save_camera_videos(rgb_data, episode_dir, fps)
    
    def _save_camera_videos(self, rgb_array, save_dir: str, fps: float):
        """Save per-camera videos.
        
        Args:
            rgb_array: Shape [frame_num, cam_num, H, W, 3] or list of such
        """
        if isinstance(rgb_array, list):
            rgb_array = np.stack(rgb_array, axis=0)
        
        if rgb_array.ndim != 5 or rgb_array.shape[-1] != 3:
            return  # Skip if not proper multi-camera format
        
        _, cam_num, _, _, _ = rgb_array.shape
        
        for cam_idx in range(cam_num):
            video_path = os.path.join(save_dir, f"cam_{cam_idx}.mp4")
            frames = [rgb_array[i, cam_idx].astype(np.uint8) 
                     for i in range(rgb_array.shape[0])]
            imageio.mimsave(video_path, frames, fps=fps, codec='libx264')
            print(f"Saved camera {cam_idx} video to {video_path}")
