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
        robot: Robot = None,
        cameras=None,
        gripper=None,
        is_image_encoded: bool = True,
        jpeg_quality: int = 95,
    ):
        """Initialize data collector.
        
        Args:
            robot: franky.Robot instance (optional for external state mode)
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
        
        # Video recording buffer (always raw for video saving)
        self.record_images: List[np.ndarray] = []
        
        # Initialize data storage
        self.clear()
    
    def clear(self):
        """Clear all collected data."""
        self.data = {
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
                    "abs_joints": [],
                    "gripper_control": [],
                    "gripper_width": [],
                },
                "joint": {
                    "position": [],
                    "gripper_control": [],
                    "gripper_width": [],
                },
                "timestamp": [],
            },
            "observation": {
                "is_image_encode": self.is_image_encoded,
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
                "timestamp": [],
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
    
    def record_observation_with_images(
        self,
        instruction: str = "",
        images: np.ndarray = None,
        depth: np.ndarray = None,
    ):
        """Record observation with pre-fetched images (no camera access).
        
        This is faster than record_observation() when images are pre-fetched
        outside the critical control path.
        
        Args:
            instruction: Task instruction string
            images: Pre-fetched RGB images [num_cams, H, W, 3]
            depth: Pre-fetched depth images [num_cams, H, W]
        """
        timestamp = time.time()
        
        # Use provided images or fallback to zeros
        if images is not None:
            # Store raw concatenated image for video recording
            if len(images) >= 2:
                self.record_images.append(np.concatenate(images, axis=1))
            elif len(images) > 0:
                self.record_images.append(images[0])
            
            # Encode images if needed
            images_to_store = self._encode_images(images)
        else:
            images_to_store = np.zeros((1, 480, 640, 3), dtype=np.uint8)
        
        if depth is None:
            depth = np.zeros((1, 480, 640), dtype=np.float32)
        
        # Get robot state
        ee_affine = self.robot.current_pose.end_effector_pose
        joints = np.array(self.robot.current_joint_state.position)
        gripper_width = self.gripper.width if self.gripper else 0.08
        
        # Store observation data
        self.data["observation"]["rgb"].append(images_to_store)
        self.data["observation"]["rgb_timestamp"].append(timestamp)
        self.data["observation"]["depth"].append(depth)
        self.data["observation"]["depth_timestamp"].append(timestamp)
        
        # Store state data
        self.data["state"]["end_effector"]["position"].append(np.array(ee_affine.translation))
        self.data["state"]["end_effector"]["orientation"].append(np.array(ee_affine.quaternion))
        self.data["state"]["end_effector"]["gripper_width"].append(gripper_width)
        self.data["state"]["joint"]["position"].append(joints)
        self.data["state"]["timestamp"].append(timestamp)
        
        # Store task info
        self.data["task_info"]["instruction"].append(instruction)
    
    def record_observation(self, instruction: str = ""):
        """Record current observation (images, robot state).
        
        Args:
            instruction: Task instruction string
        """
        timestamp = time.time()
        
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
        
        # Store observation data
        self.data["observation"]["rgb"].append(images_to_store)
        self.data["observation"]["rgb_timestamp"].append(timestamp)
        self.data["observation"]["depth"].append(depth)
        self.data["observation"]["depth_timestamp"].append(timestamp)
        
        # Store state data
        self.data["state"]["end_effector"]["position"].append(np.array(ee_affine.translation))
        self.data["state"]["end_effector"]["orientation"].append(np.array(ee_affine.quaternion))  # [w, x, y, z]
        self.data["state"]["end_effector"]["gripper_width"].append(gripper_width)
        self.data["state"]["joint"]["position"].append(joints)
        self.data["state"]["timestamp"].append(timestamp)
        
        # Store task info
        self.data["task_info"]["instruction"].append(instruction)
    
    def record_observation_external(
        self,
        instruction: str = "",
        ee_position: np.ndarray = None,
        ee_orientation: np.ndarray = None,
        joints: np.ndarray = None,
        gripper_width: float = 0.08,
        timestamp: float = None,
    ):
        """Record observation with externally provided robot state.
        
        Use this method when robot state comes from a separate process
        (e.g., multiprocessing architecture).
        
        Args:
            instruction: Task instruction string
            ee_position: End-effector position [3]
            ee_orientation: End-effector orientation as quaternion [4] (w, x, y, z)
            joints: Joint positions [7]
            gripper_width: Gripper width in meters
            timestamp: Optional timestamp (uses time.time() if None)
        """
        if timestamp is None:
            timestamp = time.time()
        
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
            images_to_store = [np.zeros((100,), dtype=np.uint8)]  # Placeholder
            depth = np.zeros((1, 480, 640), dtype=np.float32)
        
        # Store observation data
        self.data["observation"]["rgb"].append(images_to_store)
        self.data["observation"]["rgb_timestamp"].append(timestamp)
        self.data["observation"]["depth"].append(depth)
        self.data["observation"]["depth_timestamp"].append(timestamp)
        
        # Store state data (from external source)
        self.data["state"]["end_effector"]["position"].append(
            ee_position.copy() if ee_position is not None else np.zeros(3)
        )
        self.data["state"]["end_effector"]["orientation"].append(
            ee_orientation.copy() if ee_orientation is not None else np.array([1.0, 0.0, 0.0, 0.0])
        )
        self.data["state"]["end_effector"]["gripper_width"].append(gripper_width)
        self.data["state"]["joint"]["position"].append(
            joints.copy() if joints is not None else np.zeros(7)
        )
        self.data["state"]["timestamp"].append(timestamp)
        
        # Store task info
        self.data["task_info"]["instruction"].append(instruction)
    
    def record_action(
        self,
        delta_xyz: np.ndarray,
        delta_euler: np.ndarray,
        delta_orientation: np.ndarray,
        abs_position: np.ndarray,
        abs_rotation: np.ndarray,
        gripper_control: float,
        abs_joints: Optional[np.ndarray] = None,
        gripper_width: Optional[float] = None,
        timestamp: Optional[float] = None,
    ):
        """Record action data.
        
        Args:
            delta_xyz: Position delta [3]
            delta_euler: Euler angle delta [3] (sxyz)
            delta_orientation: Orientation delta as quaternion [4] (w, x, y, z)
            abs_position: Absolute position [3]
            abs_rotation: Absolute rotation matrix [3,3]
            gripper_control: Gripper action (0=close, 1=open)
            abs_joints: Absolute joint positions [7] (for IK-based control)
            gripper_width: Gripper width in meters (physical value)
            timestamp: Optional timestamp for the action (uses time.time() if None)
        """
        action_timestamp = time.time() if timestamp is None else timestamp
        
        # Use provided gripper_width or default to 0 for closed, 0.08 for open
        if gripper_width is None:
            gripper_width = 0.0 if gripper_control < 0.5 else 0.08
        
        action = self.data["action"]["end_effector"]
        action["delta_position"].append(delta_xyz.copy())
        action["delta_euler"].append(delta_euler.copy())
        action["delta_orientation"].append(delta_orientation.copy())
        action["abs_position"].append(abs_position.copy())
        action["abs_euler"].append(np.array(mat2euler(abs_rotation, 'sxyz')))
        action["gripper_control"].append(gripper_control)
        action["gripper_width"].append(gripper_width)
        
        if abs_joints is not None:
            action["abs_joints"].append(abs_joints.copy())
        
        # Also store joint action
        self.data["action"]["joint"]["position"].append(
            abs_joints.copy() if abs_joints is not None else np.zeros(7)
        )
        self.data["action"]["joint"]["gripper_control"].append(gripper_control)
        self.data["action"]["joint"]["gripper_width"].append(gripper_width)
        
        self.data["action"]["timestamp"].append(action_timestamp)
    
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
            'images': self.data["observation"]["rgb"][-1],
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
        
        # Flatten the dictionary
        flat_data = self._flatten_dict(data_np)
        
        # Process each field to ensure it can be saved
        processed_data = {}
        for k, v in flat_data.items():
            if isinstance(v, list):
                # Check if it's a list of lists (e.g., encoded images)
                if len(v) > 0 and isinstance(v[0], list):
                    # Save as object array with pickle
                    processed_data[k] = np.array(v, dtype=object)
                elif len(v) > 0 and isinstance(v[0], np.ndarray):
                    # Try to stack, fall back to object array
                    try:
                        processed_data[k] = np.stack(v)
                    except ValueError:
                        processed_data[k] = np.array(v, dtype=object)
                else:
                    try:
                        processed_data[k] = np.array(v)
                    except ValueError:
                        processed_data[k] = np.array(v, dtype=object)
            else:
                processed_data[k] = v
        
        if compressed:
            # Note: savez_compressed doesn't support object arrays well,
            # so we use savez for mixed data
            try:
                np.savez_compressed(data_path, **processed_data)
                print(f"Saved data to {data_path}.npz")
            except Exception as e:
                print(f"[WARNING] savez_compressed failed: {e}, using pickle save")
                np.save(data_path + ".npy", data_np, allow_pickle=True)
                print(f"Saved data to {data_path}.npy (with pickle)")
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
                if len(value) == 0:
                    # Empty list, create empty dataset
                    h5_group.create_dataset(key, data=np.array([]))
                    continue
                
                first_elem = value[0]
                
                # Handle list of lists (e.g., rgb images from multiple cameras)
                if isinstance(first_elem, list):
                    # Check if it's encoded images (list of list of np.ndarray)
                    if len(first_elem) > 0 and isinstance(first_elem[0], np.ndarray):
                        if self.is_image_encoded and "rgb" in current_path:
                            # Encoded images: concatenate all cameras per timestep
                            dt = h5py.special_dtype(vlen=np.uint8)
                            dset = h5_group.create_dataset(key, (len(value),), dtype=dt)
                            for i, img_list in enumerate(value):
                                if len(img_list) > 1:
                                    dset[i] = np.concatenate(img_list)
                                elif len(img_list) == 1:
                                    dset[i] = img_list[0]
                                else:
                                    dset[i] = np.array([], dtype=np.uint8)
                        else:
                            # Non-encoded multi-camera images: stack
                            try:
                                arr = np.array(value)  # [T, num_cams, H, W, C]
                                h5_group.create_dataset(key, data=arr, compression="gzip")
                            except ValueError:
                                # Variable shapes, save per-timestep
                                dt = h5py.special_dtype(vlen=np.uint8)
                                dset = h5_group.create_dataset(key, (len(value),), dtype=dt)
                                for i, img_list in enumerate(value):
                                    dset[i] = np.concatenate([img.flatten() for img in img_list])
                    else:
                        # List of empty lists or other types
                        h5_group.create_dataset(key, data=np.array([]))
                        
                elif isinstance(first_elem, np.ndarray):
                    if self.is_image_encoded and "rgb" in current_path:
                        # Variable-length encoded images: use special dtype
                        dt = h5py.special_dtype(vlen=np.uint8)
                        dset = h5_group.create_dataset(key, (len(value),), dtype=dt)
                        for i, img in enumerate(value):
                            dset[i] = img
                    else:
                        # Fixed-size arrays: stack and save
                        try:
                            arr = np.stack(value)
                            # Check if stacked array is string type
                            if arr.dtype.kind in ('U', 'O', 'S'):
                                dt = h5py.special_dtype(vlen=str)
                                dset = h5_group.create_dataset(key, arr.shape, dtype=dt)
                                for idx in np.ndindex(arr.shape):
                                    dset[idx] = str(arr[idx]) if arr[idx] is not None else ""
                            else:
                                h5_group.create_dataset(key, data=arr, compression="gzip")
                        except ValueError:
                            # Variable shapes, save as object
                            dt = h5py.special_dtype(vlen=np.float32)
                            dset = h5_group.create_dataset(key, (len(value),), dtype=dt)
                            for i, v in enumerate(value):
                                dset[i] = v.flatten()
                else:
                    # List of scalars or strings
                    try:
                        arr = np.array(value)
                        if arr.dtype.kind in ('U', 'O', 'S'):
                            # String array (Unicode, Object, or byte string)
                            dt = h5py.special_dtype(vlen=str)
                            dset = h5_group.create_dataset(key, (len(value),), dtype=dt)
                            for i, s in enumerate(value):
                                dset[i] = str(s) if s is not None else ""
                        else:
                            h5_group.create_dataset(key, data=arr, compression="gzip")
                    except ValueError as e:
                        print(f"[WARNING] Cannot save {current_path}: {e}")
                        h5_group.create_dataset(key, data=np.array([]))
            elif isinstance(value, str):
                # Single string value
                dt = h5py.special_dtype(vlen=str)
                dset = h5_group.create_dataset(key, (1,), dtype=dt)
                dset[0] = value
            else:
                # Scalar or single array
                if isinstance(value, np.ndarray) and value.dtype.kind in ('U', 'O', 'S'):
                    # String array
                    dt = h5py.special_dtype(vlen=str)
                    dset = h5_group.create_dataset(key, value.shape, dtype=dt)
                    for idx in np.ndindex(value.shape):
                        dset[idx] = str(value[idx]) if value[idx] is not None else ""
                else:
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
