"""
Franka Robot Constants for franky_control.

This module contains all common constants, default parameters, and configurations
for Franka Panda/FR3 robot control. Import these constants throughout the project
to ensure consistency and easier parameter management.

Usage:
    from franky_control.robot.constants import FC
    
    robot = Robot(FC.ROBOT_IP)
    robot.move(JointWaypointMotion([JointWaypoint(FC.HOME_JOINTS)]))
"""

import math
import numpy as np
from typing import List, Tuple


class FrankaConstants:
    """Franka Robot Constants.
    
    All constants for Franka Panda/FR3 robot configuration.
    Use FC = FrankaConstants() for convenient access.
    """
    
    # ==================== Network Configuration ====================
    ROBOT_IP: str = "172.16.0.2"
    """Default robot IP address for FCI connection."""
    
    # ==================== Joint Configuration ====================
    NUM_JOINTS: int = 7
    """Number of robot arm joints."""
    
    HOME_JOINTS: List[float] = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    """Default home joint positions [rad]. 
    This is a common safe home position with elbow up."""
    
    RESET_JOINTS: List[float] = [0.0, 0.259, 0.0, -2.289, 0.0, 2.515, math.pi / 4]
    """Alternative reset joint positions [rad]."""
    
    READY_JOINTS: List[float] = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    """Ready position for manipulation tasks [rad]."""
    
    # Joint limits [rad]
    JOINT_LIMITS_LOWER: List[float] = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
    JOINT_LIMITS_UPPER: List[float] = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
    
    # ==================== Gripper Configuration ====================
    GRIPPER_MAX_WIDTH: float = 0.08
    """Maximum gripper opening width [m]."""
    
    GRIPPER_MIN_WIDTH: float = 0.0
    """Minimum gripper opening width [m]."""
    
    GRIPPER_DEFAULT_SPEED: float = 0.1
    """Default gripper movement speed [m/s]."""
    
    GRIPPER_MAX_FORCE: float = 70.0
    """Maximum gripper grasping force [N]."""
    
    GRIPPER_MIN_FORCE: float = 30.0
    """Minimum gripper grasping force [N]."""
    
    GRIPPER_DEFAULT_FORCE: float = 40.0
    """Default gripper grasping force [N]."""
    
    # ==================== Impedance Control ====================
    # Cartesian impedance (translational: N/m, rotational: Nm/rad)
    DEFAULT_CARTESIAN_IMPEDANCES: List[float] = [600.0, 600.0, 600.0, 60.0, 60.0, 60.0]
    """Default Cartesian impedance [x, y, z, rx, ry, rz]."""
    
    # Joint impedance [Nm/rad]
    DEFAULT_JOINT_IMPEDANCES: List[float] = [400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0]
    """Default joint impedance values."""

    # ==================== Collision Thresholds ====================
    # Lower thresholds (contact detection) [Nm] for joints
    COLLISION_TORQUE_LOWER: List[float] = [25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.0]
    """Lower torque thresholds for contact detection [Nm]."""
    
    # Upper thresholds (collision detection) [Nm] for joints
    COLLISION_TORQUE_UPPER: List[float] = [35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0]
    """Upper torque thresholds for collision detection [Nm]."""
    
    # Lower force thresholds [N, N, N, Nm, Nm, Nm]
    COLLISION_FORCE_LOWER: List[float] = [30.0, 30.0, 30.0, 25.0, 25.0, 25.0]
    """Lower force thresholds for contact detection [N, N, N, Nm, Nm, Nm]."""
    
    # Upper force thresholds [N, N, N, Nm, Nm, Nm]  
    COLLISION_FORCE_UPPER: List[float] = [40.0, 40.0, 40.0, 35.0, 35.0, 35.0]
    """Upper force thresholds for collision detection [N, N, N, Nm, Nm, Nm]."""
    
    # ==================== Dynamics Factors ====================
    DEFAULT_GLOBAL_DYNAMICS_FACTOR: float = 0.5
    """Default golbal scaling for vel, acc, jerk (0-1)"""
    
    DEFAULT_VELOCITY_FACTOR: float = 0.18
    """Default velocity scaling factor (0-1)."""
    
    DEFAULT_ACCELERATION_FACTOR: float = 0.13
    """Default acceleration scaling factor (0-1)."""
    
    DEFAULT_JERK_FACTOR: float = 0.11
    """Default jerk scaling factor (0-1)."""
    
    # ==================== Control Parameters ====================
    DEFAULT_CONTROL_FREQUENCY: float = 10.0
    """Default control loop frequency [Hz]."""
    
    VLA_CONTROL_FREQUENCY: float = 10.0
    """Control frequency for VLA deployment [Hz]."""
    
    DEFAULT_POS_SCALE: float = 0.015
    """Default position action scaling for teleoperation."""
    
    DEFAULT_ROT_SCALE: float = 0.025
    """Default rotation action scaling for teleoperation."""
    
    # ==================== SpaceMouse Configuration ====================
    SPACEMOUSE_VENDOR_ID: int = 0x256f
    """SpaceMouse USB vendor ID."""
    
    SPACEMOUSE_PRODUCT_ID: int = 0xc635
    """SpaceMouse USB product ID (SpaceMouse Compact)."""
    
    SPACEMOUSE_DEADZONE: float = 0.35
    """SpaceMouse deadzone threshold."""
    
    # ==================== Camera Configuration ====================
    CAMERA_WIDTH: int = 640
    """Default camera image width."""
    
    CAMERA_HEIGHT: int = 480
    """Default camera image height."""
    
    CAMERA_FPS: int = 30
    """Default camera frame rate."""
    
    # ==================== Data Collection ====================
    DEFAULT_DATASET_DIR: str = "datasets"
    """Default directory for saving datasets."""
    
    DEFAULT_LOG_DIR: str = "logs"
    """Default directory for saving logs."""
    
    MIN_EPISODE_STEPS: int = 200
    """Minimum steps for a valid episode."""
    
    MAX_EPISODE_STEPS: int = 1000
    """Maximum steps per episode."""
    
    # ==================== URDF Paths ====================
    @property
    def URDF_GRIPPER_OPEN(self) -> float:
        """Gripper joint value for open state in URDF [m]."""
        return 0.04
    
    @property  
    def URDF_GRIPPER_CLOSE(self) -> float:
        """Gripper joint value for closed state in URDF [m]."""
        return 0.0


# Create a singleton instance for convenient access
FC = FrankaConstants()
"""Singleton instance of FrankaConstants for convenient access.

Usage:
    from franky_control.robot.constants import FC
    
    print(FC.ROBOT_IP)
    print(FC.HOME_JOINTS)
"""


# ==================== Helper Functions ====================

def get_home_joints() -> np.ndarray:
    """Get home joint positions as numpy array."""
    return np.array(FC.HOME_JOINTS)


def get_reset_joints() -> np.ndarray:
    """Get reset joint positions as numpy array."""
    return np.array(FC.RESET_JOINTS)


def get_default_cartesian_impedance() -> List[float]:
    """Get default Cartesian impedance as list."""
    return list(FC.DEFAULT_CARTESIAN_IMPEDANCES)


def get_collision_behavior() -> Tuple[List[float], List[float], List[float], List[float]]:
    """Get default collision behavior parameters.
    
    Returns:
        Tuple of (torque_lower, torque_upper, force_lower, force_upper)
    """
    return (
        list(FC.COLLISION_TORQUE_LOWER),
        list(FC.COLLISION_TORQUE_UPPER),
        list(FC.COLLISION_FORCE_LOWER),
        list(FC.COLLISION_FORCE_UPPER),
    )


# ==================== Aliases for Backward Compatibility ====================

# Common constants as module-level variables
ROBOT_IP = FC.ROBOT_IP
HOME_JOINTS = FC.HOME_JOINTS
RESET_JOINTS = FC.RESET_JOINTS
GRIPPER_MAX_WIDTH = FC.GRIPPER_MAX_WIDTH
GRIPPER_MAX_FORCE = FC.GRIPPER_MAX_FORCE
DEFAULT_CONTROL_FREQUENCY = FC.DEFAULT_CONTROL_FREQUENCY


__all__ = [
    'FrankaConstants',
    'FC',
    'get_home_joints',
    'get_reset_joints',
    'get_default_cartesian_impedance',
    'get_collision_behavior',
    # Aliases
    'ROBOT_IP',
    'HOME_JOINTS',
    'RESET_JOINTS',
    'GRIPPER_MAX_WIDTH',
    'GRIPPER_MAX_FORCE',
    'DEFAULT_CONTROL_FREQUENCY',
]
