"""IK Controller for Franka Panda robot.

This module provides a unified IK control interface that can be reused
across data collection and deployment scripts.
"""

import torch
import numpy as np
from typing import Optional, Tuple
from transforms3d.quaternions import mat2quat

from franky_control.kinematics import PANDA_URDF_PATH, ensure_assets_downloaded
from franky_control.kinematics.panda_ik_solver import create_sim_aligned_ik_solver, SimPose


class IKController:
    """Unified IK controller with joint filtering and statistics tracking.
    
    Features:
    - IK computation with warm start
    - Optional low-pass joint filtering for smooth motion
    - IK success/failure statistics
    - FK verification for debugging
    """
    
    def __init__(
        self,
        device: str = "cpu",
        use_joint_filter: bool = False,
        filter_alpha: float = 0.4,
    ):
        """Initialize IK controller.
        
        Args:
            device: "cpu" or "cuda" for IK solver
            use_joint_filter: Enable low-pass filtering for smooth joint motion
            filter_alpha: Filter smoothing factor (0-1, lower=smoother)
        """
        ensure_assets_downloaded("panda")
        
        self.device = device
        self.use_joint_filter = use_joint_filter
        self.filter_alpha = filter_alpha
        
        # IK solver
        self.ik_solver = create_sim_aligned_ik_solver(
            urdf_path=PANDA_URDF_PATH,
            device=device,
        )
        
        # Statistics
        self.ik_success_count = 0
        self.ik_failure_count = 0
        
        # Joint filtering state
        self._filtered_joints: Optional[np.ndarray] = None
        self._current_joints: Optional[np.ndarray] = None
        
        print(f"[IKController] Initialized on device: {device}")
        print(f"[IKController] Solver mode: {self.ik_solver.alignment_mode}")
        if use_joint_filter:
            print(f"[IKController] Joint filter enabled (alpha={filter_alpha})")
    
    def reset(self, initial_joints: np.ndarray):
        """Reset controller state with current joint positions.
        
        Args:
            initial_joints: Current robot joint positions [7]
        """
        self._current_joints = initial_joints.copy()
        if self.use_joint_filter:
            self._filtered_joints = initial_joints.copy()
    
    def compute_ik(
        self,
        position: np.ndarray,
        rotation_matrix: np.ndarray,
        apply_filter: bool = True,
    ) -> Optional[np.ndarray]:
        """Compute IK for target pose.
        
        Args:
            position: Target position [3]
            rotation_matrix: Target rotation matrix [3,3]
            apply_filter: Whether to apply joint filtering
            
        Returns:
            Joint positions [7], or None if IK fails
        """
        # Convert rotation matrix to quaternion (wxyz format)
        quat_wxyz = mat2quat(rotation_matrix)
        
        target_pose = SimPose.from_pq(
            position=position,
            quaternion=quat_wxyz,
            device=self.device
        )
        
        # Solve IK with warm start
        ik_solution = self.ik_solver.compute_ik(
            target_pose,
            initial_qpos=self._current_joints
        )
        
        if ik_solution is None:
            self.ik_failure_count += 1
            return None
        
        self.ik_success_count += 1
        target_joints = ik_solution[0].cpu().numpy()
        
        # Apply low-pass filter
        if self.use_joint_filter and apply_filter and self._filtered_joints is not None:
            self._filtered_joints = (
                self.filter_alpha * target_joints +
                (1 - self.filter_alpha) * self._filtered_joints
            )
            filtered_target = self._filtered_joints.copy()
        else:
            filtered_target = target_joints
        
        # Update state for next warm start (use unfiltered for IK)
        self._current_joints = target_joints
        
        return filtered_target
    
    def verify_solution(
        self,
        joint_solution: np.ndarray,
        position: np.ndarray,
        rotation_matrix: np.ndarray,
    ) -> Tuple[float, float]:
        """Verify IK solution with FK.
        
        Args:
            joint_solution: IK solution [7]
            position: Target position [3]
            rotation_matrix: Target rotation matrix [3,3]
            
        Returns:
            (position_error, orientation_error) in meters and radians
        """
        quat_wxyz = mat2quat(rotation_matrix)
        target_pose = SimPose.from_pq(
            position=position,
            quaternion=quat_wxyz,
            device=self.device
        )
        
        joint_tensor = torch.from_numpy(joint_solution).unsqueeze(0).to(self.device)
        return self.ik_solver.verify_ik_solution(joint_tensor, target_pose)
    
    @property
    def current_joints(self) -> Optional[np.ndarray]:
        """Get current (unfiltered) joint positions."""
        return self._current_joints
    
    @property
    def stats(self) -> dict:
        """Get IK statistics."""
        total = self.ik_success_count + self.ik_failure_count
        success_rate = 100.0 * self.ik_success_count / total if total > 0 else 0.0
        return {
            "success": self.ik_success_count,
            "failure": self.ik_failure_count,
            "total": total,
            "success_rate": success_rate,
        }
    
    def print_stats(self):
        """Print IK statistics."""
        s = self.stats
        if s["total"] > 0:
            print(f"\n[IK Statistics]")
            print(f"  Success: {s['success']}")
            print(f"  Failure: {s['failure']}")
            print(f"  Success Rate: {s['success_rate']:.2f}%")
    
    @property
    def alignment_mode(self) -> str:
        """Get IK solver alignment mode."""
        return self.ik_solver.alignment_mode
    
    @property
    def max_iterations(self) -> int:
        """Get IK solver max iterations."""
        return self.ik_solver.max_iterations
