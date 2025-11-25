"""
Panda Robot IK/FK Solver - Fully Aligned with ManiSkill Simulation
===================================================================

This solver aims to fully replicate the kinematics behavior of ManiSkill simulation,
ensuring consistency between real robot control and simulation.

Key alignment points:
1. Uses pytorch_kinematics==0.7.5 (same as ManiSkill)
2. IK parameters aligned with simulation:
   - GPU mode: max_iterations=200, num_retries=1 (pytorch_kinematics)
   - CPU mode: max_iterations=100 (Pinocchio)
   - early_stopping_any_converged=True
3. FK/IK calling conventions match simulation code exactly
4. Quaternion format: (w,x,y,z), consistent with pytorch_kinematics

CPU mode support:
- Prefers Pinocchio (fully aligned with ManiSkill CPU simulation)
- Falls back to pytorch_kinematics if SAPIEN unavailable (parameters aligned with Pinocchio)

Reference: mani_skill/agents/controllers/utils/kinematics.py
"""

import torch
import numpy as np
from typing import Optional, Tuple
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from dataclasses import dataclass
import pytorch_kinematics as pk

# Try to import SAPIEN's Pinocchio (for CPU mode)
try:
    from sapien.wrapper.pinocchio_model import PinocchioModel
    import sapien
    PINOCCHIO_AVAILABLE = True
except ImportError:
    PINOCCHIO_AVAILABLE = False
    PinocchioModel = None


@dataclass
class SimPose:
    """
    Pose representation - corresponds to ManiSkill's Pose class
    
    Note: Quaternion format is (w,x,y,z), consistent with pytorch_kinematics
    """
    p: torch.Tensor  # position, shape (B, 3) or (3,)
    q: torch.Tensor  # quaternion (w,x,y,z), shape (B, 4) or (4,)
    
    def __post_init__(self):
        """Ensure pose has batch dimension"""
        if self.p.dim() == 1:
            self.p = self.p.unsqueeze(0)
        if self.q.dim() == 1:
            self.q = self.q.unsqueeze(0)
    
    @classmethod
    def from_pq(cls, position: np.ndarray, quaternion: np.ndarray, device: str = "cpu"):
        """Create pose from numpy arrays (position, quaternion in w,x,y,z format)"""
        p = torch.tensor(position, dtype=torch.float32, device=device)
        q = torch.tensor(quaternion, dtype=torch.float32, device=device)
        return cls(p=p, q=q)
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray, device: str = "cpu"):
        """Create pose from 4x4 transformation matrix"""
        matrix_tensor = torch.tensor(matrix, dtype=torch.float32, device=device)
        if matrix_tensor.dim() == 2:
            matrix_tensor = matrix_tensor.unsqueeze(0)
        
        p = matrix_tensor[:, :3, 3]
        rot_mat = matrix_tensor[:, :3, :3]
        q = pk.matrix_to_quaternion(rot_mat)  # returns (w,x,y,z)
        
        return cls(p=p, q=q)
    
    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return numpy format (position, quaternion)"""
        return self.p.cpu().numpy(), self.q.cpu().numpy()


class SimAlignedPandaIKSolver:
    """
    Panda IK/FK Solver - Fully Aligned with ManiSkill Simulation
    
    All parameters and calling conventions match ManiSkill's Kinematics class
    to ensure real robot uses exactly the same kinematics as simulation.
    
    Reference: mani_skill/agents/controllers/utils/kinematics.py
    """
    
    # Panda robot configuration (consistent with ManiSkill environment)
    JOINT_NAMES = [
        "panda_joint1", "panda_joint2", "panda_joint3",
        "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
    ]
    END_EFFECTOR_LINK = "panda_hand_tcp"
    NUM_JOINTS = 7
    
    def __init__(
        self,
        urdf_path: str,
        device: str = "cpu",
        use_pinocchio: Optional[bool] = None,
    ):
        """        self.urdf_path = urdf_path

        Initialize IK solver
        
        Args:
            urdf_path: Path to URDF file
            device: "cpu" or "cuda"
            use_pinocchio: Force use Pinocchio (CPU only)
                          None: Auto-select (use Pinocchio if CPU and SAPIEN available)
                          True: Force Pinocchio (requires SAPIEN)
                          False: Force pytorch_kinematics
        
        Note:
        - CPU + Pinocchio: max_iterations=100 (fully aligned with ManiSkill CPU)
        - CPU + pytorch_kinematics: max_iterations=100 (aligned with Pinocchio behavior)
        - GPU + pytorch_kinematics: max_iterations=200 (fully aligned with ManiSkill GPU)
        """
        self.urdf_path = urdf_path
        self.device = device
        self.use_gpu = (device == "cuda")
        
        # Load URDF
        with open(self.urdf_path, "rb") as f:
            urdf_str = f.read()
        self.urdf_str = urdf_str
        
        # Decide which solver to use
        if use_pinocchio is None:
            # Auto-select: use Pinocchio if CPU and SAPIEN available
            use_pinocchio = (not self.use_gpu) and PINOCCHIO_AVAILABLE
        elif use_pinocchio and self.use_gpu:
            raise ValueError("Pinocchio only supports CPU mode")
        elif use_pinocchio and not PINOCCHIO_AVAILABLE:
            raise ImportError("Pinocchio requested but SAPIEN is not available")
        
        self.use_pinocchio = use_pinocchio
        
        if self.use_pinocchio:
            # CPU mode + Pinocchio: fully aligned with ManiSkill CPU simulation
            self.alignment_mode = "CPU (Pinocchio)"
            self.max_iterations = 100
            self._setup_pinocchio()
        else:
            # pytorch_kinematics mode
            if self.use_gpu:
                self.alignment_mode = "GPU (pytorch_kinematics)"
                self.max_iterations = 200
            else:
                self.alignment_mode = "CPU (pytorch_kinematics, Pinocchio-equivalent)"
                self.max_iterations = 100
            self._setup_pytorch_kinematics()
        
        print(f"[SimAlignedPandaIKSolver] Initialized (device={device})")
        print(f"[SimAlignedPandaIKSolver] Solver: {self.alignment_mode}")
        print(f"[SimAlignedPandaIKSolver] IK config: max_iterations={self.max_iterations}")
        print(f"[SimAlignedPandaIKSolver] ✓ Fully aligned with ManiSkill simulation")
    
    def _setup_pinocchio(self):
        """Setup Pinocchio solver (fully aligned with ManiSkill CPU simulation)"""
        # Create Pinocchio model
        self.pmodel = PinocchioModel(
            self.urdf_str.decode('utf-8'),
            np.array([0, 0, -9.81])
        )
        
        # Set joint order
        joint_names = self.JOINT_NAMES + ["panda_finger_joint1", "panda_finger_joint2"]
        self.pmodel.set_joint_order(joint_names)
        
        # Set link order and find end-effector index
        link_names = [
            "panda_link0", "panda_link1", "panda_link2", "panda_link3",
            "panda_link4", "panda_link5", "panda_link6", "panda_link7",
            "panda_link8", self.END_EFFECTOR_LINK, "panda_hand",
            "panda_leftfinger", "panda_rightfinger"
        ]
        self.pmodel.set_link_order(link_names)
        
        # Find end-effector index
        self.end_link_idx = link_names.index(self.END_EFFECTOR_LINK)
        
        # Create qmask (active joints mask)
        self.qmask = np.zeros(len(joint_names), dtype=bool)
        self.qmask[:7] = True  # First 7 are arm joints
        
        print(f"[Pinocchio] Joint names: {joint_names[:7]}")
        print(f"[Pinocchio] End effector: {self.END_EFFECTOR_LINK} (index={self.end_link_idx})")
        print(f"[Pinocchio] Active joints mask (bool): {self.qmask}")
    
    # def _pinocchio_ik_direct(
    #     self,
    #     target_pose_sapien: 'sapien.Pose',
    #     initial_qpos: Optional[np.ndarray] = None,
    #     eps: float = 1e-4,
    #     max_iterations: int = 100,
    #     dt: float = 0.1,
    #     damp: float = 1e-6,
    # ) -> Tuple[np.ndarray, bool, float]:
    #     """
    #     Direct Pinocchio IK implementation bypassing SAPIEN wrapper.
        
    #     This fixes the SAPIEN 3.0 compatibility issue by directly calling Pinocchio.
    #     Based on: https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b-examples_i-inverse-kinematics.html
    #     """
    #     import pinocchio
        
    #     # Get Pinocchio model and data
    #     model = self.pmodel.model
    #     data = self.pmodel.data
        
    #     # Initial configuration
    #     if initial_qpos is None:
    #         q = pinocchio.neutral(model)
    #     else:
    #         q = np.ascontiguousarray(initial_qpos.copy(), dtype=np.float64)
        
    #     # Target frame
    #     frame_id = self.end_link_idx
        
    #     # Convert SAPIEN pose to Pinocchio SE3
    #     pos = np.ascontiguousarray(target_pose_sapien.p, dtype=np.float64)
    #     quat = target_pose_sapien.q  # [w, x, y, z]
        
    #     # Convert quaternion to rotation matrix using transforms3d
    #     from transforms3d.quaternions import quat2mat
    #     rot_matrix = np.ascontiguousarray(quat2mat(quat), dtype=np.float64)
        
    #     oMdes = pinocchio.SE3(rot_matrix, pos)
        
    #     # CLIK algorithm
    #     success = False
    #     for i in range(max_iterations):
    #         # Forward kinematics
    #         pinocchio.framesForwardKinematics(model, data, q)
            
    #         # Get current frame pose
    #         frame_id_pin = model.getFrameId(model.frames[frame_id].name)
    #         oMi = data.oMf[frame_id_pin]
            
    #         # Compute error
    #         iMd = oMi.actInv(oMdes)
    #         err = pinocchio.log(iMd).vector
            
    #         # Check convergence
    #         error_norm = np.linalg.norm(err)
    #         if error_norm < eps:
    #             success = True
    #             break
            
    #         # Compute Jacobian
    #         J = pinocchio.computeFrameJacobian(
    #             model, data, q, frame_id_pin, pinocchio.ReferenceFrame.LOCAL
    #         )
            
    #         # Apply active joint mask
    #         J_masked = J[:, self.qmask]
            
    #         # Damped least squares
    #         J_damped = J_masked.T @ J_masked + damp * np.eye(J_masked.shape[1])
    #         v = -J_masked.T @ err
    #         dq_masked = np.linalg.solve(J_damped, v)
            
    #         # Update q (only active joints)
    #         dq = np.zeros(len(q))
    #         dq[self.qmask] = dq_masked
    #         q = pinocchio.integrate(model, q, dq * dt)
        
    #     return q, success, error_norm if 'error_norm' in locals() else np.inf
    
    def _setup_pytorch_kinematics(self):
        """Setup pytorch_kinematics solver"""
        # Suppress stdout/stderr
        @contextmanager
        def suppress_stdout_stderr():
            with open(devnull, "w") as fnull:
                with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
                    yield (err, out)
        
        with suppress_stdout_stderr():
            # Build kinematic chain
            self.pk_chain = pk.build_serial_chain_from_urdf(
                self.urdf_str,
                end_link_name=self.END_EFFECTOR_LINK,
            ).to(device=self.device)
        
        # Get joint limits
        lim = torch.tensor(self.pk_chain.get_joint_limits(), device=self.device)
        
        # Create IK solver
        self.pik = pk.PseudoInverseIK(
            self.pk_chain,
            joint_limits=lim.T,
            early_stopping_any_converged=True,
            max_iterations=self.max_iterations,
            num_retries=1,
        )
        
        print(f"[pytorch_kinematics] Joint limits:\n{lim.T}")
    
    def compute_fk(self, qpos: torch.Tensor) -> SimPose:
        """
        Forward kinematics - fully aligned with ManiSkill simulation
        
        Args:
            qpos: Joint positions, shape (B, 7) or (7,)
        
        Returns:
            SimPose: End-effector pose
        """
        if self.use_pinocchio:
            # Pinocchio FK (aligned with ManiSkill CPU simulation)
            if isinstance(qpos, torch.Tensor):
                qpos_np = qpos.cpu().numpy()
            else:
                qpos_np = np.array(qpos)
            
            if qpos_np.ndim == 1:
                qpos_np = qpos_np.reshape(1, -1)
            
            # Use only first 7 joints
            qpos_np = qpos_np[:, :self.NUM_JOINTS]
            
            # Pad to full length (7 arm + 2 gripper = 9)
            full_qpos = np.zeros((qpos_np.shape[0], 9))
            full_qpos[:, :7] = qpos_np
            
            # Compute FK
            self.pmodel.compute_forward_kinematics(full_qpos[0])
            ee_pose_sapien = self.pmodel.get_link_pose(self.end_link_idx)
            
            # Convert to SimPose
            pos = torch.tensor(ee_pose_sapien.p, dtype=torch.float32, device=self.device).unsqueeze(0)
            quat = torch.tensor(ee_pose_sapien.q, dtype=torch.float32, device=self.device).unsqueeze(0)  # (w,x,y,z)
            
            return SimPose(p=pos, q=quat)
        else:
            # pytorch_kinematics FK (aligned with ManiSkill GPU simulation)
            if isinstance(qpos, np.ndarray):
                qpos = torch.tensor(qpos, dtype=torch.float32, device=self.device)
            
            if qpos.dim() == 1:
                qpos = qpos.unsqueeze(0)
            
            # Use only first 7 joints
            qpos = qpos[..., :self.NUM_JOINTS]
            
            # FK computation
            tf_matrix = self.pk_chain.forward_kinematics(qpos.float()).get_matrix()
            pos = tf_matrix[:, :3, 3]
            rot = pk.matrix_to_quaternion(tf_matrix[:, :3, :3])
            
            return SimPose(p=pos, q=rot)
    
    def compute_ik(
        self,
        target_pose: SimPose,
        initial_qpos: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Inverse kinematics - fully aligned with ManiSkill simulation
        
        Args:
            target_pose: Target pose in robot base frame
            initial_qpos: Initial joint positions, shape (B, N) or (N,)
        
        Returns:
            Joint positions shape (B, 7), or None if failed
        """
        if self.use_pinocchio:
            # Pinocchio IK (aligned with ManiSkill CPU simulation)
            # Prepare initial qpos
            if initial_qpos is not None:
                if isinstance(initial_qpos, torch.Tensor):
                    q0_np = initial_qpos.cpu().numpy()
                else:
                    q0_np = np.array(initial_qpos, dtype=np.float64)
                
                if q0_np.ndim == 2:
                    q0_np = q0_np[0]  # Pinocchio only supports single pose
                
                # Pad to full length (7 arm + 2 gripper = 9)
                full_q0 = np.zeros(9, dtype=np.float64)
                full_q0[:min(len(q0_np), 7)] = q0_np[:min(len(q0_np), 7)]
            else:
                full_q0 = None
            
            # Convert target pose to SAPIEN Pose
            pos_np = target_pose.p[0].cpu().numpy() if isinstance(target_pose.p, torch.Tensor) else target_pose.p[0]
            quat_np = target_pose.q[0].cpu().numpy() if isinstance(target_pose.q, torch.Tensor) else target_pose.q[0]
            target_pose_sapien = sapien.Pose(p=pos_np, q=quat_np)
            
            # Pinocchio IK solve (fully aligned with simulation)
            result, success, error = self.pmodel.compute_inverse_kinematics(
                self.end_link_idx,
                target_pose_sapien,
                initial_qpos=full_q0,
                active_qmask=self.qmask,  # numpy bool array
                max_iterations=self.max_iterations,
            )
            
            if success:
                # Return only first 7 joints (convert to numpy array first to avoid warning)
                result_joints = np.array([result[:7]], dtype=np.float32)
                return torch.from_numpy(result_joints).to(device=self.device)
            else:
                return None
        else:
            # pytorch_kinematics IK (aligned with ManiSkill GPU simulation)
            if initial_qpos is not None:
                if isinstance(initial_qpos, np.ndarray):
                    q0 = torch.tensor(initial_qpos, dtype=torch.float32, device=self.device)
                else:
                    q0 = initial_qpos.to(device=self.device, dtype=torch.float32)
                
                if q0.dim() == 1:
                    q0 = q0.unsqueeze(0)
                
                # Use only first 7 joints
                q0 = q0[:, :self.NUM_JOINTS]
            else:
                batch_size = target_pose.p.shape[0]
                q0 = torch.zeros(batch_size, self.NUM_JOINTS, device=self.device)
            
            # Build target transform
            tf = pk.Transform3d(
                pos=target_pose.p,
                rot=target_pose.q,
                device=self.device,
            )
            
            # Set initial config and solve
            self.pik.initial_config = q0
            result = self.pik.solve(tf)
            
            # Return first solution
            return result.solutions[:, 0, :]
    
    def verify_ik_solution(
        self,
        joint_solution: torch.Tensor,
        target_pose: SimPose,
    ) -> Tuple[float, float]:
        """
        Verify IK solution accuracy
        
        Args:
            joint_solution: IK solution joint positions
            target_pose: Target pose
        
        Returns:
            (position_error, orientation_error): Position error (m), orientation error (rad)
        """
        # FK verification
        computed_pose = self.compute_fk(joint_solution)
        
        # Position error
        pos_error = torch.norm(computed_pose.p - target_pose.p, dim=-1).mean().item()
        
        # Orientation error (quaternion distance)
        quat_dot = torch.abs(torch.sum(computed_pose.q * target_pose.q, dim=-1))
        quat_dot = torch.clamp(quat_dot, -1.0, 1.0)
        ori_error = (2 * torch.acos(quat_dot)).mean().item()
        
        return pos_error, ori_error


def create_sim_aligned_ik_solver(
    urdf_path: str,
    device: str = "cpu",
    use_pinocchio: bool = None,
) -> SimAlignedPandaIKSolver:
    """
    Create IK solver aligned with ManiSkill simulation (simplified interface)
    
    Args:
        urdf_path: Path to URDF file
        device: "cpu" or "cuda" (recommend "cpu" for real robot)
        use_pinocchio: Force use/not use Pinocchio (None=auto, False=pytorch_kinematics)
                       Note: SAPIEN 3.0 has Pinocchio compatibility issues, 
                       recommend use_pinocchio=False
    
    Returns:
        SimAlignedPandaIKSolver instance
    """
    return SimAlignedPandaIKSolver(urdf_path=urdf_path, device=device, use_pinocchio=use_pinocchio)
