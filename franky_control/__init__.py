"""
Franky Control - High-level control and data collection tools for Franka robots.

This package provides:
- Robot control examples (realtime pose control, joint control, etc.)
- Data collection utilities for robot learning
- Hardware drivers (SpaceMouse, RealSense cameras)
- Kinematics solvers (IK/FK using PyTorch and SAPIEN)

Installation:
    pip install -e .                    # Basic installation
    pip install -e ".[all]"             # Full installation with all features
    pip install -e ".[spacemouse]"      # SpaceMouse support
    pip install -e ".[realsense]"       # RealSense camera support
    pip install -e ".[ik]"              # IK solver support

Usage:
    python -m franky_control.examples.realtime_pose_control
    python -m franky_control.examples.reset_arm
    python -m franky_control.data_collection.data_collection_with_ik --help
"""

__version__ = "0.1.0"
__author__ = "Bingwen Wei"
__email__ = "pancake.wbw@gmail.com"

# Lazy imports to avoid loading heavy dependencies unless needed
def __getattr__(name):
    if name == "RealsenseAPI":
        from franky_control.driver.realsense_api import RealsenseAPI
        return RealsenseAPI
    elif name == "SpaceMouse":
        from franky_control.driver.space_mouse import SpaceMouse
        return SpaceMouse
    elif name == "SimAlignedPandaIKSolver":
        from franky_control.kinematics.panda_ik_solver import SimAlignedPandaIKSolver
        return SimAlignedPandaIKSolver
    elif name == "create_sim_aligned_ik_solver":
        from franky_control.kinematics.panda_ik_solver import create_sim_aligned_ik_solver
        return create_sim_aligned_ik_solver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    "RealsenseAPI",
    "SpaceMouse", 
    "SimAlignedPandaIKSolver",
    "create_sim_aligned_ik_solver",
]
