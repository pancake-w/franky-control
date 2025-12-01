"""VLA deployment modules for franky_control.

This module provides VLA (Vision-Language-Action) deployment scripts
that support both Cartesian pose control and IK-based joint control.

All default parameters (robot IP, control frequency, impedance values, etc.)
are centralized in franky_control.robot.constants.FC.

Usage:
    # Cartesian control (direct pose control)
    python -m franky_control.deploy.query_vla --help
    
    # IK-based joint control (simulation-aligned)
    python -m franky_control.deploy.query_vla_with_ik --help

Example:
    # Deploy with Cartesian control (uses default robot_ip from FC.ROBOT_IP)
    python -m franky_control.deploy.query_vla \\
        --instruction "pick up the red cube" \\
        --vla_server_ip "localhost" \\
        --vla_server_port 9876 \\
        --episode_idx "test_001"
    
    # Deploy with IK-based joint control
    python -m franky_control.deploy.query_vla_with_ik \\
        --instruction "pick up the red cube" \\
        --vla_server_ip "localhost" \\
        --vla_server_port 9876 \\
        --episode_idx "test_001" \\
        --verify_ik
"""

# Lazy imports to avoid loading heavy dependencies unless needed
def __getattr__(name):
    if name == "VLADeploy":
        from franky_control.deploy.query_vla import VLADeploy
        return VLADeploy
    elif name == "VLADeployWithIK":
        from franky_control.deploy.query_vla_with_ik import VLADeployWithIK
        return VLADeployWithIK
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'VLADeploy',
    'VLADeployWithIK',
]
