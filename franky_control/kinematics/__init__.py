"""Kinematics module for Franka Panda robot."""

import os
import sys

# Path to assets
ASSETS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
PANDA_URDF_PATH = os.path.join(ASSETS_PATH, "panda", "panda_v3.urdf")

# Gripper constants
URDF_GRIPPER_OPEN = 0.04
URDF_GRIPPER_CLOSE = 0


def ensure_assets_downloaded(robot_name: str = "panda") -> str:
    """
    Ensure robot assets are downloaded.
    
    Args:
        robot_name: Robot name (default: "panda")
        
    Returns:
        Path to robot assets directory
    """
    assets_dir = os.path.join(ASSETS_PATH, robot_name)
    
    # Check if assets exist
    if robot_name == "panda":
        check_file = os.path.join(assets_dir, "panda_v3.urdf")
    else:
        check_file = assets_dir
    
    if os.path.exists(check_file):
        return assets_dir
    
    # Assets not found, try to download
    print(f"[INFO] Robot assets not found at {assets_dir}")
    print(f"[INFO] Attempting to download...")
    
    try:
        from franky_control.scripts.check_and_download_assets import check_and_download_assets
        result_dir = check_and_download_assets(robot_name, ASSETS_PATH)
        print(f"[INFO] Assets downloaded successfully to {result_dir}")
        return result_dir
    except ImportError as e:
        print(f"[ERROR] Failed to import download script: {e}")
        print(f"[INFO] Please run manually:")
        print(f"  python -m franky_control.scripts.check_and_download_assets --robot_name {robot_name}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to download assets: {e}")
        print(f"[INFO] Please run manually:")
        print(f"  python -m franky_control.scripts.check_and_download_assets --robot_name {robot_name}")
        sys.exit(1)
