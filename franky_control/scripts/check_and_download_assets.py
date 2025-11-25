"""
Check and download robot assets (URDF files) for franky_control.

This script downloads robot URDF and related files needed for kinematics.
"""
import os
import sys
import shutil
import requests
import zipfile
from tqdm import tqdm
from dataclasses import dataclass

# Get franky_control base path
FRANKY_CONTROL_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROBOT_URLS = {
    "panda": "https://cloud.tsinghua.edu.cn/f/402cfb147de84d1990d0/?dl=1",
}

ROBOT_CHECK_FILES = {
    "panda": "panda_v3.urdf",
}


@dataclass
class Args:
    robot_name: str = "panda"
    assets_base_dir: str = os.path.join(FRANKY_CONTROL_PATH, "..", "assets")


def safe_rmtree(path: str):
    """Safely remove directory tree."""
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def safe_remove(path: str):
    """Safely remove file."""
    if os.path.isfile(path):
        os.remove(path)


def get_dir_size(path: str) -> int:
    """Calculate the total size of a directory in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


def check_and_download_assets(
    robot_name: str = "panda",
    assets_base_dir: str = None
) -> str:
    """
    Check and download the assets directory for the specified robot.

    Args:
        robot_name: Robot name, e.g., 'panda'
        assets_base_dir: Base directory for storing assets

    Returns:
        str: Final robot directory path
    """
    if assets_base_dir is None:
        assets_base_dir = os.path.join(FRANKY_CONTROL_PATH, "..", "assets")
    
    assets_base_dir = os.path.abspath(assets_base_dir)
    
    if robot_name not in ROBOT_URLS:
        raise ValueError(
            f"Unknown robot '{robot_name}', please add the corresponding "
            f"download URL in ROBOT_URLS"
        )

    assets_target_dir = os.path.join(assets_base_dir, robot_name)
    tmp_dir = "./tmp_assets"
    tmp_zip_path = os.path.join(tmp_dir, f"{robot_name}.zip")
    tmp_extract_dir = os.path.join(tmp_dir, f"{robot_name}")

    # Step 1: Check if target directory already exists and is valid
    if os.path.isdir(assets_target_dir):
        dir_size_bytes = get_dir_size(assets_target_dir)
        check_file = os.path.join(assets_target_dir, ROBOT_CHECK_FILES[robot_name])
        
        # Check if size is greater than 1 MB and check file exists
        if dir_size_bytes > 1048576 and os.path.isfile(check_file):
            dir_size_mb = dir_size_bytes / (1024 * 1024)
            print(f"[✓] Directory found and valid (size: {dir_size_mb:.2f} MB): {assets_target_dir}")
            return assets_target_dir
        else:
            dir_size_mb = dir_size_bytes / (1024 * 1024)
            print(f"[!] Directory incomplete (size: {dir_size_mb:.2f} MB). Re-downloading.")
            safe_rmtree(assets_target_dir)

    # Ensure base directory exists
    os.makedirs(assets_base_dir, exist_ok=True)

    url = ROBOT_URLS[robot_name]

    try:
        # Step 2: Download zip file
        print(f"[↓] Downloading {robot_name} assets from {url}")

        # Clean up and recreate temporary directory
        safe_rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 KB
        
        with open(tmp_zip_path, "wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {robot_name}.zip"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                pbar.update(len(chunk))

        print(f"[✓] Download completed: {tmp_zip_path}")

        # Step 3: Extract to temporary directory
        print(f"[↓] Extracting to {tmp_extract_dir}")

        with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_extract_dir)
        print(f"[✓] Extraction completed")

        # Step 4: Find robot directory and move it
        robot_src = None
        for root, dirs, files in os.walk(tmp_extract_dir):
            if robot_name in dirs:
                robot_src = os.path.join(root, robot_name)
                break

        if robot_src is None:
            raise FileNotFoundError(
                f"{robot_name} directory not found after extraction"
            )

        shutil.move(robot_src, assets_target_dir)
        print(f"[✓] Moved {robot_name} assets to {assets_target_dir}")

    except Exception as e:
        print(f"[✗] Error: {e}")
        # Clean up residues
        safe_rmtree(tmp_dir)
        safe_rmtree(assets_target_dir)
        raise RuntimeError(
            f"Failed to download or install {robot_name} assets"
        ) from e

    finally:
        # Clean up temporary directory
        safe_rmtree(tmp_dir)

    return assets_target_dir


def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download robot assets for franky_control"
    )
    parser.add_argument(
        "--robot_name",
        type=str,
        default="panda",
        choices=list(ROBOT_URLS.keys()),
        help="Robot name (default: panda)"
    )
    parser.add_argument(
        "--assets_dir",
        type=str,
        default=None,
        help="Assets directory (default: <franky_control>/../assets)"
    )
    
    args = parser.parse_args()
    
    try:
        result_dir = check_and_download_assets(
            robot_name=args.robot_name,
            assets_base_dir=args.assets_dir
        )
        print(f"\n[✓] Success! Assets available at: {result_dir}")
        return 0
    except Exception as e:
        print(f"\n[✗] Failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
