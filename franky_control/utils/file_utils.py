"""File and directory utilities."""

import os
from typing import Optional


def ensure_dir(path: str) -> str:
    """Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        The path (for chaining)
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_next_episode_idx(base_dir: str) -> int:
    """Find next episode index by scanning existing episode directories.
    
    Args:
        base_dir: Base directory containing episode_* subdirectories
        
    Returns:
        Next available episode index (0 if directory empty or doesn't exist)
    """
    if not os.path.exists(base_dir):
        return 0
    
    episode_dirs = [
        item for item in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, item)) and item.startswith("episode_")
    ]
    
    if not episode_dirs:
        return 0
    
    episode_numbers = []
    for dir_name in episode_dirs:
        try:
            episode_numbers.append(int(dir_name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    
    return max(episode_numbers) + 1 if episode_numbers else 0
