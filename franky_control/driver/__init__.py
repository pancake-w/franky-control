"""Input device drivers for data collection.

This module provides drivers for SpaceMouse and RealSense cameras.
"""

try:
    from .space_mouse import SpaceMouse
    SPACEMOUSE_AVAILABLE = True
except ImportError:
    SPACEMOUSE_AVAILABLE = False
    SpaceMouse = None

try:
    from .realsense_api import RealsenseAPI
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    RealsenseAPI = None

__all__ = [
    'SpaceMouse',
    'RealsenseAPI',
    'SPACEMOUSE_AVAILABLE',
    'REALSENSE_AVAILABLE',
]
