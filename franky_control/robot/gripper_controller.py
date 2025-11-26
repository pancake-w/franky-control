"""Gripper controller for Franka robot using franky library.

This module provides a simple interface for controlling the Franka gripper.
"""

from typing import Optional
from franky import Gripper


class GripperController:
    """
    Gripper controller wrapper for Franka gripper.
    
    Provides simplified interface for common gripper operations.
    Based on franky.Gripper API.
    """
    
    def __init__(self, robot_ip: str):
        """
        Initialize gripper controller.
        
        Args:
            robot_ip: Robot IP address (same as robot control IP)
        """
        self.gripper = Gripper(robot_ip)
        self._is_homed = False
        self._max_width = 0.08  # Default max width [m]
        
    def home(self, async_call: bool = False) -> bool:
        """
        Home (calibrate) the gripper.
        
        This must be called before any other gripper operations.
        
        Args:
            async_call: If True, use asynchronous call (non-blocking)
        
        Returns:
            bool or BoolFuture: True if homing successful (sync), or BoolFuture (async)
        """
        print("[Gripper] Homing...")
        if async_call:
            return self.gripper.homing_async()
        
        success = self.gripper.homing()
        if success:
            self._is_homed = True
            # Update max width after homing
            try:
                # Note: franky.Gripper may not have max_width property
                # This is a placeholder
                self._max_width = 0.08  # Panda gripper max width
            except:
                pass
            print("[Gripper] Homing successful")
        else:
            print("[Gripper] Homing failed")
        return success
    
    def open(self, speed: float = 0.1, async_call: bool = False) -> bool:
        """
        Fully open the gripper.
        
        Args:
            speed: Opening speed [m/s] (max ~0.1 m/s)
            async_call: If True, use asynchronous call (non-blocking)
            
        Returns:
            bool or BoolFuture: True if successful (sync), or BoolFuture (async)
        """
        if not self._is_homed:
            print("[Gripper] Warning: Gripper not homed, calling home() first")
            self.home()
        
        if async_call:
            return self.gripper.open_async(speed=speed)
        return self.gripper.open(speed=speed)
    
    def close(self, speed: float = 0.1, force: float = 40.0, async_call: bool = False) -> bool:
        """
        Close the gripper (attempt to grasp).
        
        Args:
            speed: Closing speed [m/s]
            force: Grasping force [N] (1-70 N)
            async_call: If True, use asynchronous call (non-blocking)
            
        Returns:
            bool or BoolFuture: True if object grasped (sync), or BoolFuture (async)
        """
        if not self._is_homed:
            print("[Gripper] Warning: Gripper not homed, calling home() first")
            self.home()
        
        # Close to minimum width (0.0) with specified force
        if async_call:
            return self.gripper.grasp_async(
                width=0.0,
                speed=speed,
                force=force
            )
        return self.gripper.grasp(
            width=0.0,
            speed=speed,
            force=force
        )
    
    def move(self, width: float, speed: float = 0.05, async_call: bool = False) -> bool:
        """
        Move gripper to specified width.
        
        Args:
            width: Target width [m] (0.0 to ~0.08 m)
            speed: Movement speed [m/s]
            async_call: If True, use asynchronous call (non-blocking)
            
        Returns:
            bool or BoolFuture: True if successful (sync), or BoolFuture (async)
        """
        if not self._is_homed:
            print("[Gripper] Warning: Gripper not homed, calling home() first")
            self.home()
        
        # Clamp width to valid range
        width = max(0.0, min(width, self._max_width))
        
        if async_call:
            return self.gripper.move_async(width=width, speed=speed)
        return self.gripper.move(width=width, speed=speed)
    
    def grasp(
        self,
        width: float = 0.0,
        speed: float = 0.05,
        force: float = 40.0,
        epsilon_inner: float = 0.005,
        epsilon_outer: float = 0.005,
        async_call: bool = False
    ) -> bool:
        """
        Grasp an object with specified parameters.
        
        Args:
            width: Target grasp width [m] (minimum width to attempt)
            speed: Grasping speed [m/s]
            force: Grasping force [N] (1-70 N)
            epsilon_inner: Inner width tolerance [m]
            epsilon_outer: Outer width tolerance [m]
            async_call: If True, use asynchronous call (non-blocking)
            
        Returns:
            bool or BoolFuture: True if object successfully grasped (sync), or BoolFuture (async)
        """
        if not self._is_homed:
            print("[Gripper] Warning: Gripper not homed, calling home() first")
            self.home()
        
        if async_call:
            return self.gripper.grasp_async(
                width=width,
                speed=speed,
                force=force,
                epsilon_inner=epsilon_inner,
                epsilon_outer=epsilon_outer
            )
        return self.gripper.grasp(
            width=width,
            speed=speed,
            force=force,
            epsilon_inner=epsilon_inner,
            epsilon_outer=epsilon_outer
        )
    
    def stop(self, async_call: bool = False) -> bool:
        """
        Stop gripper motion immediately.
        
        Args:
            async_call: If True, use asynchronous call (non-blocking)
        
        Returns:
            bool or BoolFuture: True if successful (sync), or BoolFuture (async)
        """
        if async_call:
            return self.gripper.stop_async()
        return self.gripper.stop()
    
    @property
    def max_width(self) -> float:
        """Get maximum gripper width [m]."""
        return self._max_width
    
    @property
    def is_homed(self) -> bool:
        """Check if gripper has been homed."""
        return self._is_homed


# Example usage
if __name__ == "__main__":
    import time
    
    # Initialize gripper
    gripper = GripperController("172.16.0.2")
    
    # Home gripper
    if not gripper.home():
        print("Failed to home gripper")
        exit(1)
    
    # Open gripper
    print("Opening gripper...")
    gripper.open()
    time.sleep(1)
    
    # Move to specific width
    print("Moving to 0.04m...")
    gripper.move(0.04)
    time.sleep(1)
    
    # Close and grasp
    print("Attempting to grasp...")
    is_grasped = gripper.grasp(force=30.0)
    if is_grasped:
        print("Object grasped!")
    else:
        print("No object detected")
    
    time.sleep(2)
    
    # Open again
    print("Opening gripper...")
    gripper.open()
    
    print("Done!")
