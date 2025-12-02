"""Gripper controller for Franka robot using franky library.

This module provides a simple interface for controlling the Franka gripper.
"""

import time
import threading
from typing import Optional
from franky import Gripper
from franky_control.robot.constants import FC

class GripperController:
    """
    Gripper controller wrapper for Franka gripper.
    
    Provides simplified interface for common gripper operations.
    Based on franky.Gripper API.
    
    Features:
    - Cached gripper state for fast non-blocking reads
    - Background thread updates state periodically
    """
    
    def __init__(self, robot_ip: str, cache_update_rate: float = 30.0):
        """
        Initialize gripper controller.
        
        Args:
            robot_ip: Robot IP address (same as robot control IP)
            cache_update_rate: Rate to update cached gripper state [Hz] (default: 30 Hz)
        """
        self.gripper = Gripper(robot_ip)
        self._is_homed = False
        self._max_width = FC.GRIPPER_MAX_WIDTH  # Default max width [m]
        
        # Cached state for fast non-blocking reads
        self._cached_width = 0.0
        self._cached_is_grasped = False
        self._cache_lock = threading.Lock()
        self._cache_update_interval = 1.0 / cache_update_rate
        
        # Background thread for state updates
        self._cache_thread = None
        self._cache_thread_running = False
    
    def start_state_cache(self):
        """Start background thread to cache gripper state."""
        if self._cache_thread is not None and self._cache_thread.is_alive():
            return  # Already running
        
        self._cache_thread_running = True
        self._cache_thread = threading.Thread(target=self._cache_update_loop, daemon=True)
        self._cache_thread.start()
        print(f"[Gripper] State cache started ({1.0/self._cache_update_interval:.0f} Hz)")
    
    def stop_state_cache(self):
        """Stop background state cache thread."""
        self._cache_thread_running = False
        if self._cache_thread is not None:
            self._cache_thread.join(timeout=1.0)
            self._cache_thread = None
            print("[Gripper] State cache stopped")
    
    def _cache_update_loop(self):
        """Background loop to update cached gripper state."""
        while self._cache_thread_running:
            try:
                state = self.gripper.state
                with self._cache_lock:
                    self._cached_width = state.width
                    self._cached_is_grasped = state.is_grasped
            except Exception as e:
                pass  # Silently ignore errors in background thread
            time.sleep(self._cache_update_interval)
        
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
                self._max_width = FC.GRIPPER_MAX_WIDTH  # Panda gripper max width
            except:
                pass
            print("[Gripper] Homing successful")
            # Auto-start state cache for fast reads
            self.start_state_cache()
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
    def width(self) -> float:
        """Get current gripper width [m] from cache (non-blocking).
        
        Returns:
            Current gripper finger width in meters (0.0 to 0.08).
            Uses cached value if cache thread is running, otherwise reads directly.
        """
        if self._cache_thread_running:
            with self._cache_lock:
                return self._cached_width
        # Fallback to direct read (blocking)
        state = self.gripper.state
        return state.width
    
    @property
    def width_sync(self) -> float:
        """Get current gripper width [m] directly (blocking).
        
        This always reads from the gripper directly, which can take 30-50ms.
        Use `width` property for fast cached reads.
        
        Returns:
            Current gripper finger width in meters (0.0 to 0.08).
        """
        state = self.gripper.state
        return state.width
    
    @property
    def state(self):
        """Get full gripper state.
        
        Returns:
            GripperState object with width, max_width, is_grasped, temperature
        """
        return self.gripper.state
    
    @property
    def is_grasped(self) -> bool:
        """Check if gripper is currently grasping an object (from cache).
        
        Returns:
            True if object is grasped, False otherwise.
            Uses cached value if cache thread is running, otherwise reads directly.
        """
        if self._cache_thread_running:
            with self._cache_lock:
                return self._cached_is_grasped
        # Fallback to direct read (blocking)
        state = self.gripper.state
        return state.is_grasped
    
    @property
    def is_homed(self) -> bool:
        """Check if gripper has been homed."""
        return self._is_homed


# Example usage
if __name__ == "__main__":
    import time
    
    # Initialize gripper
    gripper = GripperController(FC.ROBOT_IP)
    
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
