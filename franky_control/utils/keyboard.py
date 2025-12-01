"""Keyboard utilities for terminal-based control.

This module provides non-blocking keyboard input handling for control loops.
"""

import sys
import select
import termios
import tty
from typing import Optional


class KeyboardHandler:
    """Non-blocking keyboard input handler for control loops.
    
    Usage:
        keyboard = KeyboardHandler()
        keyboard.setup()
        try:
            while True:
                if keyboard.check_quit():
                    break
                # ... control loop
        finally:
            keyboard.restore()
    
    Or use as context manager:
        with KeyboardHandler() as kb:
            while not kb.check_quit():
                # ... control loop
    """
    
    def __init__(self):
        self._old_settings = None
    
    def setup(self):
        """Setup terminal for non-blocking keyboard input."""
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
    
    def restore(self):
        """Restore terminal settings."""
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None
    
    def check_quit(self, quit_key: str = 'q') -> bool:
        """Check if quit key was pressed (non-blocking).
        
        Args:
            quit_key: Key to check for quit signal (default 'q')
            
        Returns:
            True if quit key was pressed
        """
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            return key.lower() == quit_key.lower()
        return False
    
    def get_key(self) -> Optional[str]:
        """Get pressed key (non-blocking).
        
        Returns:
            Key character if pressed, None otherwise
        """
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None
    
    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.restore()
        return False
