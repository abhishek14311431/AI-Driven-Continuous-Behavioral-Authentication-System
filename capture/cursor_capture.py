"""
STEP 1: Cursor Behavior Capture (NO ML)
========================================

This module captures raw cursor/touchpad movement data.
NO machine learning is performed here - only clean data collection.

Captures:
- Cursor position (x, y)
- Timestamp
- Movement speed
- Acceleration
- Idle gaps (time between movements)

All data is stored as raw CSV logs for later feature engineering.
"""

import time
import csv
import os
from typing import List, Optional
from dataclasses import dataclass
import platform

# Platform-specific imports (optional, fail gracefully if not available)
try:
    if platform.system() == "Windows":
        import win32api
    elif platform.system() == "Darwin":
        # macOS - using Quartz
        try:
            from Quartz import CGEventSourceSecondsSinceLastEventType, kCGEventSourceStateID
        except ImportError:
            pass
    else:
        # Linux
        try:
            import Xlib.display
        except ImportError:
            pass
except ImportError:
    pass


@dataclass
class CursorEvent:
    """
    Represents a single cursor movement event.
    This is RAW data - no ML processing.
    """
    timestamp: float      # Unix timestamp
    x: float              # X coordinate
    y: float              # Y coordinate
    dx: float             # Change in X (pixels)
    dy: float             # Change in Y (pixels)
    velocity: float       # Movement speed (pixels/second)
    acceleration: float   # Acceleration (pixels/second²)
    idle_gap: float       # Time since last movement (seconds)


class CursorCapture:
    """
    Captures cursor movement data for behavioral authentication.
    
    This is STEP 1 of the implementation - pure data collection.
    No feature engineering, no ML - just clean, stable raw data.
    """
    
    def __init__(self, log_file: str = "data/raw/cursor_logs.csv"):
        """
        Initialize cursor capture.
        
        Args:
            log_file: Path to CSV file for storing raw cursor events
        """
        self.log_file = log_file
        self.last_position: Optional[tuple] = None
        self.last_time: Optional[float] = None
        self.last_velocity: float = 0.0
        self.is_capturing: bool = False
        self.events_buffer: List[CursorEvent] = []
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def get_cursor_position(self) -> Optional[tuple]:
        """
        Get current cursor position (x, y).
        
        Returns:
            Tuple of (x, y) coordinates or None if error
        """
        try:
            if platform.system() == "Windows":
                return win32api.GetCursorPos()
            elif platform.system() == "Darwin":
                # macOS - using Quartz
                try:
                    from Quartz import CGEventCreate, kCGEventMouseMoved
                    event = CGEventCreate(None)
                    if event:
                        pos = event.location()
                        return (pos.x, pos.y)
                except:
                    # Fallback: use pyautogui or similar if available
                    try:
                        import pyautogui
                        return pyautogui.position()
                    except:
                        pass
            else:
                # Linux - using Xlib
                try:
                    display = Xlib.display.Display()
                    root = display.screen().root
                    pointer = root.query_pointer()
                    return (pointer.root_x, pointer.root_y)
                except:
                    # Fallback: use pyautogui
                    try:
                        import pyautogui
                        return pyautogui.position()
                    except:
                        pass
        except Exception as e:
            # Silent error handling - don't spam console
            return None
        
        return None
    
    def calculate_velocity(self, dx: float, dy: float, dt: float) -> float:
        """
        Calculate movement velocity (speed) in pixels per second.
        
        Args:
            dx: Change in X coordinate
            dy: Change in Y coordinate
            dt: Time delta in seconds
        
        Returns:
            Velocity in pixels/second
        """
        if dt <= 0:
            return 0.0
        
        # Euclidean distance
        distance = (dx**2 + dy**2)**0.5
        return distance / dt
    
    def calculate_acceleration(self, velocity: float, dt: float) -> float:
        """
        Calculate acceleration (change in velocity) in pixels/second².
        
        Args:
            velocity: Current velocity
            dt: Time delta in seconds
        
        Returns:
            Acceleration in pixels/second²
        """
        if dt <= 0:
            return 0.0
        
        return (velocity - self.last_velocity) / dt
    
    def capture_event(self) -> Optional[CursorEvent]:
        """
        Capture a single cursor movement event.
        
        Returns:
            CursorEvent if movement detected, None otherwise
        """
        current_pos = self.get_cursor_position()
        if current_pos is None:
            return None
        
        current_time = time.time()
        x, y = current_pos
        
        # First event - initialize but don't return (no movement yet)
        if self.last_position is None:
            self.last_position = (x, y)
            self.last_time = current_time
            return None
        
        # Calculate movement
        dx = x - self.last_position[0]
        dy = y - self.last_position[1]
        dt = current_time - self.last_time
        
        # Skip if no time has passed or no movement
        if dt <= 0:
            return None
        
        # Calculate idle gap (time since last movement)
        idle_gap = dt
        
        # Only create event if there's actual movement (avoid noise from tiny movements)
        movement_threshold = 0.1  # pixels - ignore movements smaller than this
        if abs(dx) < movement_threshold and abs(dy) < movement_threshold:
            # Update time but not position (treat as no movement)
            self.last_time = current_time
            return None
        
        # Calculate velocity and acceleration
        velocity = self.calculate_velocity(dx, dy, dt)
        acceleration = self.calculate_acceleration(velocity, dt)
        
        # Create event
        event = CursorEvent(
            timestamp=current_time,
            x=x,
            y=y,
            dx=dx,
            dy=dy,
            velocity=velocity,
            acceleration=acceleration,
            idle_gap=idle_gap
        )
        
        # Update state
        self.last_position = (x, y)
        self.last_time = current_time
        self.last_velocity = velocity
        
        return event
    
    def start_capture(self, interval: float = 0.01):
        """
        Start continuous cursor capture.
        
        Args:
            interval: Sampling interval in seconds (default: 0.01 = 100Hz)
        """
        self.is_capturing = True
        self.last_position = None
        self.last_time = None
        self.last_velocity = 0.0
        self.events_buffer.clear()
        
        # Initialize CSV file with headers
        try:
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'x',
                    'y',
                    'dx',
                    'dy',
                    'velocity',
                    'acceleration',
                    'idle_gap'
                ])
        except Exception as e:
            print(f"Error initializing cursor log file: {e}")
            return
        
        print(f"[OK] Cursor capture started. Logging to: {self.log_file}")
        print(f"  Sampling rate: {1.0/interval:.0f} Hz")
    
    def capture_loop(self, duration: Optional[float] = None, interval: float = 0.01):
        """
        Run capture loop for specified duration or until stopped.
        
        Args:
            duration: Duration to capture in seconds (None = until stopped)
            interval: Sampling interval in seconds
        """
        start_time = time.time()
        event_count = 0
        
        while self.is_capturing:
            event = self.capture_event()
            
            if event:
                self.events_buffer.append(event)
                self.save_event(event)
                event_count += 1
            
            # Check duration limit
            if duration and (time.time() - start_time) >= duration:
                break
            
            time.sleep(interval)
        
        print(f"Cursor capture stopped. Captured {event_count} events.")
    
    def save_event(self, event: CursorEvent):
        """
        Save a single cursor event to CSV file.
        
        Args:
            event: CursorEvent to save
        """
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    event.timestamp,
                    event.x,
                    event.y,
                    event.dx,
                    event.dy,
                    event.velocity,
                    event.acceleration,
                    event.idle_gap
                ])
        except Exception as e:
            print(f"Error saving cursor event: {e}")
    
    def stop_capture(self):
        """Stop cursor capture."""
        self.is_capturing = False
        print("Cursor capture stopped.")
    
    def get_recent_events(self, count: int = 100) -> List[CursorEvent]:
        """
        Get recent events from buffer.
        
        Args:
            count: Number of recent events to return
        
        Returns:
            List of recent CursorEvent objects
        """
        return self.events_buffer[-count:] if self.events_buffer else []
    
    def get_event_count(self) -> int:
        """Get total number of events captured."""
        return len(self.events_buffer)
    
    def clear_buffer(self):
        """Clear the events buffer (keeps file intact)."""
        self.events_buffer.clear()


# Test function for development
if __name__ == "__main__":
    print("Testing cursor capture...")
    print("Move your mouse/touchpad for 5 seconds...")
    
    capture = CursorCapture(log_file="test_cursor_logs.csv")
    capture.start_capture(interval=0.01)
    capture.capture_loop(duration=5.0, interval=0.01)
    capture.stop_capture()
    
    print(f"\nCaptured {capture.get_event_count()} events")
    print(f"Log file: {capture.log_file}")
